import sys
import numpyro

from tb_macro.constants import (
    OUTPUT_PATH,
    ISO3,
    START_TIME,
    END_TIME,
    CALIB_END_TIME,
    OUTPUT_TIME_STEP,
    AGE_STRATA,
    SOLVER_KWARGS,
    N_CHAINS_REMOTE,
    N_SAMPLE_CHUNK,
    N_OUTPUT_SAMPLES,
    SCENARIO_PARAMS,
)

# Must run before JAX is imported (including via tb_macro / numpyro.infer).
numpyro.set_host_device_count(N_CHAINS_REMOTE)

from numpyro import distributions as dist
from numpyro import infer
import arviz as az
import pandas as pd
from jax.random import PRNGKey
import gc
import jax
from datetime import datetime, UTC

from tb_macro.utils import get_logger, get_git_provenance, write_run_log
from tb_macro.parameters import BASE_PARAMS, PARAM_BOUNDS, get_nuts_init_values
from tb_macro.inputs import load_demography, load_fertility, load_who_outcomes
from tb_macro.demography import prepare_pop_data_for_entries
from tb_macro.epi import get_base_model, add_flows_to_model, initialise_pops
from tb_macro.targets import (
    NOTIF_TARGET,
    LATENT_TARGET,
    PULM_PREV_TARGET,
    INF_PREV_TARGET,
    PREV_DECLINE_TARGET,
)
from tb_macro.calibration import (
    MCMC_EXTRA_FIELDS,
    inference_data_from_mcmc,
    make_log_likelihood,
)
from tb_macro.outputs import (
    get_posterior_samples,
    rerun_model_for_outputs,
    save_sampled_outputs,
)


if __name__ == "__main__":
    # Usage: calibrate.py <task> <n_runs> [<task to resume>]
    # When resuming, output keeps going to the original task's directory
    task = sys.argv[1]
    n_runs = int(sys.argv[2])
    resume_task = sys.argv[3] if len(sys.argv) > 3 else None
    path = OUTPUT_PATH / (resume_task or task)
    path.mkdir(parents=True, exist_ok=True)
    logger = get_logger(path / "run.log")
    if resume_task:
        logger.info("job %s resuming task %s", task, resume_task)
    provenance = get_git_provenance()
    logger.info(
        "git commit %s (dirty=%s) branch %s",
        provenance["commit"],
        provenance["dirty"],
        provenance["branch"],
    )

    group_popsize, death_rates, age_weights = load_demography(ISO3)
    fert_padded = load_fertility(ISO3)
    tsr, death_in_unsucc, who_mort = load_who_outcomes(ISO3)
    start_apops = [1000.0] * len(AGE_STRATA)
    entry_times, entry_rates = prepare_pop_data_for_entries(group_popsize, START_TIME, sum(start_apops))

    def build_model(end_time):
        spec = get_base_model(START_TIME, end_time)
        add_flows_to_model(
            spec.epi_model,
            spec.disease_state,
            spec.age_strat,
            spec.clin_strat,
            spec.infect_strat,
            age_weights,
            group_popsize,
            fert_padded,
            death_rates,
            tsr,
            death_in_unsucc,
            entry_times,
            entry_rates,
        )
        initialise_pops(spec.epi_model, spec.disease_state, spec.age_strat, start_apops)
        return spec

    # Calibration only solves as far as the targets; interp_annual_output clamps
    # silently past the last model time, so check every target is covered
    last_target = max(
        NOTIF_TARGET.index.max(),
        who_mort.index.max(),
        LATENT_TARGET.index.max(),
        PULM_PREV_TARGET.index.max(),
        INF_PREV_TARGET.index.max(),
        PREV_DECLINE_TARGET.index.max(),
    )
    assert last_target <= CALIB_END_TIME - OUTPUT_TIME_STEP, (
        f"Target at {last_target} beyond calibration end {CALIB_END_TIME}"
    )
    calib_spec = build_model(CALIB_END_TIME)
    logger.info("calibration solve %s to %s", START_TIME, CALIB_END_TIME)

    priors = {k: dist.Uniform(v[0], v[1]) for k, v in PARAM_BOUNDS.items()}
    log_like = make_log_likelihood(
        calib_spec.epi_model,
        calib_spec.disease_state,
        calib_spec.age_strat,
        calib_spec.infect_strat,
        SOLVER_KWARGS,
        who_mort,
        age_weights,
        group_popsize,
        fert_padded,
    )

    def model():
        params = BASE_PARAMS | {k: numpyro.sample(k, v) for k, v in priors.items()}
        ll = log_like(params)
        numpyro.factor("ll", ll)

    kernel = infer.NUTS(
        model,
        max_tree_depth=5,
        init_strategy=infer.init_to_value(values=get_nuts_init_values()),
    )
    # Warmup runs as one block (its adaptation windows depend on its total
    # length), then sampling runs in chunks; after warmup and after every
    # chunk the sampler state and draws so far are saved, so a timed-out job
    # can be resumed by passing its task id as the third argument
    ckpt_path = path / "checkpoint.pkl"
    partial_path = path / "partial.nc"
    if resume_task:
        if not ckpt_path.exists():
            raise FileNotFoundError(f"No checkpoint in {path}; warmup never finished, start afresh")
        ckpt = pd.read_pickle(ckpt_path)
        if ckpt["n_runs"] != n_runs:
            logger.warning("n_runs %s differs from checkpoint, using %s", n_runs, ckpt["n_runs"])
            n_runs = ckpt["n_runs"]
        state, n_done = ckpt["state"], ckpt["n_done"]
        idata = az.from_netcdf(partial_path).load() if n_done else None
        logger.info("resumed from checkpoint with %s of %s draws done", n_done, n_runs)
    chunk = min(N_SAMPLE_CHUNK, n_runs)
    if n_runs % chunk:
        raise ValueError(f"n_runs {n_runs} must be a multiple of the chunk size {chunk}")
    mcmc = infer.MCMC(
        kernel,
        num_warmup=n_runs,
        num_samples=chunk,
        num_chains=N_CHAINS_REMOTE,
        progress_bar=True,
    )

    def save_checkpoint(state, n_done):
        tmp = ckpt_path.with_suffix(".tmp")
        pd.to_pickle({"state": jax.device_get(state), "n_done": n_done, "n_runs": n_runs}, tmp)
        tmp.replace(ckpt_path)

    if not resume_task:
        mcmc.warmup(PRNGKey(2), collect_warmup=False)
        state, n_done, idata = mcmc.post_warmup_state, 0, None
        save_checkpoint(state, n_done)
        logger.info("warmup done, checkpoint saved")

    while n_done < n_runs:
        mcmc.post_warmup_state = state
        mcmc.run(state.rng_key, extra_fields=MCMC_EXTRA_FIELDS)
        state = mcmc.last_state
        new = inference_data_from_mcmc(mcmc)
        idata = new if idata is None else az.concat(idata, new, dim="draw", reset_dim=True)
        n_done += chunk
        tmp = partial_path.with_suffix(".tmp")
        idata.to_netcdf(tmp)
        tmp.replace(partial_path)
        save_checkpoint(state, n_done)
        logger.info("%s of %s draws done, checkpoint saved", n_done, n_runs)

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%MZ")
    idata.to_netcdf(path / f"{timestamp}.nc")
    partial_path.unlink()
    ckpt_path.unlink()
    logger.info("Wrote %s", f"{timestamp}.nc")

    # Remote NUTS's XLAs from memory
    del mcmc, kernel, log_like, model, calib_spec
    jax.clear_caches()
    gc.collect()

    # Full-horizon model for the scenario projections
    epi_model, disease_state, age_strat, clin_strat, infect_strat = build_model(END_TIME)

    samples = get_posterior_samples(idata, N_OUTPUT_SAMPLES)
    outputs, sample_labels = rerun_model_for_outputs(
        epi_model,
        age_strat,
        disease_state,
        clin_strat,
        infect_strat,
        idata,
        SCENARIO_PARAMS,
        samples,
    )
    save_sampled_outputs(path / f"{timestamp}_outputs.pkl", outputs, sample_labels)
    write_run_log(
        path / f"{timestamp}.log",
        extra={
            "task": resume_task or task,
            "final_job": task,
            "n_runs": n_runs,
            "idata": f"{timestamp}.nc",
            "outputs": f"{timestamp}_outputs.pkl",
            "scenario_params": SCENARIO_PARAMS,
        },
    )
    logger.info("Wrote %s, %s and %s", f"{timestamp}.nc", f"{timestamp}_outputs.pkl", f"{timestamp}.log")
