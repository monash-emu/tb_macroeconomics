import sys
import numpyro

from tb_macro.constants import (
    OUTPUT_PATH,
    ISO3,
    START_TIME,
    END_TIME,
    AGE_STRATA,
    SOLVER_KWARGS,
    N_CHAINS_REMOTE,
)

# Must run before JAX is imported (including via tb_macro / numpyro.infer).
numpyro.set_host_device_count(N_CHAINS_REMOTE)

from numpyro import distributions as dist
from numpyro import infer
from jax.random import PRNGKey
import jax
import arviz as az
from datetime import datetime, UTC

from tb_macro.utils import get_logger
from tb_macro.parameters import BASE_PARAMS, PARAM_BOUNDS
from tb_macro.inputs import load_demography, load_fertility, load_who_outcomes
from tb_macro.demography import prepare_pop_data_for_entries
from tb_macro.epi import get_base_model, add_flows_to_model, initialise_pops
from tb_macro.calibration import make_log_likelihood


if __name__ == "__main__":
    task = sys.argv[1]
    n_runs = int(sys.argv[2])
    path = OUTPUT_PATH / task
    path.mkdir(parents=True, exist_ok=True)
    logger = get_logger(path / "run.log")

    group_popsize, death_rates, age_weights = load_demography(ISO3)
    fert_padded = load_fertility(ISO3)
    tsr, death_in_unsucc, who_mort = load_who_outcomes(ISO3)
    epi_model, disease_state, age_strat, clin_strat, infect_strat = get_base_model(START_TIME, END_TIME)
    start_apops = [1000.0] * len(AGE_STRATA)
    entry_times, entry_rates = prepare_pop_data_for_entries(group_popsize, START_TIME, sum(start_apops))
    add_flows_to_model(
        epi_model, 
        disease_state,
        age_strat,
        clin_strat,
        infect_strat,
        age_weights,
        group_popsize,
        fert_padded,
        death_rates,
        tsr,
        death_in_unsucc,
        entry_times,
        entry_rates,
    )
    initialise_pops(epi_model, disease_state, age_strat, start_apops)

    priors = {k: dist.Uniform(v[0], v[1]) for k, v in PARAM_BOUNDS.items()}
    log_like = make_log_likelihood(epi_model, disease_state, age_strat, infect_strat, SOLVER_KWARGS, who_mort)

    def model():
        params = BASE_PARAMS | {k: numpyro.sample(k, v) for k, v in priors.items()}
        ll = log_like(params)
        numpyro.factor("ll", ll)

    kernel = infer.NUTS(model, max_tree_depth=5, init_strategy=infer.init_to_median())
    mcmc = infer.MCMC(
        kernel,
        num_warmup=n_runs,
        num_samples=n_runs,
        num_chains=N_CHAINS_REMOTE,
    )
    mcmc.run(PRNGKey(2))
    idata = az.from_numpyro(mcmc)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%MZ")
    idata.to_netcdf(path / f"{timestamp}.nc")
