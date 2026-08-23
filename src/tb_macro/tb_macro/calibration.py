from jax import numpy as jnp
from jax import jit
from numpyro import distributions as dist
import diffrax as dfx

from summer3.epi import CompartmentalModelODE, build_istate, dti_to_epoch

from tb_macro.constants import INFECTED_STATES, PREV_STATES
from tb_macro.targets import NOTIF_TARGET, LATENT_TARGET


def make_log_likelihood(
    epi_model,
    disease_state,
    infect_strat,
    solver_kwargs,
    who_mort,
):

    @jit
    def get_log_likelihood(params):
        results = epi_model.run(params, solver_kwargs=solver_kwargs)

        # Latent target
        latent_date = LATENT_TARGET.index[0]
        latent_target_val = LATENT_TARGET.iloc[0] / 1e2
        latent = (
            results["compartments"]
            .query(compartment=disease_state[INFECTED_STATES], time=latent_date)
            .sum(to_dims="time")
        )
        total = results["compartments"].query(time=latent_date).sum(to_dims="time")
        latent_prop = latent / (total + 1e-32)
        latent_ll = dist.Normal(latent_target_val, 0.05).log_prob(latent_prop.data[0])

        # Notification target
        notif = (
            results["flows"]["detection"]
            .query(time=NOTIF_TARGET.index)
            .sum(to_dims="time")
            .data
        )
        notif_ll = dist.Normal(NOTIF_TARGET.to_numpy(), 5e3).log_prob(notif).mean()

        # Deaths target
        community_deaths = (
            results["flows"]["tb_mortality"]
            .query(time=who_mort.index)
            .sum(to_dims="time")
            .data
        )
        rx_deaths = (
            results["flows"]["rx_death"]
            .query(time=who_mort.index)
            .sum(to_dims="time")
            .data
        )
        deaths = community_deaths + rx_deaths
        death_ll = dist.Normal(who_mort.to_numpy(), 5e3).log_prob(deaths).mean()

        # Prevalence target
        prev_time = 2017.0
        prev_targ = 322.0 / 1e5
        high_inf = results["compartments"].query(
            compartment=infect_strat["high"], 
            time=prev_time,
        ).sum(to_dims="time").data
        low_inf = results["compartments"].query(
            compartment=infect_strat["low"], 
            time=prev_time,
        ).sum(to_dims="time").data
        prev_val = high_inf + low_inf * 2.0 / 3.0
        prev_ll = dist.Normal(prev_targ, 0.0005).log_prob(prev_val[0])

        ll = latent_ll + notif_ll + death_ll + prev_ll

        return jnp.where(
            results["aux"].result == dfx._solution.RESULTS.successful, ll, -1e10
        )

    return get_log_likelihood


def get_runner(epi_model):
    istate = build_istate(epi_model.cmap, epi_model.base_pops, epi_model.pop_splits)
    cmodel = CompartmentalModelODE(epi_model.cmap, epi_model.flows)
    runner = cmodel.get_runner(
        len(epi_model.times), dti_to_epoch(epi_model.times), True
    )
    return runner, istate
