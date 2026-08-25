from jax import numpy as jnp
from jax import jit
from numpyro import distributions as dist
import diffrax as dfx

from summer3.epi import CompartmentalModelODE, build_istate, dti_to_epoch

from tb_macro.constants import AGE_STRATA, INFECTED_STATES, YOUNG_END_AGE
from tb_macro.targets import NOTIF_TARGET, LATENT_TARGET, PREV_TARGET
from tb_macro.parameters import BASE_PARAMS


def get_latent_log_likelihood(results, disease_state):
    target_time = LATENT_TARGET.index[0]
    target_val = LATENT_TARGET.iloc[0] / 1e2
    latent = (
        results["compartments"]
        .query(compartment=disease_state[INFECTED_STATES], time=target_time)
        .sum(to_dims="time")
    )
    total = results["compartments"].query(time=target_time).sum(to_dims="time")
    latent_prop = latent / (total + 1e-32)
    return dist.Normal(target_val, 0.05).log_prob(latent_prop.data[0])


def get_notification_log_likelihood(results):
    notif = (
        results["flows"]["detection"]
        .query(time=NOTIF_TARGET.index)
        .sum(to_dims="time")
        .data
    )
    return dist.Normal(NOTIF_TARGET.to_numpy(), 5e3).log_prob(notif).mean()


def get_death_log_likelihood(results, who_mort):
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
    return dist.Normal(who_mort.to_numpy(), 5e3).log_prob(deaths).mean()


def get_prevalence_log_likelihood(results, disease_state, age_strat, infect_strat):
    adult_ages = age_strat[[str(a) for a in AGE_STRATA if a >= YOUNG_END_AGE]]
    target_time = PREV_TARGET.index[0]
    target_val = PREV_TARGET.iloc[0] / 1e5
    high_inf = (
        results["compartments"]
        .query(compartment=(infect_strat["high"], adult_ages), time=target_time)
        .sum(to_dims="time")
    )
    low_inf = (
        results["compartments"]
        .query(compartment=(infect_strat["low"], adult_ages), time=target_time)
        .sum(to_dims="time")
    )
    on_rx = (
        results["compartments"]
        .query(
            compartment=(disease_state["treatment"], adult_ages),
            time=target_time,
        )
        .sum(to_dims="time")
    )
    adult_pop = (
        results["compartments"]
        .query(compartment=adult_ages, time=target_time)
        .sum(to_dims="time")
    )
    prev_prop = (
        high_inf + low_inf * BASE_PARAMS["prop_lowinf_bactpos"] + on_rx
    ) / (adult_pop + 1e-32)
    return dist.Normal(target_val, 0.0005).log_prob(prev_prop.data[0])


def make_log_likelihood(
    epi_model,
    disease_state,
    age_strat,
    infect_strat,
    solver_kwargs,
    who_mort,
):
    @jit
    def get_log_likelihood(params):
        results = epi_model.run(params, solver_kwargs=solver_kwargs)
        ll = (
            get_latent_log_likelihood(results, disease_state)
            + get_notification_log_likelihood(results)
            + get_death_log_likelihood(results, who_mort)
            + get_prevalence_log_likelihood(
                results, disease_state, age_strat, infect_strat
            )
        )
        return jnp.where(
            results["aux"].result == dfx._solution.RESULTS.successful, ll, -1e10
        )

    return get_log_likelihood
