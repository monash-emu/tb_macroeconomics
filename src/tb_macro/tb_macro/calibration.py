from jax import numpy as jnp
from jax import jit
from numpyro import distributions as dist
import diffrax as dfx

from tb_macro.constants import AGE_STRATA, INFECTED_STATES, YOUNG_END_AGE
from tb_macro.targets import (
    NOTIF_TARGET,
    LATENT_TARGET,
    PULM_PREV_TARGET,
    INF_PREV_TARGET,
    PREV_DECLINE_TARGET,
)
from tb_macro.parameters import BASE_PARAMS

COUNT_LOG_SD = 0.1
PROP_LOGIT_SD = 0.2
_EPS = 1e-32


def _logit(x):
    x = jnp.clip(x, _EPS, 1.0 - _EPS)
    return jnp.log(x) - jnp.log1p(-x)


def _log_normal_log_prob(modelled, target, sd=COUNT_LOG_SD):
    modelled = jnp.maximum(modelled, _EPS)
    target = jnp.maximum(jnp.asarray(target), _EPS)
    return dist.Normal(jnp.log(target), sd).log_prob(jnp.log(modelled))


def _logit_normal_log_prob(modelled, target, sd=PROP_LOGIT_SD):
    return dist.Normal(_logit(jnp.asarray(target)), sd).log_prob(_logit(modelled))


def get_latent_log_likelihood(results, disease_state):
    target_time = LATENT_TARGET.index[0]
    target_val = LATENT_TARGET.iloc[0] / 1e2
    latent = (
        results["compartments"]
        .query(compartment=disease_state[INFECTED_STATES], time=target_time)
        .sum(to_dims="time")
    )
    total = results["compartments"].query(time=target_time).sum(to_dims="time")
    latent_prop = latent / (total + _EPS)
    return _logit_normal_log_prob(latent_prop.data[0], target_val)


def get_notification_log_likelihood(results):
    notif = (
        results["flows"]["detection"]
        .query(time=NOTIF_TARGET.index)
        .sum(to_dims="time")
        .data
    )
    return _log_normal_log_prob(notif, NOTIF_TARGET.to_numpy()).mean()


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
    return _log_normal_log_prob(deaths, who_mort.to_numpy()).mean()


def get_adult_pulm_prev(results, disease_state, age_strat, infect_strat, target_time):
    adult_ages = age_strat[[str(a) for a in AGE_STRATA if a >= YOUNG_END_AGE]]
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
    pulm_prev = high_inf + low_inf * BASE_PARAMS["prop_lowinf_bactpos"] + on_rx
    return high_inf, pulm_prev, adult_pop


def get_pulm_prev_log_likelihood(results, disease_state, age_strat, infect_strat):
    target_time = PULM_PREV_TARGET.index[0]
    target_val = PULM_PREV_TARGET.iloc[0] / 1e5
    _, pulm_prev, adult_pop = get_adult_pulm_prev(
        results, disease_state, age_strat, infect_strat, target_time
    )
    prev_prop = pulm_prev / (adult_pop + _EPS)
    return _logit_normal_log_prob(prev_prop.data[0], target_val)


def get_prev_decline_log_likelihood(results, disease_state, age_strat, infect_strat):
    decline_target = PREV_DECLINE_TARGET.sort_index()
    base_time, end_time = decline_target.index
    target_decline = decline_target.iloc[1] / decline_target.iloc[0]
    _, pulm_end, pop_end = get_adult_pulm_prev(
        results, disease_state, age_strat, infect_strat, end_time
    )
    _, pulm_base, pop_base = get_adult_pulm_prev(
        results, disease_state, age_strat, infect_strat, base_time
    )
    prev_end = pulm_end / (pop_end + _EPS)
    prev_base = pulm_base / (pop_base + _EPS)
    predicted_decline = prev_end / (prev_base + _EPS)
    return _log_normal_log_prob(predicted_decline.data[0], target_decline)


def get_infprop_log_likelihood(results, disease_state, age_strat, infect_strat):
    target_time = INF_PREV_TARGET.index[0]
    target_val = INF_PREV_TARGET.iloc[0]
    high_inf, pulm_prev, _ = get_adult_pulm_prev(
        results, disease_state, age_strat, infect_strat, target_time
    )
    inf_prop = high_inf / (pulm_prev + _EPS)
    return _logit_normal_log_prob(inf_prop.data[0], target_val)


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
            + get_pulm_prev_log_likelihood(
                results, disease_state, age_strat, infect_strat
            )
            + get_prev_decline_log_likelihood(
                results, disease_state, age_strat, infect_strat
            )
            + get_infprop_log_likelihood(
                results, disease_state, age_strat, infect_strat
            )
        )
        return jnp.where(
            results["aux"].result == dfx._solution.RESULTS.successful, ll, -1e10
        )

    return get_log_likelihood
