import pandas as pd
from jax import numpy as jnp
from jax import jit
from numpyro import distributions as dist
import diffrax as dfx

from summer3.epi import CompartmentalEpiModel, Stratification

from tb_macro.constants import (
    AGE_STRATA,
    COUNT_LOG_SD,
    INFECTED_STATES,
    PROP_LOGIT_SD,
    YOUNG_END_AGE,
)
from tb_macro.targets import (
    NOTIF_TARGET,
    LATENT_TARGET,
    PULM_PREV_TARGET,
    INF_PREV_TARGET,
    PREV_DECLINE_TARGET,
)
from tb_macro.parameters import BASE_PARAMS

_EPS = 1e-32


def _logit(x):
    """Transform a proportion onto the log-odds scale.

    Args:
        x: The proportion to transform, clipped away from zero and one

    Returns:
        The log-odds of the input
    """
    x = jnp.clip(x, _EPS, 1.0 - _EPS)
    return jnp.log(x) - jnp.log1p(-x)


def _log_normal_log_prob(modelled, target, sd=COUNT_LOG_SD):
    """Compare a count or ratio against its target on the log scale.

    Args:
        modelled: The modelled values
        target: The observation to compare against
        sd: The standard deviation of the comparison in log space

    Returns:
        The log-densities of the comparisons
    """
    modelled = jnp.maximum(modelled, _EPS)
    target = jnp.maximum(jnp.asarray(target), _EPS)
    return dist.Normal(jnp.log(target), sd).log_prob(jnp.log(modelled))


def _logit_normal_log_prob(modelled, target, sd=PROP_LOGIT_SD):
    """Compare a proportion against its target on the log-odds scale.

    Args:
        modelled: The modelled proportion
        target: The observed proportion to compare against
        sd: The standard deviation of the comparison on the log-odds scale

    Returns:
        The log-densities of the comparisons
    """
    return dist.Normal(_logit(jnp.asarray(target)), sd).log_prob(_logit(modelled))


def get_latent_log_likelihood(results: dict, disease_state: Stratification):
    r"""Get the likelihood contribution from the prevalence of Mtb infection.

    Args:
        results: The outputs of a single model run
        disease_state: The compartmental stratification object

    Returns:
        The log-likelihood contribution

    Notes:
    -----
    The proportion of the population ever infected with _Mtb_ was compared
    against the estimate from the tuberculin survey reported by Marks et al.
    (Bulletin of the World Health Organization).

    The modelled equivalent is everyone not previously infected with 
    _Mtb_, which is represented by all modelled compartments other than
    _Mtb_ naive. That is, the {{INFECTED_STATES}} states, 
    divided by the total population at the time of the survey.
    All modelled age groups contributed to both the numerator and the denominator.

    Being a proportion, this quantity was compared on the log-odds scale with
    a standard deviation of {{PROP_LOGIT_SD}}.
    """
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


def get_notification_log_likelihood(results: dict):
    r"""Get the likelihood contribution from TB case notifications.

    Args:
        results: The outputs of a single model run

    Returns:
        The log-likelihood contribution

    Notes:
    -----
    Modelled case detections were compared against the notifications
    obtained from the Vietnam National Tuberculosis Program. These are
    reported by calendar year, and so are offset by
    {{CALENDAR_YEAR_MIDPOINT}} of a year to sit at mid-year in model time
    (because we consider whole numbers of years in modelled time 
    to represent the starts and ends of years).

    As counts, notifications are compared on the log scale with a
    standard deviation of {{COUNT_LOG_SD}}, which relates the difference
    to the size of the target, rather than being absolute.
    The log-density is averaged rather than summed over the years of data, so
    that this multi-year series contributes comparable weight to the likelihood
    as the single-point targets.
    """
    notif = (
        results["flows"]["detection"]
        .query(time=NOTIF_TARGET.index)
        .sum(to_dims="time")
        .data
    )
    return _log_normal_log_prob(notif, NOTIF_TARGET.to_numpy()).mean()


def get_death_log_likelihood(results: dict, who_mort: pd.Series):
    r"""Get the likelihood contribution from TB deaths.

    Args:
        results: The outputs of a single model run
        who_mort: The WHO estimates of annual TB deaths

    Returns:
        The log-likelihood contribution

    Notes:
    -----
    Modelled TB deaths are compared against the WHO estimates of TB
    mortality for {{ISO3}}. Deaths occurring in the community and during
    treatment are summed before comparison, because the estimates do not
    distinguish between them.

    As for notifications, these counts are compared on the log scale with a
    standard deviation of {{COUNT_LOG_SD}}, and averaged over the years for
    which estimates are available.
    """
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


def get_adult_pulm_prev(
    results: dict,
    disease_state: Stratification,
    age_strat: Stratification,
    infect_strat: Stratification,
    target_time: float,
):
    r"""Get the quantities needed for the adult prevalence-based targets.

    Args:
        results: The outputs of a single model run
        disease_state: The compartmental stratification object
        age_strat: The age stratification object
        infect_strat: The infectiousness stratification object
        target_time: The time at which to evaluate prevalence

    Returns:
        The highly infectious population size,
            the prevalent population size and
            the total adult population size

    Notes:
    -----
    Prevalence is calculated to approximate the quantity ascertained by a
    bacteriologically confirmed prevalence survey, and so is restricted to
    adults, taken here as those aged {{YOUNG_END_AGE}} years and over.

    Three groups contribute to the numerator: all those with active disease
    in the high infectiousness stratum, a fraction of those in the low
    infectiousness stratum given by the "{{prop_lowinf_bactpos}}", and
    everyone currently receiving treatment. Clinical status does not
    enter this calculation, such that subclinical disease contributes 
    on the same basis as clinical disease.

    The denominator is the total adult population at the same time point.
    """
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


def get_pulm_prev_log_likelihood(
    results: dict,
    disease_state: Stratification,
    age_strat: Stratification,
    infect_strat: Stratification,
):
    r"""Get the likelihood contribution from adult pulmonary TB prevalence.

    Args:
        results: The outputs of a single model run
        disease_state: The compartmental stratification object
        age_strat: The age stratification object
        infect_strat: The infectiousness stratification object

    Returns:
        The log-likelihood contribution

    Notes:
    -----
    Modelled adult prevalence of bacteriologically confirmed pulmonary TB was
    compared against the second Vietnamese national prevalence survey
    (PLOS One). The target is published per 100,000 population and converted
    to a proportion of the adult population.

    Being a proportion, this quantity was compared on the log-odds scale with
    a standard deviation of {{PROP_LOGIT_SD}}.
    """
    target_time = PULM_PREV_TARGET.index[0]
    target_val = PULM_PREV_TARGET.iloc[0] / 1e5
    _, pulm_prev, adult_pop = get_adult_pulm_prev(
        results, disease_state, age_strat, infect_strat, target_time
    )
    prev_prop = pulm_prev / (adult_pop + _EPS)
    return _logit_normal_log_prob(prev_prop.data[0], target_val)


def get_prev_decline_log_likelihood(
    results: dict,
    disease_state: Stratification,
    age_strat: Stratification,
    infect_strat: Stratification,
):
    r"""Get the likelihood contribution from the decline in TB prevalence
    over serial prevalence surveys.

    Args:
        results: The outputs of a single model run
        disease_state: The compartmental stratification object
        age_strat: The age stratification object
        infect_strat: The infectiousness stratification object

    Returns:
        The log-likelihood contribution

    Notes:
    -----
    The two survey rounds reported by Nguyen et al. (Emerging Infectious
    Diseases) are used to target the decline in prevalence rather than absolute values.
    Only the ratio of the later to the earlier estimate is taken, 
    so that the target constrains the trend in prevalence while remaining
    insensitive to any discrepancy between the quantity these surveys
    assess and our definition of prevalence.

    The modelled ratio applies the same adult prevalence definition at both
    time points. The ratio is strictly positive, so it is compared on the
    log scale with a standard deviation of {{COUNT_LOG_SD}}.
    """
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


def get_infprop_log_likelihood(
    results: dict,
    disease_state: Stratification,
    age_strat: Stratification,
    infect_strat: Stratification,
):
    r"""Get the likelihood contribution from the infectiousness of prevalent TB.

    Args:
        results: The outputs of a single model run
        disease_state: The compartmental stratification object
        age_strat: The age stratification object
        infect_strat: The infectiousness stratification object

    Returns:
        The log-likelihood contribution

    Notes:
    -----
    The proportion of prevalent adult TB that is highly infectious was
    compared against the equivalent proportion from the second Vietnamese
    national prevalence survey.

    The numerator is the high infectiousness stratum of the active
    compartment and the denominator is total adult prevalence, as defined
    above. This target therefore constrains how prevalent disease is
    distributed across the infectiousness strata, without further
    constraining the overall size of the prevalent pool.

    As a proportion, this quantity is compared on the log-odds scale with
    a standard deviation of {{PROP_LOGIT_SD}}.
    """
    target_time = INF_PREV_TARGET.index[0]
    target_val = INF_PREV_TARGET.iloc[0]
    high_inf, pulm_prev, _ = get_adult_pulm_prev(
        results, disease_state, age_strat, infect_strat, target_time
    )
    inf_prop = high_inf / (pulm_prev + _EPS)
    return _logit_normal_log_prob(inf_prop.data[0], target_val)


def make_log_likelihood(
    epi_model: CompartmentalEpiModel,
    disease_state: Stratification,
    age_strat: Stratification,
    infect_strat: Stratification,
    solver_kwargs: dict,
    who_mort: pd.Series,
):
    r"""Build the log-likelihood function used to calibrate the model.

    Args:
        epi_model: The epidemiological model to run
        disease_state: The compartmental stratification object
        age_strat: The age stratification object
        infect_strat: The infectiousness stratification object
        solver_kwargs: Arguments passed through to the ODE solver
        who_mort: The WHO estimates of annual TB deaths

    Returns:
        A function mapping a parameter set to its log-likelihood

    Notes:
    -----
    The log-likelihood is the sum of six contributions, taken as
    independent: the prevalence of _Mtb_ infection, case notifications, TB
    deaths, adult bacteriologically-confirmed prevalence, the decline in
    prevalence between the two prevalence survey rounds,
    and the proportion of prevalent disease that is highly infectious.

    Two error models are applied throughout. Counts and ratios are compared
    on the log scale, as
    $\log \hat{y} \sim \mathcal{N}(\log y, \sigma_{c})$ with $\sigma_{c}$ of
    {{COUNT_LOG_SD}}, so that the discrepancy scales with the magnitude of
    the target. Proportions are compared on the log-odds scale, as
    $\mathrm{logit}(\hat{p}) \sim \mathcal{N}(\mathrm{logit}(p), \sigma_{p})$
    with $\sigma_{p}$ of {{PROP_LOGIT_SD}}, which respects their bounds at
    zero and one. In both cases the transformed target provides the mean of
    the distribution and the transformed modelled value is evaluated against
    it, which is equivalent to the reverse for these symmetric
    distributions.

    Where a target spans multiple years, the log-density is averaged over
    those years, so that each of the six contributions carries comparable
    weight irrespective of how many observations it contains.

    Parameter sets for which the solver does not reach a successful solution
    are assigned a large negative log-likelihood, so that they are rejected
    rather than contributing invalid model output to the likelihood.
    """

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
