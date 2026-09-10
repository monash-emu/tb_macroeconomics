from jax import numpy as jnp

from summer3.graph import Parameter, Time, defer
from summer3.epi import CompartmentalEpiModel, Stratification, TransitionFlow

from tb_macro.constants import AGE_STRATA, ACF_MIN_AGE
from tb_macro.utils import get_scale_data, get_cos_multicurve


def get_acf_screen_rate(coverage):
    """Convert annual screening coverage into a continuous-time
    screening rate.

    Args:
        coverage: Annual proportion of the population screened

    Returns:
        Screening rate per year
    """
    return -jnp.log(1.0 - coverage)


def add_acf(
    epi_model: CompartmentalEpiModel,
    disease_state: Stratification,
    age_strat: Stratification,
    infect_strat: Stratification,
    start_time: float,
):
    r"""Add the active case finding process to the model.

    Args:
        epi_model: The epidemiological model to add the flows to
        disease_state: The compartmental stratification object
        age_strat: The age stratification object
        infect_strat: The infectiousness stratification object
        start_time: The model starting time as a calendar year

    Notes:
    -----
    Active case finding screens people with active TB aged
    "{{ACF_MIN_AGE}}" years and over, and transitions detected
    cases into treatment. Unlike routine detection, this
    includes subclinical disease.

    The peak screening rate is $-\ln(1 - c)$, where $c$ is
    "{{acf_coverage}}". This converts annual coverage into a
    hazard over time. Detection then scales that rate by a
    stratum-specific Xpert Ultra sensitivity: "{{acf_sensitivity_high}}"
    in the high infectiousness stratum, and "{{acf_sensitivity_low}}"
    in the low infectiousness stratum. The low-infectious rate is
    further multiplied by the "{{prop_lowinf_bactpos}}".

    These sensitivities are taken from Zifodya et al. (Cochrane
    Database of Systematic Reviews, 2021) on Xpert Ultra for
    pulmonary TB in adults. The high-infectious value is the
    smear-positive estimate; the low-infectious value is the
    smear-negative estimate, applied only to the bacteriologically
    detectable fraction of that stratum.

    The rate is zero until the "{{acf_start}}", then follows a
    cosine-smoothed scale-up over "{{acf_scaling_time}}" years
    to its peak rate as defined above, remains at the peak
    through the "{{acf_duration}}", and cosine-smooths back to
    zero over a further "{{acf_scaling_time}}" years.
    """
    peak_screen_rate = defer(get_acf_screen_rate)(
        Parameter("acf_coverage", 0.0),
    )

    sim_time = Time + start_time

    def acf_screen_curve(t, t_start, scale_time, duration, peak_rate):
        times = get_scale_data(
            jnp.array(
                [
                    t_start,
                    t_start + scale_time,
                    t_start + duration,
                    t_start + duration + scale_time,
                ]
            )
        )
        vals = get_scale_data(jnp.array([0.0, peak_rate, peak_rate, 0.0]))
        return get_cos_multicurve(t, times, vals)

    screen_func = defer(acf_screen_curve)(
        sim_time,
        Parameter("acf_start", 0.0),
        Parameter("acf_scaling_time", 0.0),
        Parameter("acf_duration", 0.0),
        peak_screen_rate,
    )

    def acf_infect_rates(screen_rate, s_low, s_high, prop_bactpos):
        return infect_strat.categories().wrap(
            jnp.array([screen_rate * s_low * prop_bactpos, screen_rate * s_high])
        )

    detect_rate = defer(acf_infect_rates)(
        screen_func,
        Parameter("acf_sensitivity_low", 0.0),
        Parameter("acf_sensitivity_high", 0.0),
        Parameter("prop_lowinf_bactpos", 0.0),
    )

    source = disease_state["active"]
    dest = disease_state["treatment"]
    detect = TransitionFlow("acf", source, dest, detect_rate)
    detect.adjustments_source.append(
        age_strat.categories().wrap(
            jnp.where(jnp.array(AGE_STRATA) >= ACF_MIN_AGE, 1.0, 0.0)
        )
    )
    epi_model.add_flow(detect)
