import numpy as np
from jax import numpy as jnp

from summer3.graph import Parameter, Time, defer
from summer3.epi import CompartmentalEpiModel, Stratification, TransitionFlow

from tb_macro.utils import get_four_element_multicurve


def get_acf_rate(coverage, sensitivity):
    """Convert annual screening coverage and diagnostic sensitivity
    into a continuous-time detection rate.

    Args:
        coverage: Annual proportion of the population screened
        sensitivity: Probability that screening detects a true case

    Returns:
        Detection rate per year
    """
    return -jnp.log(1.0 - coverage) * sensitivity


def add_acf(
    epi_model: CompartmentalEpiModel,
    disease_state: Stratification,
    start_time: float,
):
    r"""Add the active case finding process to the model.

    Args:
        epi_model: The epidemiological model to add the flows to
        disease_state: The compartmental stratification object
        start_time: The model starting time as a calendar year

    Notes:
    -----
    Active case finding screens people with active TB and transitions
    detected cases into treatment. Unlike routine detection, this
    includes subclinical disease.

    The peak rate is given by $-\ln(1 - c) \times s$, where $c$ is
    "{{acf_coverage}}" and $s$ is "{{acf_sensitivity}}".
    This converts annual coverage into a hazard over time,
    and then scales this rate by diagnostic sensitivity.

    The rate is zero until the "{{acf_start}}", rises over
    the course of "{{acf_scaling_time}}" years
    to its peak rate as defined above, 
    remains at the peak for the "{{acf_duration}}", 
    and then returns to zero.
    """
    peak_rate = defer(get_acf_rate)(
        Parameter("acf_coverage", 0.0),
        Parameter("acf_sensitivity", 0.0),
    )

    sim_time = Time + start_time
    
    detect_func = defer(get_four_element_multicurve)(
        sim_time,
        Parameter("acf_start", 0.0),
        0.0,
        Parameter("acf_start", 0.0) + Parameter("acf_scaling_time", 0.0),
        peak_rate,
        Parameter("acf_start", 0.0) + Parameter("acf_duration", 0.0),
        peak_rate,
        Parameter("acf_start", 0.0) + Parameter("acf_duration", 0.0) + Parameter("acf_scaling_time", 0.0),
        0.0,
    )

    source = disease_state["active"]
    dest = disease_state["treatment"]
    detect = TransitionFlow("acf", source, dest, detect_func)
    epi_model.add_flow(detect)
