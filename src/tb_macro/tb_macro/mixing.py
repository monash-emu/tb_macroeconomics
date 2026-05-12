import pandas as pd
from jax import numpy as jnp

from tb_macro.constants import AGE_STRATA, MAX_AGE


def get_year_index(
    ends: jnp.array,
    time: float,
) -> jnp.array:
    """Get the relevant row index from a former dataframe
    with consecutive index values representing years
    given a time/year input.

    Args:
        ends: The values representing the start and finish of the
            former dataframe's index
        time: Model time

    Returns:
        The relevant row of data
    """
    clamped_time = jnp.clip(time, ends[0], ends[1])
    return (clamped_time - ends[0]).astype(jnp.int32)


def build_full_s_matrix_single_age(
    current_weights: jnp.array,
    fert: jnp.array,
    fert_ends: jnp.array,
    time: float,
    bg_mixing: float,
    a_spread: float,
    pc_strength: float,
) -> jnp.array:
    """Construct the full single-age transmission matrix.

    Computes the transmission kernel at single-year age resolution
    before aggregation into groups.

    Args:
        current_weights: Weight distribution across all single ages
        fert: The fertility data (padded with zeroes)
        fert_ends: The start and finish of the fertility index
        time: Model time
        bg_mixing: Background mixing value
        a_spread: Decay parameter for assortative mixing
        pc_strength: Scaling parameter for parent-child contacts

    Returns:
        (MAX_AGE + 1) x (MAX_AGE + 1) transmission matrix
    """
    ages = jnp.arange(MAX_AGE + 1)

    # Assortative component: depends on age difference only
    age_diff_mat = jnp.abs(ages[:, None] - ages[None, :])
    assort_mat = (1.0 / a_spread) * jnp.exp(-age_diff_mat / a_spread)

    # Child-parent component: depends on fertility and age gap
    age_gap_mat = jnp.abs(ages[:, None] - ages[None, :]).astype(jnp.int32)
    child_age_mat = jnp.minimum(ages[:, None], ages[None, :])
    child_birth_years = time - child_age_mat
    clamped_birth_years = get_year_index(fert_ends, child_birth_years)
    child_parent_mat = pc_strength * fert[clamped_birth_years, age_gap_mat]

    # Weight outer product and combine components
    weight_prod = current_weights[:, None] * current_weights[None, :]
    full_mat = bg_mixing + weight_prod * (assort_mat + child_parent_mat)

    return full_mat


def aggregate_full_matrix_to_groups(
    full_mat: jnp.array,
) -> jnp.array:
    """Aggregate single-age transmission matrix to group-level matrix.

    Sums blocks of the full single-age matrix according to AGE_STRATA
    boundaries to produce the group-level transmission matrix.

    Args:
        full_mat: (MAX_AGE + 1) x (MAX_AGE + 1) single-age matrix

    Returns:
        len(AGE_STRATA) x len(AGE_STRATA) group transmission matrix
    """
    n_groups = len(AGE_STRATA)
    s_matrix = jnp.zeros((n_groups, n_groups))

    for i, lower_i in enumerate(AGE_STRATA):
        upper_i = MAX_AGE + 1 if lower_i == AGE_STRATA[-1] else AGE_STRATA[i + 1]

        for j, lower_j in enumerate(AGE_STRATA[: i + 1]):
            upper_j = MAX_AGE + 1 if lower_j == AGE_STRATA[-1] else AGE_STRATA[j + 1]

            # Sum the block from the full matrix
            block_value = jnp.sum(full_mat[lower_i:upper_i, lower_j:upper_j])
            s_matrix = s_matrix.at[i, j].set(block_value)
            s_matrix = s_matrix.at[j, i].set(block_value)

    return s_matrix


def build_s_matrix(
    weights: jnp.array,
    weight_ends: jnp.array,
    fert: jnp.array,
    fert_ends: jnp.array,
    time: float,
    bg_mixing: float,
    a_spread: float,
    pc_strength: float,
) -> jnp.array:
    """Construct the symmetric s_matrix matrix.

    Computes transmission kernels at single-age resolution and aggregates
    results to group level for efficiency and clarity.

    Args:
        weights: Within age brackets weight by age group and year
        weight_ends: The start and finish of the weight index
        fert: The fertility data (padded with zeroes)
        fert_ends: The start and finish of the fertility index
        time: Model time
        bg_mixing: Background mixing value
        a_spread: Decay parameter for assortative mixing
        pc_strength: Scaling parameter for the strength of parent-child contacts

    Returns:
        The s_matrix matrix (n_groups x n_groups)
    """
    year_idx = get_year_index(weight_ends, time)
    current_weights = weights[year_idx, :]

    # Compute full single-age transmission matrix
    full_mat = build_full_s_matrix_single_age(
        current_weights, fert, fert_ends, time, bg_mixing, a_spread, pc_strength
    )

    # Aggregate to group level
    s_matrix = aggregate_full_matrix_to_groups(full_mat)

    return s_matrix


def build_c_matrix(
    weights: jnp.array,
    weight_ends: jnp.array,
    pops: jnp.array,
    pop_ends: jnp.array,
    fert: jnp.array,
    fert_ends: jnp.array,
    time: float,
    bg_mixing: float,
    a_spread: float,
    pc_strength: float,
) -> jnp.array:
    """Get the C matrix, being the per capita
    or frequency-dependent transmission matrix
    from the per capita, per capita or
    density-dependent transmission matrix.
    Note that the [None, :] is not strictly necessary
    but makes it clearer that this is row vector.

    Args:
        weights: Within age brackets weight by age group and year
        weight_ends: The start and finish of the weight index
        pops: Population sizes by age group
        pop_ends: The start and finish of the population index
        fert: The fertility data (padded with zeroes)
        fert_ends: The start and finish of the fertility index
        time: Model time
        bg_mixing: Background mixing value
        a_spread: Decay parameter for assortative mixing
        pc_strength: Scaling parameter for the strength of parent-child contacts

    Returns:
        The C matrix
    """
    year_idx = get_year_index(pop_ends, time)
    pops = pops[year_idx, :]
    return pops[None, :] * build_s_matrix(
        weights, weight_ends, fert, fert_ends, time, bg_mixing, a_spread, pc_strength
    )


def get_norm_c_matrix(
    weights: jnp.array,
    weight_ends: jnp.array,
    pops: jnp.array,
    pop_ends: jnp.array,
    fert: jnp.array,
    fert_ends: jnp.array,
    time: float,
    bg_mixing: float,
    a_spread: float,
    pc_strength: float,
) -> jnp.array:
    """Get the normalised version of the per capita
    mixing matrix created by build_c_matrix.

    Args:
        weights: Within age brackets weight by age group and year
        weight_ends: The start and finish of the weight index
        pops: Population sizes by age group
        pop_ends: The start and finish of the population index
        fert: The fertility data (padded with zeroes)
        fert_ends: The start and finish of the fertility index
        time: Model time
        bg_mixing: Background mixing value
        a_spread: Decay parameter for assortative mixing
        pc_strength: Scaling parameter for the strength of parent-child contacts

    Returns:
        The normalised C matrix
    """
    c_matrix = build_c_matrix(
        weights,
        weight_ends,
        pops,
        pop_ends,
        fert,
        fert_ends,
        time,
        bg_mixing,
        a_spread,
        pc_strength,
    )
    eigvals = jnp.linalg.eigvals(c_matrix)
    spectral_radius = jnp.max(jnp.abs(eigvals))
    return c_matrix / spectral_radius
