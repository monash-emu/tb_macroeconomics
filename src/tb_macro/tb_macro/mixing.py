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


def build_s_matrix_single_age(
    fert: jnp.array,
    fert_ends: jnp.array,
    time: float,
    bg_mixing: float,
    a_spread: float,
    pc_strength: float,
) -> jnp.array:

    """Construct the full single-age transmission kernel matrix.

    This function builds an unweighted contact kernel at single-year age resolution,
    representing the per-pair intensity of interaction between individuals of
    each pair of ages.

    The kernel combines three components:
    - Background (uniform) mixing
    - Assortative mixing based on age similarity
    - Parent-child mixing based on fertility-derived age gaps

    Importantly, this matrix does *not* include any information about the population
    distribution. Weighting by age structure is applied later when aggregating to
    age groups.

    Args:
        fert: The fertility data (padded with zeroes)
        fert_ends: The start and finish of the fertility index
        time: Model time
        bg_mixing: Background mixing value
        a_spread: Decay parameter for assortative mixing
        pc_strength: Scaling parameter for parent-child contacts

    Returns:
        (MAX_AGE + 1) x (MAX_AGE + 1) unweighted transmission kernel
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

    # Combine components (weights applied later)
    return bg_mixing + assort_mat + child_parent_mat


def get_full_normalised_within_age_band_weights(
    current_weights: jnp.array,
) -> jnp.array:
    """Embed within-group age distributions into full age space.

    Constructs a matrix W where each row corresponds to an age group and contains
    the distribution of single-year ages within that group, with zeros elsewhere.

    Each row therefore represents a probability distribution over ages conditional
    on belonging to that age group (therefore summing to one).

    This allows group-level interactions to be computed by projecting the full
    single-age kernel into group space via matrix multiplication.

    Args:
        current_weights: Array of length (MAX_AGE + 1) giving the relative population
            weights of each single age, assumed to already be normalised within each age 

    Returns:
        The weight matrix
    """
    w_group = jnp.zeros((len(AGE_STRATA), MAX_AGE + 1))
    for a, lower in enumerate(AGE_STRATA):
        upper = MAX_AGE + 1 if lower == AGE_STRATA[-1] else AGE_STRATA[a + 1]
        w_group = w_group.at[a, lower:upper].set(current_weights[lower:upper])
    return w_group


def aggregate_full_matrix_to_groups(
    full_kernel: jnp.array,
    current_weights: jnp.array,
) -> jnp.array:
    """Aggregate a single-age transmission kernel to age-group level.

    This function combines the detailed single-age kernel with the age structure
    of each group to produce a smaller matrix defined over age groups.

    The process works by:
    - Representing each age group as a distribution over single-year ages
    (a row in the weight matrix)
    - Using these distributions to take weighted averages across the full kernel

    Each entry in the resulting matrix can be interpreted as:
    the expected interaction intensity between a randomly chosen individual
    from one age group (row) and a randomly chosen individual from another
    age group (column), based on their underlying age distributions.

    Args:
        full_kernel: (MAX_AGE + 1) x (MAX_AGE + 1) unweighted kernel
        current_weights: Weight distribution across all single ages

    Returns:
        len(AGE_STRATA) x len(AGE_STRATA) weighted group transmission matrix
    """
    w_group = get_full_normalised_within_age_band_weights(current_weights)
    return w_group @ full_kernel @ w_group.T


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
    """Construct the group-level mixing matrix S.

    This function:
    1. Retrieves the within-group age distributions for the current time
    2. Builds the single-age interaction kernel
    3. Aggregates to age-group level via weighted projection

    The resulting matrix represents per-pair interaction intensities between
    age groups, independent of population sizes.

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

    # Compute full single-age transmission kernel (unweighted)
    full_kernel = build_s_matrix_single_age(fert, fert_ends, time, bg_mixing, a_spread, pc_strength)

    # Aggregate to group level with weighting
    return aggregate_full_matrix_to_groups(full_kernel, current_weights)


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
    """Construct the population-scaled contact matrix C.

    This rescales the per-pair interaction matrix S to account for population sizes,
    producing a matrix C suitable for use in frequency-dependent transmission models.

    Each column is multiplied by the population size of the contacting/infecting group.

    This represents the total rate at which individuals in the group represented
    by the matrix rows encounter individuals from the groups represented by the columns.

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
    s_mat = build_s_matrix(weights, weight_ends, fert, fert_ends, time, bg_mixing, a_spread, pc_strength)
    return pops[None, :] * s_mat


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
    """Normalise the contact matrix by its spectral radius.

    Constructs the population-scaled contact matrix and rescales it so that its
    dominant eigenvalue (spectral radius) is equal to 1.

    This is useful for calibrating transmission models, as the spectral radius of
    the "c" matrix is directly related to the basic reproduction number under homogeneous
    infectiousness assumptions.

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
