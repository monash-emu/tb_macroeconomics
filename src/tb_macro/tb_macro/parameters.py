BASE_PARAMS = {
    "raw_transmission_rate": 7.0,
    "bg_mixing": 0.029,  # U(0.01, 0.05)
    "a_spread": 9.83,  # U(5.0, 15.0)
    "pc_strength": 0.99,  # U(1.0, 2.0)
    "rel_sus_mtb_naive": 1.0,  # Required, but should remain one by definition
    "rel_sus_contained": 0.3,  # U(0.2, 0.5)
    "rel_sus_cleared": 0.7,  # U(0.5, 1.0)
    "rel_sus_children": 0.5,  # U(0.5, 1.0)
    "rel_infectiousness_subclin": 0.5,
    "rel_infectiousness_lowinf": 0.4,
    "progression_rate_age0": 2.4,
    "progression_rate_age5": 2.0,
    "progression_rate_age15": 0.1,
    "progression_prop_infectious": 0.5,
    "containment_rate_age0": 4.4,
    "containment_rate_age5": 4.4,
    "containment_rate_age15": 2.0,
    "breakdown_rate": 0.57,  # U(0.01, 1.0)
    "clearance_rate": 0.056,  # U(0.01, 0.1)
    "clinical_progression_rate": 2.28,  # U(0.5, 5.0)
    "clinical_regression_rate": 1.0,
    "infectiousness_gain_rate": 2.8,  # U(0.5, 5.0)
    "infectiousness_loss_rate": 1.0,
    "tb_mortality_rate_inf": 0.389,
    "tb_mortality_rate_lowinf": 0.025,
    "self_recovery_rate": 0.4,
    "detect_rate_current": 0.8,
    "rel_detect_2010": 0.8,
    "rel_detect_1986": 0.5,
    "detect_gap_reduction": 0.0,  # Intervention-related only
    "rx_duration": 0.5,  # Should remain at this value
    "seed_peak_time": 1830.0,
    "seed_duration": 10.0,
    "seed_peak_rate": 0.01,
}
