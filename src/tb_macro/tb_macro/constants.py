# Compartments for TB model
NAIVE_COMP = ["mtb_naive"]
LATENT_COMPS = ["incipient", "contained", "cleared"]
ACTIVE_COMPS = ["active"]
POST_DETECT_COMPS = ["treatment", "recovered"]

# Compartments that can be reinfected
INFECT_COMPS = ["mtb_naive", "contained", "cleared", "recovered"]

# Age-related
AGE_STRATA = ["0", "5", "15"]