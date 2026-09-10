# TODO

## CXR then Xpert screening

Add a CXR+Xpert algorithm alongside the current sputum NAAT-only process.

CXR is a rule-out before confirmatory Xpert, so cascade sensitivity cannot exceed NAAT-only among people who would have been Xpert-positive. The extra misses are mainly smear-negative, culture-positive disease that is CXR-negative, i.e. the low-infectious bacteriologically detectable group. High-infectious / smear-positive disease is rarely CXR-negative, so that sensitivity can stay close to the NAAT-only value.

A workable implementation is a pair of CXR sensitivities (high- and low-infectious) that multiply the existing Ultra parameters, with the low-infectious `prop_lowinf_bactpos` factor unchanged. Correlation between CXR abnormality and bacillary load should be kept in mind: treating the two tests as independent would understate cascade sensitivity.

## Intervention parameter uncertainty

The ACF parameters (coverage, duration, start, scale-up, and the two Ultra sensitivities) are currently fixed point values. We need a way to vary all intervention-related parameters across their uncertainty ranges when producing scenario outputs, separately from the calibrated posterior.

## Outputs vs moving model code

Calibration `.nc` files and output pickles are produced by whatever code was current at run time. Notebooks such as `05-Outputs.ipynb` then rebuild the model from present source and either inspect the posterior or re-simulate with `rerun_model_for_outputs`. Structural changes (new parameters, different ACF rates, age restrictions) make those two states diverge: old posteriors may not share parameter names with `BASE_PARAMS`, and running an old sample through new flows is not the simulation that was calibrated.

Each saved run now writes a `{timestamp}.log` beside the `.nc` with git commit, dirty flag, and output filenames. That is only a label. Later options, once we decide:

- Prefer the pickled indicator outputs when the question is what that run said, rather than re-simulating
- Refuse to re-simulate if HEAD does not match the log (a dirty tree makes the hash a warning, not a guarantee)
- Commit or tag before production runs
- Longer term, store enough of the parameter vector and scenario settings that a run is self-describing
