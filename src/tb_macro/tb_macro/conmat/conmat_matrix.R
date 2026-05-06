args <- commandArgs(trailingOnly = TRUE)
iso3 <- args[1]
year <- args[2]
dir <- args[3]

setwd(dir)

library(conmat)
polymod_contact_data <- get_polymod_contact_data(setting = "all")
polymod_survey_data <- get_polymod_population()
contact_model <- fit_single_contact_model(
  contact_data = polymod_contact_data,
  population = polymod_survey_data
)
pop_df <- read.csv(paste0(iso3, "_pop_", year, ".csv"))
population <- conmat_population(pop_df, "age", "population")
synthetic_predictions <- predict_contacts(
  model = contact_model,
  population = population,
  age_breaks = c(pop_df$age, Inf)
)
write.csv(synthetic_predictions, paste0("conmat_all_", iso3, ".csv"))
# matrix <- predictions_to_matrix(synthetic_predictions)
# autoplot(matrix)