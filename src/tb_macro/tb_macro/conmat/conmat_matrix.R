iso3 <- "KIR"

library(conmat)
setwd("/Users/jtrauer/dev/tb_macroeconomics/src/tb_macro/tb_macro/conmat/")
polymod_contact_data <- get_polymod_contact_data(setting = "all")
polymod_survey_data <- get_polymod_population()
contact_model <- fit_single_contact_model(
  contact_data = polymod_contact_data,
  population = polymod_survey_data
)
pop_df <- read.csv(paste0(iso3, "_pop_2025.csv"))
population <- conmat_population(pop_df, "lower.age.limit", "population")
synthetic_predictions <- predict_contacts(
  model = contact_model,
  population = population,
  age_breaks = c(pop_df$lower.age.limit, Inf)
)
write.csv(synthetic_predictions, paste0("conmat_all_", iso3, ".csv"))
# matrix <- predictions_to_matrix(synthetic_predictions)
# autoplot(matrix)