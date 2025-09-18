library(tidyverse)
library(tidymodels)
library(vroom)
library(glmnet)

# Cleaning data
bike <- vroom("~/Documents/STAT 348/KaggleBikeShare/bike-sharing-demand/train.csv")
trainData <- bike |>
  select(-casual, -registered) |>
  mutate(count = log(count))

# Read in test data set
testData <- vroom("~/Documents/STAT 348/KaggleBikeShare/bike-sharing-demand/test.csv")

# Defining a recipe
bike_recipe <- recipe(count ~ ., data = trainData) |>
  step_mutate(weather = ifelse(weather == 4, 3, weather)) |>
  step_mutate(weather = factor(weather, 
                               levels = c(1, 2, 3), 
                               labels = c("clear", "cloudy", "precipitation"))) |>
  step_time(datetime, features = c("hour")) |>
  step_rm(datetime) |>
  step_mutate(season = factor(season, 
                              levels = c(1, 2, 3, 4),
                              labels = c("spring", "summer", "fall", "winter"))) |>
  step_corr(all_numeric_predictors(), threshold = 0.7) |>
  step_dummy(all_nominal_predictors()) |>
  step_normalize(all_numeric_predictors())

## Linear Regression Model with Recipe
lin_model <- linear_reg() |>
  set_engine("lm") |>
  set_mode("regression")

bike_workflow <- workflow() |>
  add_recipe(bike_recipe) |>
  add_model(lin_model) |>
  fit(data = trainData)

bike_predictions <- predict(bike_workflow, new_data = testData) |>
  mutate(count = exp(count)) # back-transform log(count) prediction

## Penalized Regression Model
preg_model <- linear_reg(penalty= tune(), mixture= tune()) |>
  set_engine("glmnet")

## Set Workflow
preg_wf <- workflow() |>
  add_recipe(bike_recipe) |>
  add_model(preg_model)

## Grid of values to tune over
grid_of_tuning_params <- grid_regular(penalty(),
                                      mixture(),
                                      levels = 5)

## Split data for CV
folds <- vfold_cv(trainData, v = 5, repeats = 1)

## Run the CV
CV_results <- preg_wf |>
  tune_grid(resamples = folds,
            grid = grid_of_tuning_params,
            metrics = metric_set(rmse, mae))

##Find Best Tuning Parameters
bestTune <- CV_results |>
  select_best(metric="rmse")

## Finalize the Workflow & fit it
final_wf <- preg_wf |>
  finalize_workflow(bestTune) |>
  fit(data = trainData)

## Predict
preds <- predict(final_wf, new_data = testData)

## Format the Predictions for Submission to Kaggle
kaggle_submission <- preds |>
  bind_cols(testData |> select(datetime)) |>
  transmute(
    datetime = format(datetime, "%Y-%m-%d %H:%M:%S"),
    count = pmax(0, exp(.pred))
  )

## Write out the file
vroom_write(kaggle_submission,
            file = "~/Documents/STAT 348/KaggleBikeShare/penalized_tuning.csv",
            delim = ",")

# Print first 5 rows of baked data set
bike_prep <- prep(bike_recipe)
baked_test <- bake(bike_prep, new_data = testData)
head(baked_test, 5)

