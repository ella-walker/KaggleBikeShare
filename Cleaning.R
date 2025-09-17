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
preg_model_1 <- linear_reg(penalty= 5, mixture= 0) |>
  set_engine("glmnet")
preg_wf_1 <- workflow() |>
  add_recipe(bike_recipe) |>
  add_model(preg_model_1) |>
  fit(data=trainData)
prediction_1 <- predict(preg_wf_1, new_data=testData)

preg_model_2 <- linear_reg(penalty= 10, mixture= 1) |>
  set_engine("glmnet")
preg_wf_2 <- workflow() |>
  add_recipe(bike_recipe) |>
  add_model(preg_model_2) |>
  fit(data=trainData)
prediction_2 <- predict(preg_wf_2, new_data=testData)

preg_model_3 <- linear_reg(penalty= 20, mixture= 0.2) |>
  set_engine("glmnet")
preg_wf_3 <- workflow() |>
  add_recipe(bike_recipe) |>
  add_model(preg_model_3) |>
  fit(data=trainData)
prediction_3 <- predict(preg_wf_3, new_data=testData)

preg_model_4 <- linear_reg(penalty= 1, mixture= 0.5) |>
  set_engine("glmnet")
preg_wf_4 <- workflow() |>
  add_recipe(bike_recipe) |>
  add_model(preg_model_4) |>
  fit(data=trainData)
prediction_4 <- predict(preg_wf_4, new_data=testData)

preg_model_5 <- linear_reg(penalty= 15, mixture= 0.8) |>
  set_engine("glmnet")
preg_wf_5 <- workflow() |>
  add_recipe(bike_recipe) |>
  add_model(preg_model_5) |>
  fit(data=trainData)
prediction_5 <- predict(preg_wf_5, new_data=testData)

## Format the Predictions for Submission to Kaggle
kaggle_submission <- prediction_1 |>
  bind_cols(testData |> select(datetime)) |>
  transmute(
    datetime = format(datetime, "%Y-%m-%d %H:%M:%S"),  # force exact format
    count = pmax(0, .pred)
  )

## Write out the file
vroom_write(kaggle_submission,
            file = "~/Documents/STAT 348/KaggleBikeShare/penalized_1.csv",
            delim = ",")

# Print first 5 rows of baked data set
bike_prep <- prep(bike_recipe)
baked_test <- bake(bike_prep, new_data = testData)
head(baked_test, 5)

