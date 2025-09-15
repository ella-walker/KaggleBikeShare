library(tidyverse)
library(tidymodels)
library(vroom)

# Cleaning data (step 1)
bike <- vroom("~/Documents/STAT 348/KaggleBikeShare/bike-sharing-demand/train.csv")
trainData <- bike |>
  select(-casual, -registered) |>
  mutate(count = log(count))

# Read in test data set
testData <- vroom("~/Documents/STAT 348/KaggleBikeShare/bike-sharing-demand/test.csv")

# Defining a recipe (step 2)
bike_recipe <- recipe(count ~ ., data = trainData) |>
  step_mutate(weather = ifelse(weather == 4, 3, weather)) |>
  step_mutate(weather = factor(weather, 
                               levels = c(1, 2, 3), 
                               labels = c("clear", "cloudy", "precipitation"))) |>
  step_time(datetime, features = c("hour")) |>
  step_mutate(season = factor(season, 
                              levels = c(1, 2, 3, 4),
                              labels = c("spring", "summer", "fall", "winter"))) |>
  step_corr(all_numeric_predictors(), threshold = 0.7)

# Linear Regression Model with Recipe (step 3)
lin_model <- linear_reg() |>
  set_engine("lm") |>
  set_mode("regression")

bike_workflow <- workflow() |>
  add_recipe(bike_recipe) |>
  add_model(lin_model) |>
  fit(data = trainData)

bike_predictions <- predict(bike_workflow, new_data = testData) |>
  mutate(count = exp(count)) # back-transform log(count) prediction (step 4)

## Format the Predictions for Submission to Kaggle
kaggle_submission <- testData |>
  mutate(count = pmax(0, bike_predictions |> pull(.pred) |> exp()),
         datetime = format(datetime, "%Y-%m-%d %H:%M:%S")) |>
  select(datetime, count)

## Write out the file
vroom_write(kaggle_submission,
            file = "~/Documents/STAT 348/KaggleBikeShare/submission2.csv",
            delim = ",")

# Print first 5 rows of baked data set (step 5)
bike_prep <- prep(bike_recipe)
baked_test <- bake(bike_prep, new_data = testData)
head(baked_test, 5)

