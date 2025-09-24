#install.packages("rpart")
library(tidyverse)
library(tidymodels)
library(vroom)
library(glmnet)
library(bonsai)

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
bake(prep(bike_recipe), trainData)

## Linear Regression Model with Recipe
lin_model <- linear_reg() |>
  set_engine("lm") |>
  set_mode("regression")

bike_workflow <- workflow() |>
  add_recipe(bike_recipe) |>
  add_model(lin_model) |>
  fit(data = trainData)

# bike_predictions <- predict(bike_workflow, new_data = testData) |>
#   mutate(count = exp(count)) # back-transform log(count) prediction

## Penalized Regression Model
preg_model <- linear_reg(penalty= tune(), mixture= tune()) |>
  set_engine("glmnet")

## Set Workflow
my_mod <- workflow() |>
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

## Run the CV with Regression Trees
CV_results <- my_mod |>
  tune_grid(resamples = folds,
            grid = grid_of_tuning_params,
            metrics = metric_set(rmse, mae))

## Find Best Tuning Parameters
bestTune <- CV_results |>
  select_best(metric="rmse")

## Finalize the Workflow & fit it
final_wf <- my_mod |>
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
            file = "~/Documents/STAT 348/KaggleBikeShare/regression_trees.csv",
            delim = ",")

# Print first 5 rows of baked data set
bike_prep <- prep(bike_recipe)
baked_test <- bake(bike_prep, new_data = testData)
head(baked_test, 5)

###
## Regression Tree
###

## Create Regression Trees in R
tree_mod <- decision_tree(tree_depth = tune(),
                        cost_complexity = tune(),
                        min_n = tune()) |>
  set_engine("rpart") |>
  set_mode("regression")

tree_wf <- workflow() |>
  add_recipe(bike_recipe) |>
  add_model(tree_mod)

tree_grid_of_tuning_params <- grid_regular(tree_depth(),
                                      cost_complexity(),
                                      min_n(),
                                      levels = 5)

folds <- vfold_cv(trainData, v = 5, repeats = 1)

tree_CV_results <- tree_wf |>
  tune_grid(resamples = folds,
            grid = tree_grid_of_tuning_params,
            metrics = metric_set(rmse, mae))

tree_bestTune <- tree_CV_results |>
  select_best(metric="rmse")

tree_final_wf <- tree_wf |>
  finalize_workflow(tree_bestTune) |>
  fit(data = trainData)

tree_preds <- predict(tree_final_wf, new_data = testData)

tree_kaggle_submission <- tree_preds |>
  bind_cols(testData |> select(datetime)) |>
  transmute(
    datetime = format(datetime, "%Y-%m-%d %H:%M:%S"),
    count = pmax(0, exp(.pred))
  )

## Write out the file
vroom_write(tree_kaggle_submission,
            file = "~/Documents/STAT 348/KaggleBikeShare/regression_trees.csv",
            delim = ",")

###
## Random Forest
###

random_forest_mod <- rand_forest(mtry = tune(),
                                 min_n = tune(),
                                 trees = 500) |>
  set_engine("ranger") |>
  set_mode("regression")

random_forest_wf <- workflow() |>
  add_recipe(bike_recipe) |>
  add_model(random_forest_mod)

random_forest_grid_of_tuning_params <- grid_regular(mtry(range = c(1, ncol(trainData) - 1)),
                                           min_n(),
                                           levels = 5)

folds <- vfold_cv(trainData, v = 5, repeats = 1)

random_forest_CV_results <- random_forest_wf |>
  tune_grid(resamples = folds,
            grid = random_forest_grid_of_tuning_params,
            metrics = metric_set(rmse, mae))

random_forest_bestTune <- random_forest_CV_results |>
  select_best(metric="rmse")

random_forest_final_wf <- random_forest_wf |>
  finalize_workflow(random_forest_bestTune) |>
  fit(data = trainData)

random_forest_preds <- predict(random_forest_final_wf, new_data = testData)

random_forest_kaggle_submission <- random_forest_preds |>
  bind_cols(testData |> select(datetime)) |>
  transmute(
    datetime = format(datetime, "%Y-%m-%d %H:%M:%S"),
    count = pmax(0, exp(.pred))
  )

vroom_write(random_forest_kaggle_submission,
            file = "~/Documents/STAT 348/KaggleBikeShare/random_forest.csv",
            delim = ",")

###
## BART
###


bart_model <- bart(trees = tune()) |>
  set_engine("dbarts") |>
  set_mode("regression")

bart_wf <- workflow() |>
  add_recipe(bike_recipe) |>
  add_model(bart_model)

bart_grid_of_tuning_params <- grid_regular(
  trees(range = c(20,200)),
  levels = 5
)

folds <- vfold_cv(trainData, v = 5, repeats = 1)

bart_CV_results <- bart_wf |>
  tune_grid(
    resamples = folds,
    grid = bart_grid_of_tuning_params,
    metrics = metric_set(rmse, mae))

bart_bestTune <- bart_CV_results |>
  select_best(metric="rmse")

bart_final_wf <- bart_wf |>
  finalize_workflow(bart_bestTune) |>
  fit(data = trainData)

bart_preds <- predict(bart_final_wf, new_data = testData)

bart_kaggle_submission <- bart_preds |>
  bind_cols(testData |> select(datetime)) |>
  transmute(
    datetime = format(datetime, "%Y-%m-%d %H:%M:%S"),
    count = pmax(0, exp(.pred))
  )

vroom_write(bart_kaggle_submission,
            file = "~/Documents/STAT 348/KaggleBikeShare/bart.csv",
            delim = ",")
