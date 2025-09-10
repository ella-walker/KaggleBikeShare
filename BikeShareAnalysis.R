library(tidyverse)
library(tidymodels)
library(vroom)

bike <- vroom("~/Documents/STAT 348/KaggleBikeShare/bike-sharing-demand/train.csv")
trainData <- bike |>
  select(-casual, -registered)
testData <- vroom("~/Documents/STAT 348/KaggleBikeShare/bike-sharing-demand/test.csv")

# Fitting a linear regression model

## Setup and Fit the Linear Regression Model
my_linear_model <- linear_reg() |>
  set_engine("lm") |>
  set_mode("regression") |>
  fit(formula=count~.-datetime, data=trainData)

## Generate Predictions Using Linear Model
bike_predictions <- predict(my_linear_model,
                            new_data=testData)
bike_predictions

## Format the Predictions for Submission to Kaggle
kaggle_submission <- bike_predictions |>
  bind_cols(testData) |>
  rename(count = .pred) |>
  mutate(count = pmax(0, count)) |>
  mutate(datetime = as.character(format(datetime))) |>
  select(datetime, count)

## Write out the file
vroom_write(x = kaggle_submission, file="~/Documents/STAT 348/KaggleBikeShare/LinearPreds.csv", delim=",")
  