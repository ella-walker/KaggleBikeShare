library(tidyverse)
library(tidymodels)
library(vroom)
library(skimr)
library(ggplot2)
library(patchwork)

bike <- vroom("~/Documents/STAT 348/KaggleBikeShare/bike-sharing-demand/train.csv")

dplyr::glimpse(bike)
skimr::skim(bike)
DataExplorer::plot_intro(bike) #There are no missing values
DataExplorer::plot_correlation(bike)
DataExplorer::plot_bar(bike) #temp and atemp are highly correlated, as are count and registered 
DataExplorer::plot_histogram(bike) 
GGally::ggpairs(bike)

tempplot <- ggplot(data = bike, aes(x = temp)) +
  geom_histogram(bins = 50) 
tempplot

weatherplot <- ggplot(data = bike, aes(x = weather, fill = count)) +
  geom_bar()
weatherplot

windplot <- ggplot(data = bike, aes(x = windspeed, y = count)) +
  geom_point()
windplot

tempplot2 <- ggplot(data = bike, aes(x = temp, y = atemp)) +
  geom_point()
tempplot2

(tempplot + tempplot2) / (weatherplot + windplot)

