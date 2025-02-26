# install.packages("raster")
library(raster) # needed?

# install.packages("ncdf4")
library(ncdf4) # package for netcdf manipulation


# library(rgdal) # package for geospatial analysis
library(ggplot2) # package for plotting



# read the file "enso_son.nc"  without raster stored in the same directory

enso <- nc_open("enso_son.nc")
enso_y <- ncvar_get(enso, "Year")
enso_v <- ncvar_get(enso, "enso")

# print the timeseries of enso
# plot(enso_y, enso_v, type = "l", xlab = "Year", ylab = "ENSO")


# read the file "iod_son.nc" stored in the same directory

iod <- nc_open("iod_son.nc")
print(iod)
iod_y <- ncvar_get(iod, "Year")
iod_v <- ncvar_get(iod, "iod")

# print the timeseries of iod
# plot(iod_y, iod_v, type = "l", xlab = "Year", ylab = "IOD")


# read the file "precip_au_son.nc" stored in the same directory

precip <- nc_open("precip_au_son.nc")
print(precip)
precip_y <- ncvar_get(precip, "year")
precip_v <- ncvar_get(precip, "precip")

# print the timeseries of precipitation
# plot(precip_y, precip_v, type = "l", xlab = "Year", ylab = "Precipitation")


# plot all togehter
par(mfrow=c(3,1))
plot(enso_y, enso_v, type = "l", xlab = "Year", ylab = "ENSO")
plot(iod_y, iod_v, type = "l", xlab = "Year", ylab = "IOD")
plot(precip_y, precip_v, type = "l", xlab = "Year", ylab = "AU Precipitation")

# check the years: all from 1950-2019
# enso_y
# iod_y
# precip_y

# check the values: different scales
# enso_v
# iod_v
# precip_v

# 3 histograms of the data
# par(mfrow=c(3,1))
# hist(enso_v, main = "ENSO", xlab = "ENSO")
# hist(iod_v, main = "IOD", xlab = "IOD")
# hist(precip_v, main = "Precipitation", xlab = "Precipitation")

#===============================================================================
# Data preprocessing
#===============================================================================

# standardize (zero to mean, unit variance)
ENSO <- (enso_v - mean(enso_v))/sd(enso_v)
IOD <- (iod_v - mean(iod_v))/sd(iod_v)
AU <- (precip_v - mean(precip_v))/sd(precip_v)

# 3 histograms of the standardized data
par(mfrow=c(3,1))
hist(ENSO, main = "ENSO", xlab = "ENSO")
hist(IOD, main = "IOD", xlab = "IOD")
hist(AU, main = "Precipitation", xlab = "Precipitation")

# detrend

ENSO_detrended <- residuals(lm(ENSO ~ enso_y))
IOD_detrended <- residuals(lm(IOD ~ iod_y))
AU_detrended <- residuals(lm(AU ~ precip_y))

# plot the 3 detrended variables (Title "standardized-detrended")
par(mfrow=c(3,1))

plot(enso_y, ENSO_detrended, type = "l", xlab = "Year", ylab = "ENSO", 
     main = "standardized-detrended")
plot(iod_y, IOD_detrended, type = "l", xlab = "Year", ylab = "IOD",
     main = "standardized-detrended")
plot(precip_y, AU_detrended, type = "l", xlab = "Year", ylab = "AU Precipitation",
     main = "standardized-detrended")

#===============================================================================
# Re-Creating Experiment 5 (Cross-Tables)
#===============================================================================

# Load required libraries
# install.packages("dplyr")
# install.packages("tidyverse")  # Includes ggplot2, dplyr, tidyr
# install.packages("gmodels")    # For CrossTable
# library(dplyr)
# library(tidyverse)
# library(gmodels)
# 
# # Step 1: Create Data Frame with Detrended Variables
# df <- data.frame(
#   ENSO = ENSO_detrended,
#   IOD = IOD_detrended,
#   AU = AU_detrended
# )
# 
# # Step 2: Convert Numeric Variables to Categorical
# df$AU <- cut(df$AU, breaks = quantile(df$AU, probs = c(0, 0.5, 1)), labels = c("low", "high"), include.lowest = TRUE)
# df$ENSO <- cut(df$ENSO, breaks = quantile(df$ENSO, probs = c(0, 1/3, 2/3, 1)), labels = c("Nina", "neut", "Nino"), include.lowest = TRUE)
# df$IOD <- cut(df$IOD, breaks = quantile(df$IOD, probs = c(0, 1/3, 2/3, 1)), labels = c("neg", "zero", "pos"), include.lowest = TRUE)
# 
# # Step 3a: Contingency Table (Absolute Frequencies)
# table1 <- xtabs(~ AU + IOD + ENSO, data = df)
# ftable(table1)  # Print contingency table
# 
# # Step 3b: Conditional Probabilities P(IOD, ENSO)
# table2 <- prop.table(xtabs(~ IOD + ENSO, data = df))  
# print(table2)  # Shows probability distribution of IOD given ENSO
# 
# # Step 3c: Conditional Probability P(AU | IOD, ENSO)
# table3 <- prop.table(xtabs(~ AU + ENSO + IOD, data = df), margin = c(2, 3))
# ftable(table3)  # Conditional probability table
# 
# # Step 3d: Marginal Probability of ENSO (P(AU | ENSO))
# table4 <- prop.table(xtabs(~ AU + ENSO, data = df), margin = 2)
# print(table4)
# 
# # Step 3e: Marginal Probability of IOD (P(AU | IOD))
# table5 <- prop.table(xtabs(~ AU + IOD, data = df), margin = 2)
# print(table5)


# ---> All good



