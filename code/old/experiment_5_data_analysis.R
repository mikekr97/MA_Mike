

#==============================================================================
# Read real data
#==============================================================================

library(ncdf4)
 


# read the file "enso_son.nc"  without raster stored in the same directory

enso <- nc_open("data/enso_son.nc")
enso_y <- ncvar_get(enso, "Year")
enso_v <- ncvar_get(enso, "enso")

# print the timeseries of enso
# plot(enso_y, enso_v, type = "l", xlab = "Year", ylab = "ENSO")


# read the file "iod_son.nc" stored in the same directory

iod <- nc_open("data/iod_son.nc")
print(iod)
iod_y <- ncvar_get(iod, "Year")
iod_v <- ncvar_get(iod, "iod")

# print the timeseries of iod
# plot(iod_y, iod_v, type = "l", xlab = "Year", ylab = "IOD")


# read the file "precip_au_son.nc" stored in the same directory

precip <- nc_open("data/precip_au_son.nc")
print(precip)
precip_y <- ncvar_get(precip, "year")
precip_v <- ncvar_get(precip, "precip")
# hist(precip_v)

# plot all together
par(mfrow=c(3,1))
plot(enso_y, enso_v, type = "l", xlab = "Year", ylab = "ENSO")
plot(iod_y, iod_v, type = "l", xlab = "Year", ylab = "IOD")
plot(precip_y, precip_v, type = "l", xlab = "Year", ylab = "AU Precipitation")



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

# 3 histograms of the detrended data
par(mfrow=c(3,1))
hist(ENSO_detrended, main = "ENSO", xlab = "ENSO")
hist(IOD_detrended, main = "IOD", xlab = "IOD")
hist(AU_detrended, main = "Precipitation", xlab = "Precipitation")



raw_df <- data.frame(
  ENSO = ENSO_detrended,
  IOD = IOD_detrended,
  AU = AU_detrended
)




# Pairsplot of variables (ENSO, IOD, AU)
pairs(raw_df)



#===============================================================================
# Re-Creating Experiment 5 from paper (Cross-Tables)
#===============================================================================

# Load required libraries
# install.packages("dplyr")
# install.packages("tidyverse")  # Includes ggplot2, dplyr, tidyr
# install.packages("gmodels")    # For CrossTable
library(dplyr)
library(tidyverse)
library(gmodels)

# # Step 1: Create Data Frame with Detrended Variables

df <- raw_df

# # Step 2: Convert Numeric Variables to Categorical
df$AU <- cut(df$AU, breaks = quantile(df$AU, probs = c(0, 0.5, 1)), labels = c("low", "high"), include.lowest = TRUE)
df$ENSO <- cut(df$ENSO, breaks = quantile(df$ENSO, probs = c(0, 1/3, 2/3, 1)), labels = c("Nina", "neut", "Nino"), include.lowest = TRUE)
df$IOD <- cut(df$IOD, breaks = quantile(df$IOD, probs = c(0, 1/3, 2/3, 1)), labels = c("neg", "zero", "pos"), include.lowest = TRUE)

par(mfrow=c(1,3))
plot(df$ENSO, main = "ENSO")
plot(df$IOD, main = "IOD")
plot(df$AU, main = "AU")
# 
# # Step 3a: Contingency Table (Absolute Frequencies)
table1 <- xtabs(~ AU + IOD + ENSO, data = df)
ftable(table1)  # Print contingency table
# 
# # Step 3b: Conditional Probabilities P(IOD, ENSO)
table2 <- prop.table(xtabs(~ IOD + ENSO, data = df))
print(table2)  # Shows probability distribution of IOD given ENSO

# # Step 3c: Conditional Probability P(AU | IOD, ENSO)
table3 <- prop.table(xtabs(~ AU + ENSO + IOD, data = df), margin = c(2, 3))
ftable(round(table3, 3))  # Conditional probability table

# # Step 3d: Marginal Probability of ENSO (P(AU | ENSO))
table4 <- prop.table(xtabs(~ AU + ENSO, data = df), margin = 2)
print(round(table4, 3))

# # Step 3e: Marginal Probability of IOD (P(AU | IOD))
table5 <- prop.table(xtabs(~ AU + IOD, data = df), margin = 2)
print(round(table5, 3))

# gam of iod ~enso
gam_model <- gam(IOD ~ s(ENSO), data = raw_df)
plot(gam_model)

# gam of au ~enso
gam_model <- gam(AU ~ s(ENSO), data = raw_df)
plot(gam_model)

# gam of au ~iod
gam_model <- gam(AU ~ s(IOD), data = raw_df)
plot(gam_model)

# gam of au ~enso + iod
par(mfrow=c(1,2))
gam_model_1 <- gam(AU ~ s(ENSO) + s(IOD), data = raw_df)
plot(gam_model_1)



#===============================================================================
# Check Polr
#===============================================================================

# make the df ordered
df$ENSO <- factor(df$ENSO, levels = c("Nina", "neut", "Nino"), ordered = TRUE)
df$IOD <- factor(df$IOD, levels = c("neg", "zero", "pos"), ordered = TRUE)
df$AU <- factor(df$AU, levels = c("low", "high"), ordered = TRUE)

df$ENSO

library(tram)

fit21 <- polr(IOD ~ ENSO , data = df, Hess=TRUE)
summary(fit21)

# same coefficients as polr
fit21 <- Polr(IOD ~ ENSO , data = df)
summary(fit21)

fit22 <- polr(AU ~ ENSO + IOD , data = df, Hess=TRUE)

#===============================================================================
# Check predictive value of variables
#===============================================================================



library(mgcv)  # For GAM
library(ggplot2)
library(dplyr)

raw_df

# Create lagged AU for simple lag and AR models
pred_df <- raw_df %>%
  mutate(AU_lag1 = lag(AU, 1))  # Remove NAs from lagging

# 1) Baseline model (predict mean AU)
mean_AU <- mean(pred_df$AU)
pred_df$AU_pred_mean <- mean_AU

# 2) Simple lag (AU_t = AU_t-1)
pred_df$AU_pred_simple_lag <- pred_df$AU_lag1  # Just copying the last value

# 3) Linear Regression Model using predictors (BK, Ural, NP)
lm_model <- lm(AU ~ ENSO + IOD, data = pred_df)
pred_df$AU_pred_lm <- predict(lm_model, newdata = pred_df)
summary(lm_model)

# 4) AR(1) model
ar1_model <- ar(pred_df$AU, order.max = 1, method = "mle")  
phi1 <- ar1_model$ar  # AR(1) coefficient
intercept <- ar1_model$x.mean  # Mean of AU (used as intercept)
# Compute fitted values manually
pred_df$AU_pred_ar1 <- intercept + phi1 * lag(pred_df$AU, 1)


# 5) GAM model to allow for non-linear relationships
gam_model <- gam(AU ~ s(ENSO) + s(IOD) , data = pred_df)
pred_df$AU_pred_gam <- predict(gam_model, newdata = pred_df)
par(mfrow=c(1,2))
plot(gam_model)

# Compute RMSEs
compute_rmse <- function(actual, predicted) {
  sqrt(mean((actual - predicted)^2, na.rm = TRUE))
}

rmse_results <- data.frame(
  Model = c("Mean", "Simple Lag", "Linear Regression", "AR(1)", "GAM"),
  RMSE = c(
    compute_rmse(pred_df$AU, pred_df$AU_pred_mean),
    compute_rmse(pred_df$AU, pred_df$AU_pred_simple_lag),
    compute_rmse(pred_df$AU, pred_df$AU_pred_lm),
    compute_rmse(pred_df$AU[-1], pred_df$AU_pred_ar1[-1]),  # Remove first NA
    compute_rmse(pred_df$AU, pred_df$AU_pred_gam)
  ),
  Description = c(
    "Mean AU as prediction",
    "Lagged AU as prediction",
    "LinearRegression using BK, Ural, NP",
    "Lagged AU with coefficient as prediction",
    "Same as Linear Regression but with varying coefficients")
  
)


pred_df$Year <- enso_y
pred_df_plot <- pred_df[-1,]  # Remove first row with NA



p <- ggplot(pred_df_plot, aes(x = Year)) +
  geom_line(aes(y = AU, color = "True AU"), linewidth = 0.8) +
  geom_line(aes(y = AU_pred_mean, color = "Baseline Mean"), linetype = "dashed") +
  geom_line(aes(y = AU_pred_simple_lag, color = "Simple Lag (1)"), linetype = "dashed") +
  geom_line(aes(y = AU_pred_lm, color = "Linear Regression"), linewidth = 0.8) +
  geom_line(aes(y = AU_pred_ar1, color = "AR(1)")) +
  geom_line(aes(y = AU_pred_gam, color = "GAM"), linewidth = 0.8) +
  labs(title = "Predicted vs Observed AU", x = "Time", y = "AU") +
  scale_color_manual(
    name = "Model",
    values = c("True AU" = "black", "Baseline Mean" = "red", "Simple Lag (1)" = "blue",
               "Linear Regression" = "green", "AR(1)" = "purple", "GAM" = "orange")
  ) +
  # line thickness of True AU (in the plot)
  theme_minimal()


p

rmse_results$RMSE <- round(rmse_results$RMSE,3)

# Print RMSE results
print(rmse_results)



# Scatterplots (true vs predicted) pairsplot
pairs(pred_df_plot[, c("AU", "AU_pred_mean", "AU_pred_simple_lag", "AU_pred_lm", "AU_pred_ar1", "AU_pred_gam")])


#===============================================================================
# 4-fold Cross Validation
#===============================================================================


scatter_df <- data.frame(NULL)

# Function to perform 10-fold cross-validation and calculate RMSE
perform_cv <- function(data, folds = 10) {
  # data <- pred_df
  n <- nrow(data)
  set.seed(123)
  fold_indices <- sample(rep(1:folds, length.out = n))
  rmse_results_cv <- data.frame(
    Model = c("Mean", "Simple Lag", "Linear Regression", "AR(1)", "GAM"),
    RMSE = rep(0, 5)
  )
  
  for (i in 1:folds) {
    # i = 1
    train_data <- data[fold_indices != i, ]
    test_data <- data[fold_indices == i, ]
    
    
    # 1) Baseline model (predict mean AU)
    mean_AU <- mean(train_data$AU)
    test_data$AU_pred_mean <- mean_AU
    
    # 2) Simple lag (AU_t = AU_t-1)
    test_data$AU_pred_simple_lag <- test_data$AU_lag1
    
    # 3) Linear Regression Model using predictors (BK, Ural, NP)
    lm_model <- lm(AU ~ ENSO + IOD, data = train_data)
    test_data$AU_pred_lm <- predict(lm_model, newdata = test_data)
    
    # 4) AR(1) model
    ar1_model <- ar(train_data$AU, order.max = 1, method = "mle", aic = FALSE) # set aic to false to avoid order=0
    phi1 <- ar1_model$ar
    intercept <- ar1_model$x.mean
    test_data$AU_pred_ar1 <- intercept + phi1 * test_data$AU_lag1
    
    # 5) GAM model to allow for non-linear relationships
    gam_model <- gam(AU ~ s(ENSO) + s(IOD) , data = train_data)
    test_data$AU_pred_gam <- predict(gam_model, newdata = test_data)
    
    scatter_df <- rbind(scatter_df, test_data)
    
    # Compute RMSEs for this fold
    rmse_results_fold <- data.frame(
      Model = c("Mean", "Simple Lag", "Linear Regression", "AR(1)", "GAM"),
      RMSE = c(
        compute_rmse(test_data$AU, test_data$AU_pred_mean),
        compute_rmse(test_data$AU[-1], test_data$AU_pred_simple_lag[-1]),
        compute_rmse(test_data$AU, test_data$AU_pred_lm),
        compute_rmse(test_data$AU[-1], test_data$AU_pred_ar1[-1]),
        compute_rmse(test_data$AU, test_data$AU_pred_gam)
      )
    )
    
    # Aggregate RMSEs across folds
    rmse_results_cv$RMSE <- rmse_results_cv$RMSE + rmse_results_fold$RMSE
  }
  
  # Average RMSEs across folds
  rmse_results_cv$RMSE <- rmse_results_cv$RMSE / folds
  return(list(cv = rmse_results_cv, scatter = scatter_df))
}

# Create lagged AU for simple lag and AR models
pred_df <- raw_df %>%
  mutate(AU_lag1 = lag(AU, 1))

# Compute RMSEs using 10-fold cross-validation
rmse_results_cv <- perform_cv(pred_df)


rmse_results_cv$cv$RMSE <- round(rmse_results_cv$cv$RMSE,3)
print(rmse_results_cv$cv)


# scatter plots of true vs predicted values
scatter_cv <- rmse_results_cv$scatter

# Set up a 3x2 plotting layout
par(mfrow=c(3,2), mar=c(4,4,2,1))  # Adjust margins for better spacing

plot(scatter_cv$AU, scatter_cv$AU_pred_mean, 
     xlab = "True AU", ylab = "Predicted AU (Mean)", 
     main = "Mean", xlim=c(-2.1, 2.1), ylim=c(-2.1, 2.1))
abline(0, 1, col="red", lwd=2, lty=2)

plot(scatter_cv$AU, scatter_cv$AU_pred_simple_lag, 
     xlab = "True AU", ylab = "Predicted AU (Simple Lag)", 
     main = "Simple Lag", xlim=c(-2.1, 2.1), ylim=c(-2.1, 2.1))
abline(0, 1, col="red", lwd=2, lty=2)

plot(scatter_cv$AU, scatter_cv$AU_pred_lm, 
     xlab = "True AU", ylab = "Predicted AU (Linear Regression)", 
     main = "Linear Regression", xlim=c(-2.1, 2.1), ylim=c(-2.1, 2.1))
abline(0, 1, col="red", lwd=2, lty=2)

plot(scatter_cv$AU, scatter_cv$AU_pred_ar1, 
     xlab = "True AU", ylab = "Predicted AU (AR(1))", 
     main = "AR(1)", xlim=c(-2.1, 2.1), ylim=c(-2.1, 2.1))
abline(0, 1, col="red", lwd=2, lty=2)

plot(scatter_cv$AU, scatter_cv$AU_pred_gam, 
     xlab = "True AU", ylab = "Predicted AU (GAM)", 
     main = "GAM", xlim=c(-2.1, 2.1), ylim=c(-2.1, 2.1))
abline(0, 1, col="red", lwd=2, lty=2)

# Add an empty plot for layout balance
plot.new()


