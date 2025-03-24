

#==============================================================================
# Read real data
#==============================================================================

library(ncdf4)
library(lubridate)
library(dplyr)


# Function to extract yearly averages (for certain months)
extract_yearly_avg <- function(nc_obj, months, var_name) {
  
  # Extract time values and units
  time_values <- ncvar_get(nc_obj, "time")
  time_units <- ncatt_get(nc_obj, "time", "units")$value
  
  # Extract reference date
  time_origin <- ymd_hms("1800-01-01 00:00:00", tz = "UTC")
  time_converted <- time_origin + hours(time_values)
  
  # Extract data variable (assumes variable name is "sic")
  variable_data <- ncvar_get(nc_obj, var_name)
  
  # Create dataframe
  df <- tibble(
    time = time_converted,
    value = variable_data,
    year = year(time_converted),
    month = month(time_converted)
  )
  
  # Filter by selected months
  df_filtered <- df %>% filter(month %in% months)
  
  # Compute yearly averages
  df_avg <- df_filtered %>% group_by(year) %>% summarise(value = mean(value, na.rm = TRUE))
  
  return(df_avg)
}



# load the nc objects
bk_sic <- nc_open("data/bk_sic.nc")
nh_spv <- nc_open("data/nh_spv_uwnd.nc")
ural_slp <- nc_open("data/ural_slp.nc")
np_slp <- nc_open("data/np_slp.nc")

df_avg_bk <- extract_yearly_avg(bk_sic, c(10, 11, 12), var_name = "sic")
df_avg_spv <- extract_yearly_avg(nh_spv, c(1, 2, 3), var_name = "uwnd")
df_avg_ural <- extract_yearly_avg(ural_slp, c(10, 11, 12), var_name = "slp")
df_avg_np <- extract_yearly_avg(np_slp, c(10, 11, 12), var_name = "slp")


# plot the timeseries
par(mfrow=c(4,1))

plot(df_avg_bk$year, df_avg_bk$value, type = "l", xlab = "Year", ylab = "BK")
plot(df_avg_ural$year, df_avg_ural$value, type = "l", xlab = "Year", ylab = "Ural")
plot(df_avg_np$year, df_avg_np$value, type = "l", xlab = "Year", ylab = "NP")
plot(df_avg_spv$year, df_avg_spv$value, type = "l", xlab = "Year", ylab = "SPV")


#===============================================================================
# Data preprocessing
#===============================================================================



# standardize (zero to mean, unit variance)
bk <- (df_avg_bk$value - mean(df_avg_bk$value))/sd(df_avg_bk$value)
ural <- (df_avg_ural$value - mean(df_avg_ural$value))/sd(df_avg_ural$value)
np <- (df_avg_np$value - mean(df_avg_np$value))/sd(df_avg_np$value)
spv <- (df_avg_spv$value - mean(df_avg_spv$value))/sd(df_avg_spv$value)


# 3 histograms of the standardized data
par(mfrow=c(4,1))
hist(bk, main = "BK", xlab = "BK")
hist(ural, main = "Ural", xlab = "Ural")
hist(np, main = "NP", xlab = "NP")
hist(spv, main = "SPV", xlab = "SPV")

# detrend
bk_detrended <- residuals(lm(bk ~ df_avg_bk$year))
ural_detrended <- residuals(lm(ural ~ df_avg_ural$year))
np_detrended <- residuals(lm(np ~ df_avg_np$year))
spv_detrended <- residuals(lm(spv ~ df_avg_spv$year))



# plot the 4 detrended variables (Title "standardized-detrended")
par(mfrow=c(4,1))

plot(df_avg_bk$year, bk_detrended, type = "l", xlab = "Year", ylab = "BK", 
     main = "standardized-detrended")
plot(df_avg_ural$year, ural_detrended, type = "l", xlab = "Year", ylab = "URAL",
     main = "standardized-detrended")
plot(df_avg_np$year, np_detrended, type = "l", xlab = "Year", ylab = "NP",
     main = "standardized-detrended")
plot(df_avg_spv$year, spv_detrended, type = "l", xlab = "Year", ylab = "SPV",
     main = "standardized-detrended")



# histograms of the detrended variables

par(mfrow=c(4,1))
hist(bk_detrended, main = "BK", xlab = "BK")
hist(ural_detrended, main = "Ural", xlab = "Ural")
hist(np_detrended, main = "NP", xlab = "NP")
hist(spv_detrended, main = "SPV", xlab = "SPV")


# SPV: starting in Jan-Mar of 1951 until 2019
# X: starting in Oct-Dec of 1950 until 2018

raw_df <- data.frame(
  BK = bk_detrended[-length(bk_detrended)],
  Ural = ural_detrended[-length(ural_detrended)],
  NP = np_detrended[-length(np_detrended)],
  SPV = spv_detrended[-1]
)


# Pairsplot of variables (BK, Ural, NP, SPV)
pairs(raw_df)

#===============================================================================
# Re-Creating Experiment 4 from Paper
#===============================================================================


# note the one-calendar year lag between the autumn drivers BK, URAL, NP 
# and the reponse variable of winter SPV


# SPV: starting in Jan-Mar of 1951 until 2019
# X: starting in Oct-Dec of 1950 until 2018

# Causal effect of BK on SPV by controlling for Ural and NP
lm_BK_SPV <- lm(raw_df$SPV ~ raw_df$BK 
                + raw_df$Ural 
                + raw_df$NP)

summary(lm_BK_SPV)
sigma(lm_BK_SPV)
coef(lm_BK_SPV)


# predict SPV according to above model using the same data
pred_SPV <- predict(lm_BK_SPV, raw_df[,-4])



# make a histogram of true and predicted SPV with ggplot and alpha=4 with above example. there will be only 1 histogram with 2 overlaid, and labels
df <- data.frame(true = raw_df$SPV, pred = pred_SPV)

p <- ggplot(df, aes(x=true)) +
  geom_histogram(aes(y=..density.., fill="True"), alpha=0.4) +
  geom_histogram(aes(x=pred, y=..density.., fill="Predicted"), alpha=0.4) +
  scale_fill_manual(name="Legend", values=c("True"="blue", "Predicted"="red")) +
  labs(
    title = "Observed and Predicted SPV by Linear Regression",
    x = "SPV",
    y = "Density"
  ) + 
  theme_minimal()

p




#===============================================================================
# Check predictive value of variables
#===============================================================================



library(mgcv)  # For GAM
library(ggplot2)
library(dplyr)

# Create lagged SPV for simple lag and AR models
pred_df <- raw_df %>%
  mutate(SPV_lag1 = lag(SPV, 1))  # Remove NAs from lagging

# 1) Baseline model (predict mean SPV)
mean_spv <- mean(pred_df$SPV)
pred_df$SPV_pred_mean <- mean_spv

# 2) Simple lag (SPV_t = SPV_t-1)
pred_df$SPV_pred_simple_lag <- pred_df$SPV_lag1  # Just copying the last value

# 3) Linear Regression Model using predictors (BK, Ural, NP)
lm_model <- lm(SPV ~ BK + Ural + NP, data = pred_df)
pred_df$SPV_pred_lm <- predict(lm_model, newdata = pred_df)

# 4) AR(1) model
ar1_model <- ar(pred_df$SPV, order.max = 1, method = "mle")  
phi1 <- ar1_model$ar  # AR(1) coefficient
intercept <- ar1_model$x.mean  # Mean of SPV (used as intercept)
# Compute fitted values manually
pred_df$SPV_pred_ar1 <- intercept + phi1 * lag(pred_df$SPV, 1)


# 5) GAM model to allow for non-linear relationships
gam_model <- gam(SPV ~ s(BK) + s(Ural) + s(NP), data = pred_df)
pred_df$SPV_pred_gam <- predict(gam_model, newdata = pred_df)
par(mfrow=c(1,3))
plot(gam_model)

# Compute RMSEs
compute_rmse <- function(actual, predicted) {
  sqrt(mean((actual - predicted)^2, na.rm = TRUE))
}

rmse_results <- data.frame(
  Model = c("Mean", "Simple Lag", "Linear Regression", "AR(1)", "GAM"),
  RMSE = c(
    compute_rmse(pred_df$SPV, pred_df$SPV_pred_mean),
    compute_rmse(pred_df$SPV, pred_df$SPV_pred_simple_lag),
    compute_rmse(pred_df$SPV, pred_df$SPV_pred_lm),
    compute_rmse(pred_df$SPV[-1], pred_df$SPV_pred_ar1[-1]),  # Remove first NA
    compute_rmse(pred_df$SPV, pred_df$SPV_pred_gam)
  ),
  Description = c(
    "Mean SPV as prediction",
    "Lagged SPV as prediction",
    "LinearRegression using BK, Ural, NP",
    "Lagged SPV with coefficient as prediction",
    "Same as Linear Regression but with varying coefficients")
  
)
pred_df$Year <- df_avg_bk$year[-length(df_avg_bk$year)]
pred_df_plot <- pred_df[-1,]  # Remove first row with NA



p <- ggplot(pred_df_plot, aes(x = Year)) +
  geom_line(aes(y = SPV, color = "True SPV"), linewidth = 0.8) +
  geom_line(aes(y = SPV_pred_mean, color = "Baseline Mean"), linetype = "dashed") +
  geom_line(aes(y = SPV_pred_simple_lag, color = "Simple Lag (1)"), linetype = "dashed") +
  geom_line(aes(y = SPV_pred_lm, color = "Linear Regression"), linewidth = 0.8) +
  geom_line(aes(y = SPV_pred_ar1, color = "AR(1)")) +
  geom_line(aes(y = SPV_pred_gam, color = "GAM"), linewidth = 0.8) +
  labs(title = "Predicted vs Observed SPV", x = "Time", y = "SPV") +
  scale_color_manual(
    name = "Model",
    values = c("True SPV" = "black", "Baseline Mean" = "red", "Simple Lag (1)" = "blue",
               "Linear Regression" = "green", "AR(1)" = "purple", "GAM" = "orange")
  ) +
  # line thickness of True SPV (in the plot)
  theme_minimal()


p

rmse_results$RMSE <- round(rmse_results$RMSE,3)

# Print RMSE results
print(rmse_results)



# Scatterplots (true vs predicted) pairsplot
pairs(pred_df_plot[, c("SPV", "SPV_pred_mean", "SPV_pred_simple_lag", "SPV_pred_lm", "SPV_pred_ar1", "SPV_pred_gam")])


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
    
    
    # 1) Baseline model (predict mean SPV)
    mean_spv <- mean(train_data$SPV)
    test_data$SPV_pred_mean <- mean_spv
    
    # 2) Simple lag (SPV_t = SPV_t-1)
    test_data$SPV_pred_simple_lag <- test_data$SPV_lag1
    
    # 3) Linear Regression Model using predictors (BK, Ural, NP)
    lm_model <- lm(SPV ~ BK + Ural + NP, data = train_data)
    test_data$SPV_pred_lm <- predict(lm_model, newdata = test_data)
    
    # 4) AR(1) model
    ar1_model <- ar(train_data$SPV, order.max = 1, method = "mle", aic = FALSE) # set aic to false to avoid order=0
    phi1 <- ar1_model$ar
    intercept <- ar1_model$x.mean
    test_data$SPV_pred_ar1 <- intercept + phi1 * test_data$SPV_lag1
    
    # 5) GAM model to allow for non-linear relationships
    gam_model <- gam(SPV ~ s(BK) + s(Ural) + s(NP), data = train_data)
    test_data$SPV_pred_gam <- predict(gam_model, newdata = test_data)
    
    scatter_df <- rbind(scatter_df, test_data)
    
    # Compute RMSEs for this fold
    rmse_results_fold <- data.frame(
      Model = c("Mean", "Simple Lag", "Linear Regression", "AR(1)", "GAM"),
      RMSE = c(
        compute_rmse(test_data$SPV, test_data$SPV_pred_mean),
        compute_rmse(test_data$SPV, test_data$SPV_pred_simple_lag),
        compute_rmse(test_data$SPV, test_data$SPV_pred_lm),
        compute_rmse(test_data$SPV[-1], test_data$SPV_pred_ar1[-1]),
        compute_rmse(test_data$SPV, test_data$SPV_pred_gam)
      )
    )
    
    # Aggregate RMSEs across folds
    rmse_results_cv$RMSE <- rmse_results_cv$RMSE + rmse_results_fold$RMSE
  }
  
  # Average RMSEs across folds
  rmse_results_cv$RMSE <- rmse_results_cv$RMSE / folds
  return(list(cv = rmse_results_cv, scatter = scatter_df))
}

# Create lagged SPV for simple lag and AR models
pred_df <- raw_df %>%
  mutate(SPV_lag1 = lag(SPV, 1))

# Compute RMSEs using 10-fold cross-validation
rmse_results_cv <- perform_cv(pred_df)


rmse_results_cv$cv$RMSE <- round(rmse_results_cv$cv$RMSE,3)
print(rmse_results_cv$cv)


# scatter plots of true vs predicted values
scatter_cv <- rmse_results_cv$scatter

# Set up a 3x2 plotting layout
par(mfrow=c(3,2), mar=c(4,4,2,1))  # Adjust margins for better spacing

plot(scatter_cv$SPV, scatter_cv$SPV_pred_mean, 
     xlab = "True SPV", ylab = "Predicted SPV (Mean)", 
     main = "Mean", xlim=c(-2.1, 2.1), ylim=c(-2.1, 2.1))
abline(0, 1, col="red", lwd=2, lty=2)

plot(scatter_cv$SPV, scatter_cv$SPV_pred_simple_lag, 
     xlab = "True SPV", ylab = "Predicted SPV (Simple Lag)", 
     main = "Simple Lag", xlim=c(-2.1, 2.1), ylim=c(-2.1, 2.1))
abline(0, 1, col="red", lwd=2, lty=2)

plot(scatter_cv$SPV, scatter_cv$SPV_pred_lm, 
     xlab = "True SPV", ylab = "Predicted SPV (Linear Regression)", 
     main = "Linear Regression", xlim=c(-2.1, 2.1), ylim=c(-2.1, 2.1))
abline(0, 1, col="red", lwd=2, lty=2)

plot(scatter_cv$SPV, scatter_cv$SPV_pred_ar1, 
     xlab = "True SPV", ylab = "Predicted SPV (AR(1))", 
     main = "AR(1)", xlim=c(-2.1, 2.1), ylim=c(-2.1, 2.1))
abline(0, 1, col="red", lwd=2, lty=2)

plot(scatter_cv$SPV, scatter_cv$SPV_pred_gam, 
     xlab = "True SPV", ylab = "Predicted SPV (GAM)", 
     main = "GAM", xlim=c(-2.1, 2.1), ylim=c(-2.1, 2.1))
abline(0, 1, col="red", lwd=2, lty=2)

# Add an empty plot for layout balance
plot.new()

