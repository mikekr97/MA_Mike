

#==============================================================================
# Read real data (Experiment 3)
#==============================================================================

library(ncdf4)





enso <- nc_open("data/enso_full_ond_no2002.nc")
enso_y <- ncvar_get(enso, "year")
enso_v <- ncvar_get(enso, "enso")



jet <- nc_open("data/sh_jet_ond_no2002.nc")
jet_y <- ncvar_get(jet, "year")
jet_v <- ncvar_get(jet, "sh_jet_ond")


spv <- nc_open("data/vortex_breakdown_no2002.nc")
spv_y <- ncvar_get(spv, "index")
spv_v <- ncvar_get(spv, "days")


# plot all together
par(mfrow=c(3,1))
plot(enso_y, enso_v, type = "l", xlab = "Year", ylab = "ENSO")
plot(jet_y, jet_v, type = "l", xlab = "Year", ylab = "JET")
plot(jet_y, spv_v, type = "l", xlab = "Year", ylab = "SPV")



#===============================================================================
# Data preprocessing
#===============================================================================

# standardize the 3 variables (zero to mean, unit variance)
ENSO <- (enso_v - mean(enso_v)) / sd(enso_v)
JET <- (jet_v - mean(jet_v)) / sd(jet_v)
SPV <- (spv_v - mean(spv_v)) / sd(spv_v)



# 3 histograms of the standardized data
par(mfrow=c(3,1))
hist(ENSO, main = "ENSO", xlab = "ENSO")
hist(JET, main = "JET", xlab = "JET")
hist(SPV, main = "SPV", xlab = "SPV")

# detrend with lm
ENSO_detrended <- residuals(lm(ENSO ~ enso_y))
JET_detrended <- residuals(lm(JET ~ jet_y))
SPV_detrended <- residuals(lm(SPV ~ jet_y))


# plot the 3 detrended variables (Title "standardized-detrended")
par(mfrow=c(3,1))
plot(enso_y, ENSO_detrended, type = "l", xlab = "Year", ylab = "ENSO")
plot(jet_y, JET_detrended, type = "l", xlab = "Year", ylab = "JET")
plot(jet_y, SPV_detrended, type = "l", xlab = "Year", ylab = "SPV")

# 3 histograms of the detrended data
par(mfrow=c(3,1))
hist(ENSO_detrended, main = "ENSO", xlab = "ENSO")
hist(JET_detrended, main = "JET", xlab = "JET")
hist(SPV_detrended, main = "SPV", xlab = "SPV")



raw_df <- data.frame(
  ENSO = ENSO_detrended,
  JET = JET_detrended,
  SPV = SPV_detrended
)



# Pairsplot of variables
pairs(raw_df)



#===============================================================================
# Re-Creating Experiment 3 from paper 
#===============================================================================


#================================================================
# Determine the total effect of ENSO on Jet
#================================================================

# lm of enso on jet
lm_jet_enso <- lm(JET ~ ENSO, data = raw_df)
summary(lm_jet_enso)

ENSOtoJET_tot <- coef(lm_jet_enso)[2]

cat("The total causal effect of ENSO on JET  =",round(coef(lm_jet_enso)[2], 3))

# The total causal effect of ENSO on JET  = -0.14



#==========================================================================
# Determine the tropospheric-only effect of ENSO on JET by conditioning on SPV
#==========================================================================


lm_jet_enso_spv <- lm(JET ~ ENSO + SPV, data = raw_df)

ENSOtoJET_tropo <- coef(lm_jet_enso_spv)[2]

cat("The direct effect of ENSO on Jet (conditioned on SPV) is ", round(coef(lm_jet_enso_spv)[2],3))
# The direct effect of ENSO on Jet (conditioned on SPV) is  -0.04


cat("The regression coeff. of SPV on Jet is ", round(coef(lm_jet_enso_spv)[3],3))
# The regression coeff. of SPV on Jet is  0.39

#==========================================================================
# Determine the stratoshperic pathway from ENSO --> JET
#==========================================================================

#============
# ENSO to SPV:
#============
lm_spv_enso <- lm(SPV ~ ENSO, data = raw_df)
ENSOtoVORTEX <- coef(lm_spv_enso)[2]

#============
# SPV to Jet (same as above)
#============

lm_jet_spv_enso <- lm(JET ~ SPV + ENSO, data = raw_df)
VORTEXtoJET <- coef(lm_jet_spv_enso)[2]

#============
# Strength of stratopsheric pathway
#============
Strato_pathway = ENSOtoVORTEX * VORTEXtoJET 


cat("The causal effect of ENSO on SPV is ", round(ENSOtoVORTEX,2))
cat("The causal effect of SPV on Jet (cond on ENSO) is ", round(VORTEXtoJET ,2))
cat("The strength of the stratospheric pathway is ", round(Strato_pathway,2))
# The causal effect of ENSO on SPV is  -0.26
# The causal effect of SPV on Jet (cond on ENSO) is  0.39
# The strength of the stratospheric pathway is  -0.1



#==========================================================================
# Compare the total effect with the sum of the stratospheric and tropospheric patways
#==========================================================================

cat("The total effect of ENSO on JET  =", ENSOtoJET_tot)

cat("The sum of tropospheric + stratospheric pathways is ", round(Strato_pathway + ENSOtoJET_tropo,2))

# The total effect of ENSO on JET  = -0.14
# The sum of tropospheric + stratospheric pathways is  -0.14

# Conclusions
# When regressing Jet on ENSO, a negative, total effect of - 0.14 is found. 
# Separating the indirect stratospheric pathway (ENSO VORTEX JET) from the direct 
# tropospheric pathway (ENSO JET), the direct effect of ENSO on JET is found 
# to be weaker (-0.04) than the indirect, stratopsheric pathway (-0.10).



#===============================================================================
# Check predictive value of variables
#===============================================================================



library(mgcv)  # For GAM
library(ggplot2)
library(dplyr)

raw_df

# Create lagged JET for simple lag and AR models
pred_df <- raw_df %>%
  mutate(JET_lag1 = lag(JET, 1))  # Remove NAs from lagging

# 1) Baseline model (predict mean JET)
mean_JET <- mean(pred_df$JET)
pred_df$JET_pred_mean <- mean_JET

# 2) Simple lag (JET_t = JET_t-1)
pred_df$JET_pred_simple_lag <- pred_df$JET_lag1  # Just copying the last value

# 3) Linear Regression Model using predictors (ENSO, SPV)
lm_model <- lm(JET ~ ENSO + SPV, data = pred_df)
pred_df$JET_pred_lm <- predict(lm_model, newdata = pred_df)
summary(lm_model)

# 4) AR(1) model
ar1_model <- ar(pred_df$JET, order.max = 1, method = "mle", aic = FALSE)  
phi1 <- ar1_model$ar  # AR(1) coefficient
intercept <- ar1_model$x.mean  # Mean of JET (used as intercept)
# Compute fitted values manually
pred_df$JET_pred_ar1 <- intercept + phi1 * lag(pred_df$JET, 1)


# 5) GAM model to allow for non-linear relationships
gam_model <- gam(JET ~ s(ENSO) + s(SPV) , data = pred_df)
pred_df$JET_pred_gam <- predict(gam_model, newdata = pred_df)
par(mfrow=c(1,2))
plot(gam_model)

# Compute RMSEs
compute_rmse <- function(actual, predicted) {
  sqrt(mean((actual - predicted)^2, na.rm = TRUE))
}

rmse_results <- data.frame(
  Model = c("Mean", "Simple Lag", "Linear Regression", "AR(1)", "GAM"),
  RMSE = c(
    compute_rmse(pred_df$JET, pred_df$JET_pred_mean),
    compute_rmse(pred_df$JET, pred_df$JET_pred_simple_lag),
    compute_rmse(pred_df$JET, pred_df$JET_pred_lm),
    compute_rmse(pred_df$JET[-1], pred_df$JET_pred_ar1[-1]),  # Remove first NA
    compute_rmse(pred_df$JET, pred_df$JET_pred_gam)
  ),
  Description = c(
    "Mean JET as prediction",
    "Lagged JET as prediction",
    "LinearRegression using BK, Ural, NP",
    "Lagged JET with coefficient as prediction",
    "Same as Linear Regression but with varying coefficients")
  
)


pred_df$Year <- enso_y
pred_df_plot <- pred_df[-1,]  # Remove first row with NA



p <- ggplot(pred_df_plot, aes(x = Year)) +
  geom_line(aes(y = JET, color = "True JET"), linewidth = 0.8) +
  geom_line(aes(y = JET_pred_mean, color = "Baseline Mean"), linetype = "dashed") +
  geom_line(aes(y = JET_pred_simple_lag, color = "Simple Lag (1)"), linetype = "dashed") +
  geom_line(aes(y = JET_pred_lm, color = "Linear Regression"), linewidth = 0.8) +
  geom_line(aes(y = JET_pred_ar1, color = "AR(1)")) +
  geom_line(aes(y = JET_pred_gam, color = "GAM"), linewidth = 0.8) +
  labs(title = "Predicted vs Observed JET", x = "Time", y = "JET") +
  scale_color_manual(
    name = "Model",
    values = c("True JET" = "black", "Baseline Mean" = "red", "Simple Lag (1)" = "blue",
               "Linear Regression" = "green", "AR(1)" = "purple", "GAM" = "orange")
  ) +
  # line thickness of True JET (in the plot)
  theme_minimal()


p

rmse_results$RMSE <- round(rmse_results$RMSE,3)

# Print RMSE results
print(rmse_results)



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
    
    
    # 1) Baseline model (predict mean JET)
    mean_JET <- mean(train_data$JET)
    test_data$JET_pred_mean <- mean_JET
    
    # 2) Simple lag (JET_t = JET_t-1)
    test_data$JET_pred_simple_lag <- test_data$JET_lag1
    
    # 3) Linear Regression Model using predictors (BK, Ural, NP)
    lm_model <- lm(JET ~ ENSO + SPV, data = train_data)
    test_data$JET_pred_lm <- predict(lm_model, newdata = test_data)
    
    # 4) AR(1) model
    ar1_model <- ar(train_data$JET, order.max = 1, method = "mle", aic = FALSE) # set aic to false to avoid order=0
    phi1 <- ar1_model$ar
    intercept <- ar1_model$x.mean
    test_data$JET_pred_ar1 <- intercept + phi1 * test_data$JET_lag1
    
    # 5) GAM model to allow for non-linear relationships
    gam_model <- gam(JET ~ s(ENSO) + s(SPV) , data = train_data)
    test_data$JET_pred_gam <- predict(gam_model, newdata = test_data)
    
    scatter_df <- rbind(scatter_df, test_data)
    
    # Compute RMSEs for this fold
    rmse_results_fold <- data.frame(
      Model = c("Mean", "Simple Lag", "Linear Regression", "AR(1)", "GAM"),
      RMSE = c(
        compute_rmse(test_data$JET, test_data$JET_pred_mean),
        compute_rmse(test_data$JET[-1], test_data$JET_pred_simple_lag[-1]),
        compute_rmse(test_data$JET, test_data$JET_pred_lm),
        compute_rmse(test_data$JET[-1], test_data$JET_pred_ar1[-1]),
        compute_rmse(test_data$JET, test_data$JET_pred_gam)
      )
    )
    
    # Aggregate RMSEs across folds
    rmse_results_cv$RMSE <- rmse_results_cv$RMSE + rmse_results_fold$RMSE
  }
  
  # Average RMSEs across folds
  rmse_results_cv$RMSE <- rmse_results_cv$RMSE / folds
  return(list(cv = rmse_results_cv, scatter = scatter_df))
}

# Create lagged JET for simple lag and AR models
pred_df <- raw_df %>%
  mutate(JET_lag1 = lag(JET, 1))

# Compute RMSEs using 10-fold cross-validation
rmse_results_cv <- perform_cv(pred_df)


rmse_results_cv$cv$RMSE <- round(rmse_results_cv$cv$RMSE,3)
print(rmse_results_cv$cv)


# scatter plots of true vs predicted values
scatter_cv <- rmse_results_cv$scatter

# Set up a 3x2 plotting layout
par(mfrow=c(3,2), mar=c(4,4,2,1))  # Adjust margins for better spacing

plot(scatter_cv$JET, scatter_cv$JET_pred_mean, 
     xlab = "True JET", ylab = "Predicted JET (Mean)", 
     main = "Mean", xlim=c(-2.1, 2.1), ylim=c(-2.1, 2.1))
abline(0, 1, col="red", lwd=2, lty=2)

plot(scatter_cv$JET, scatter_cv$JET_pred_simple_lag, 
     xlab = "True JET", ylab = "Predicted JET (Simple Lag)", 
     main = "Simple Lag", xlim=c(-2.1, 2.1), ylim=c(-2.1, 2.1))
abline(0, 1, col="red", lwd=2, lty=2)

plot(scatter_cv$JET, scatter_cv$JET_pred_lm, 
     xlab = "True JET", ylab = "Predicted JET (Linear Regression)", 
     main = "Linear Regression", xlim=c(-2.1, 2.1), ylim=c(-2.1, 2.1))
abline(0, 1, col="red", lwd=2, lty=2)

plot(scatter_cv$JET, scatter_cv$JET_pred_ar1, 
     xlab = "True JET", ylab = "Predicted JET (AR(1))", 
     main = "AR(1)", xlim=c(-2.1, 2.1), ylim=c(-2.1, 2.1))
abline(0, 1, col="red", lwd=2, lty=2)

plot(scatter_cv$JET, scatter_cv$JET_pred_gam, 
     xlab = "True JET", ylab = "Predicted JET (GAM)", 
     main = "GAM", xlim=c(-2.1, 2.1), ylim=c(-2.1, 2.1))
abline(0, 1, col="red", lwd=2, lty=2)

# Add an empty plot for layout balance
plot.new()

