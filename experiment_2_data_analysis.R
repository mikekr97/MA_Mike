

#==============================================================================
# Read real data (Experiment 2)
#==============================================================================

library(ncdf4)





ca <- nc_open("data/cali_precip_mean.nc")
ca_y <- ncvar_get(ca, "season_year")
ca_v <- ncvar_get(ca, "prate")



jet <- nc_open("data/cali_jet_mean.nc")
jet_y <- ncvar_get(jet, "season_year")
jet_v <- ncvar_get(jet, "slp")


enso <- nc_open("data/enso_djf.nc")
enso_y <- ncvar_get(enso, "Year")
enso_v <- ncvar_get(enso, "enso")


# plot all together
par(mfrow=c(3,1))
plot(enso_y, enso_v, type = "l", xlab = "Year", ylab = "ENSO")
plot(jet_y, jet_v, type = "l", xlab = "Year", ylab = "JET")
plot(ca_y, ca_v, type = "l", xlab = "Year", ylab = "CA")



#===============================================================================
# Data preprocessing
#===============================================================================

# standardize the 3 variables (zero to mean, unit variance)
ENSO <- (enso_v - mean(enso_v)) / sd(enso_v)
JET <- (jet_v - mean(jet_v)) / sd(jet_v)
CA <- (ca_v - mean(ca_v)) / sd(ca_v)



# 3 histograms of the standardized data
par(mfrow=c(3,1))
hist(ENSO, main = "ENSO", xlab = "ENSO")
hist(JET, main = "JET", xlab = "JET")
hist(CA, main = "CA", xlab = "CA")

# detrend with lm
ENSO_detrended <- residuals(lm(ENSO ~ enso_y))
JET_detrended <- residuals(lm(JET ~ jet_y))
CA_detrended <- residuals(lm(CA ~ ca_y))



# plot the 3 detrended variables (Title "standardized-detrended")
par(mfrow=c(3,1))
plot(enso_y, ENSO_detrended, type = "l", xlab = "Year", ylab = "ENSO")
plot(jet_y, JET_detrended, type = "l", xlab = "Year", ylab = "JET")
plot(ca_y, CA_detrended, type = "l", xlab = "Year", ylab = "SPV")

# 3 histograms of the detrended data
par(mfrow=c(3,1))
hist(ENSO_detrended, main = "ENSO", xlab = "ENSO")
hist(JET_detrended, main = "JET", xlab = "JET")
hist(CA_detrended, main = "CA", xlab = "CA")



raw_df <- data.frame(
  ENSO = ENSO_detrended,
  JET = JET_detrended,
  CA = CA_detrended
)



# Pairsplot of variables
pairs(raw_df)



#===============================================================================
# Re-Creating Experiment 2 from paper 
#===============================================================================


#================================================================
# Determine the effect of ENSO on CA conditioned on Jet
#================================================================

lm_ca_enso_jet <- lm(CA ~ ENSO + JET, data = raw_df)

ce_enso_ca_cond_jet  <- coef(lm_ca_enso_jet)[2]
ce_jet_ca <- coef(lm_ca_enso_jet)[3]

cat("The regression coeff of ENSO on CA conditioned on Jet is ", round(ce_enso_ca_cond_jet,2))
cat("The regression coeff. of Jet on CA is ", round(ce_jet_ca,2))


# The regression coeff of ENSO on CA conditioned on Jet is  0.05
# The regression coeff. of Jet on CA is  0.79

#==========================================================================
# Determine the causal effect of ENSO on CA 
#==========================================================================


lm_ca_enso <- lm(CA ~ ENSO, data = raw_df)

ce_enso_ca  <- coef(lm_ca_enso)[2]

cat("The causal effect of ENSO on CA is ", round(ce_enso_ca,2))
# The causal effect of ENSO on CA is  0.34


#==========================================================================
# Determine the causal effect of ENSO on Jet
#==========================================================================

lm_jet_enso <- lm(JET ~ ENSO, data = raw_df)

ce_enso_jet  <- coef(lm_jet_enso)[2]


cat("The causal effect of ENSO on Jet is ", round(ce_enso_jet,2))

# The causal effect of ENSO on Jet is  0.37

#================================================================
# Determine the causal effect of Jet on CA
#================================================================

lm_ca_jet <- lm(CA ~ JET, data = raw_df)

ce_jet_ca = coef(lm_ca_jet)[2]

cat("The causal effect of Jet on CA is ", round(ce_jet_ca,2))

# The causal effect of Jet on CA is  0.81


#=====================================================
# Path tracing rule:
#=====================================================

ce_along_path = ce_enso_jet * ce_jet_ca 


cat("The indirectly estimated effect of ENSO on CA along the path ENSO -> Jet --> CA is ", round(ce_along_path ,2))
cat("The directly estimated effect of ENSO on CA ", round(ce_enso_ca, 2))

# The indirectly estimated effect of ENSO on CA along the path ENSO -> Jet --> CA is  0.3
# The directly estimated effect of ENSO on CA  0.34


#===============================================================================
# Check predictive value of variables
#===============================================================================



library(mgcv)  # For GAM
library(ggplot2)
library(dplyr)

raw_df

# Create lagged JET for simple lag and AR models
pred_df <- raw_df %>%
  mutate(CA_lag1 = lag(CA, 1))  # Remove NAs from lagging

# 1) Baseline model (predict mean JET)
mean_CA <- mean(pred_df$CA)
pred_df$CA_pred_mean <- mean_CA

# 2) Simple lag (CA_t = CA_t-1)
pred_df$CA_pred_simple_lag <- pred_df$CA_lag1  # Just copying the last value

# 3) Linear Regression Model using predictors (ENSO, SPV)
lm_model <- lm(CA ~ ENSO + JET, data = pred_df)
pred_df$CA_pred_lm <- predict(lm_model, newdata = pred_df)
summary(lm_model)

# 4) AR(1) model
ar1_model <- ar(pred_df$CA, order.max = 1, method = "mle", aic = FALSE)  
phi1 <- ar1_model$ar  # AR(1) coefficient
intercept <- ar1_model$x.mean  # Mean of CA (used as intercept)
# Compute fitted values manually
pred_df$CA_pred_ar1 <- intercept + phi1 * lag(pred_df$CA, 1)


# 5) GAM model to allow for non-linear relationships
gam_model <- gam(CA ~ s(ENSO) + s(JET) , data = pred_df)
pred_df$CA_pred_gam <- predict(gam_model, newdata = pred_df)
par(mfrow=c(1,2))
plot(gam_model)

# Compute RMSEs
compute_rmse <- function(actual, predicted) {
  sqrt(mean((actual - predicted)^2, na.rm = TRUE))
}

rmse_results <- data.frame(
  Model = c("Mean", "Simple Lag", "Linear Regression", "AR(1)", "GAM"),
  RMSE = c(
    compute_rmse(pred_df$CA, pred_df$CA_pred_mean),
    compute_rmse(pred_df$CA, pred_df$CA_pred_simple_lag),
    compute_rmse(pred_df$CA, pred_df$CA_pred_lm),
    compute_rmse(pred_df$CA[-1], pred_df$CA_pred_ar1[-1]),  # Remove first NA
    compute_rmse(pred_df$CA, pred_df$CA_pred_gam)
  ),
  Description = c(
    "Mean CA as prediction",
    "Lagged CA as prediction",
    "LinearRegression using BK, Ural, NP",
    "Lagged CA with coefficient as prediction",
    "Same as Linear Regression but with varying coefficients")
  
)


pred_df$Year <- enso_y
pred_df_plot <- pred_df[-1,]  # Remove first row with NA



p <- ggplot(pred_df_plot, aes(x = Year)) +
  geom_line(aes(y = CA, color = "True CA"), linewidth = 0.8) +
  geom_line(aes(y = CA_pred_mean, color = "Baseline Mean"), linetype = "dashed") +
  geom_line(aes(y = CA_pred_simple_lag, color = "Simple Lag (1)"), linetype = "dashed") +
  geom_line(aes(y = CA_pred_lm, color = "Linear Regression"), linewidth = 0.8) +
  geom_line(aes(y = CA_pred_ar1, color = "AR(1)")) +
  geom_line(aes(y = CA_pred_gam, color = "GAM"), linewidth = 0.8) +
  labs(title = "Predicted vs Observed CA", x = "Time", y = "CA") +
  scale_color_manual(
    name = "Model",
    values = c("True CA" = "black", "Baseline Mean" = "red", "Simple Lag (1)" = "blue",
               "Linear Regression" = "green", "AR(1)" = "purple", "GAM" = "orange")
  ) +
  # line thickness of True CA (in the plot)
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
    
    
    # 1) Baseline model (predict mean CA)
    mean_CA <- mean(train_data$CA)
    test_data$CA_pred_mean <- mean_CA
    
    # 2) Simple lag (CA_t = CA_t-1)
    test_data$CA_pred_simple_lag <- test_data$CA_lag1
    
    # 3) Linear Regression Model using predictors (BK, Ural, NP)
    lm_model <- lm(CA ~ ENSO + JET, data = train_data)
    test_data$CA_pred_lm <- predict(lm_model, newdata = test_data)
    
    # 4) AR(1) model
    ar1_model <- ar(train_data$CA, order.max = 1, method = "mle", aic = FALSE) # set aic to false to avoid order=0
    phi1 <- ar1_model$ar
    intercept <- ar1_model$x.mean
    test_data$CA_pred_ar1 <- intercept + phi1 * test_data$CA_lag1
    
    # 5) GAM model to allow for non-linear relationships
    gam_model <- gam(CA ~ s(ENSO) + s(JET) , data = train_data)
    test_data$CA_pred_gam <- predict(gam_model, newdata = test_data)
    
    scatter_df <- rbind(scatter_df, test_data)
    
    # Compute RMSEs for this fold
    rmse_results_fold <- data.frame(
      Model = c("Mean", "Simple Lag", "Linear Regression", "AR(1)", "GAM"),
      RMSE = c(
        compute_rmse(test_data$CA, test_data$CA_pred_mean),
        compute_rmse(test_data$CA[-1], test_data$CA_pred_simple_lag[-1]),
        compute_rmse(test_data$CA, test_data$CA_pred_lm),
        compute_rmse(test_data$CA[-1], test_data$CA_pred_ar1[-1]),
        compute_rmse(test_data$CA, test_data$CA_pred_gam)
      )
    )
    
    # Aggregate RMSEs across folds
    rmse_results_cv$RMSE <- rmse_results_cv$RMSE + rmse_results_fold$RMSE
  }
  
  # Average RMSEs across folds
  rmse_results_cv$RMSE <- rmse_results_cv$RMSE / folds
  return(list(cv = rmse_results_cv, scatter = scatter_df))
}

# Create lagged CA for simple lag and AR models
pred_df <- raw_df %>%
  mutate(CA_lag1 = lag(CA, 1))

# Compute RMSEs using 10-fold cross-validation
rmse_results_cv <- perform_cv(pred_df)


rmse_results_cv$cv$RMSE <- round(rmse_results_cv$cv$RMSE,3)
print(rmse_results_cv$cv)


# scatter plots of true vs predicted values
scatter_cv <- rmse_results_cv$scatter

# Set up a 3x2 plotting layout
par(mfrow=c(3,2), mar=c(4,4,2,1))  # Adjust margins for better spacing

plot(scatter_cv$CA, scatter_cv$CA_pred_mean, 
     xlab = "True CA", ylab = "Predicted CA (Mean)", 
     main = "Mean", xlim=c(-2.1, 2.1), ylim=c(-2.1, 2.1))
abline(0, 1, col="red", lwd=2, lty=2)

plot(scatter_cv$CA, scatter_cv$CA_pred_simple_lag, 
     xlab = "True CA", ylab = "Predicted CA (Simple Lag)", 
     main = "Simple Lag", xlim=c(-2.1, 2.1), ylim=c(-2.1, 2.1))
abline(0, 1, col="red", lwd=2, lty=2)

plot(scatter_cv$CA, scatter_cv$CA_pred_lm, 
     xlab = "True CA", ylab = "Predicted CA (Linear Regression)", 
     main = "Linear Regression", xlim=c(-2.1, 2.1), ylim=c(-2.1, 2.1))
abline(0, 1, col="red", lwd=2, lty=2)

plot(scatter_cv$CA, scatter_cv$CA_pred_ar1, 
     xlab = "True CA", ylab = "Predicted CA (AR(1))", 
     main = "AR(1)", xlim=c(-2.1, 2.1), ylim=c(-2.1, 2.1))
abline(0, 1, col="red", lwd=2, lty=2)

plot(scatter_cv$CA, scatter_cv$CA_pred_gam, 
     xlab = "True CA", ylab = "Predicted CA (GAM)", 
     main = "GAM", xlim=c(-2.1, 2.1), ylim=c(-2.1, 2.1))
abline(0, 1, col="red", lwd=2, lty=2)

# Add an empty plot for layout balance
plot.new()

