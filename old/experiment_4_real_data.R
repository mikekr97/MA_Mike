
# Experiment 4: Real data (linear shift)

#==============================================================================

# The goal is to simulate the DAG for experiment 4 in the Teleconnections paper
# and then fit the simulated data with the TRAM-DAG and sample from the 
# distributions for different levels of Pearl's causality ladder.

#==============================================================================


# Preparations

# Make sure that the right python environment is used

# When starting a new session, execute following
if (FALSE){
  reticulate::use_python("C:/ProgramData/Anaconda3/python.exe", required = TRUE)
  # library(reticulate)
  # reticulate::py_config()
}



library(tensorflow)
library(keras)
# install keras
library(mlt)
library(tram)
library(MASS)
library(tidyverse)

# load the file with the functions
source('utils_tf.R')

#### For TFP
# reticulate::py_install("tensorflow_probability")
library(tfprobability)
source('utils_tfp.R')

##### Flavor of experiment ######

#### Saving the current version of the script into runtime
DIR = 'runs/experiment_4_real_data_linear/run'
if (!dir.exists(DIR)) {
  dir.create(DIR, recursive = TRUE)
}
# Copy this file to the directory DIR
file.copy('MA_Mike/experiment_4_real_data_linear.R', file.path(DIR, 'experiment_4_real_data_linear.R'), overwrite=TRUE)


#==============================================================================
# Define DGP and Model Structure
#==============================================================================


# Select desired DGA and type of TRAM-DAG shift

# first entry is the type of DGP (1=simple, 4= complex)
# second entry is the type of TRAM-DAG shift (ls=linear, cs=complex)

args <- c(1, 'ls') # 
# args <- c(1, 'cs') # 
# args <- c(4, 'cs') # 


# example of a connection
# F32 <- as.numeric(args[1]) # DGP complex shift
# M32 <- args[2] # Model complex shift
# print(paste("F32:", F32, "M32:", M32))

# example of a connection
F21 <- as.numeric(args[1]) # DGP complex shift
M21 <- args[2] # Model complex shift
print(paste("F21:", F21, "M21:", M21))

num_epochs <- 200  # 500
len_theta = 20 # Number of coefficients of the Bernstein polynomials
hidden_features_I = c(2,25,25,2)    #hidden_features_CS=hidden_features_I = c(2,25,25,2)
hidden_features_CS = c(2,25,25,2)


SEED = -1 #If seed > 0 then the seed is set

# if (F32 == 1){
#   FUN_NAME = 'DPGLinear'
#   f <- function(x) -0.3 * x
# } else if (F32 == 2){
#   f = function(x) 2 * x**3 + x
#   FUN_NAME = 'DPG2x3+x'
# } else if (F32 == 3){
#   f = function(x) 0.5*exp(x)
#   FUN_NAME = 'DPG0.5exp'
# } else if (F32 == 4){
#   f = function(x) 0.75*atan(5*(x+0.12)) 
#   FUN_NAME = 'DPGatan'
# } else if (F32 == 5){
#   f = function(x) 2*sin(3*x)+x 
#   FUN_NAME = 'DPGSin'
# } else {
#   stop("Unknown Function F32")
# }
# 
# # function name for DGP
# FUN_NAME





# if (M32 == 'ls') {
#   MA =  matrix(c(
#     0, 'ls', 'ls', 
#     0,    0, 'ls', 
#     0,    0,   0), nrow = 3, ncol = 3, byrow = TRUE)
#   MODEL_NAME = 'ModelLS'
# } else{
#   MA =  matrix(c(
#     0, 'ls', 'ls', 
#     0,    0, 'cs', 
#     0,    0,   0), nrow = 3, ncol = 3, byrow = TRUE)
#   MODEL_NAME = 'ModelCS'
# }


# order: x1, x2, x3, x4
#       NP, Ural, BK, SPV
MA =  matrix(c(
  0, 0, 'ls', 'ls',
  0, 0, 'ls', 'ls',
  0, 0,  0  , 'ls',
  0, 0,  0  ,   0), nrow = 4, ncol = 4, byrow = TRUE)
MODEL_NAME = 'ModelLS'

# adjacency matrix
MA

# add row and column names
rownames(MA) <- c('NP', 'Ural', 'BK', 'SPV')
colnames(MA) <- c('NP', 'Ural', 'BK', 'SPV')
MA
adjacency <- train$A
rownames(adjacency) <- c('NP', 'Ural', 'BK', 'SPV')
colnames(adjacency) <- c('NP', 'Ural', 'BK', 'SPV')

adjacency

# model name
MODEL_NAME

if (SEED < 0){
  fn = file.path(DIR, paste0('triangle_mixed_',  '_', MODEL_NAME))
} else{
  fn = file.path(DIR, paste0('triangle_mixed_',  '_', MODEL_NAME, '_SEED', SEED))
}
print(paste0("Starting experiment ", fn))




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


# Print RMSE results
print(rmse_results)

#===============================================================================
# 4-fold Cross Validation
#===============================================================================

library(mgcv)  # For GAM
library(ggplot2)
library(dplyr)

# Assuming raw_df, df_avg_bk are already defined

# Function to perform 4-fold cross-validation and calculate RMSE
perform_cv <- function(data, folds = 4) {
  n <- nrow(data)
  fold_indices <- sample(rep(1:folds, length.out = n))
  rmse_results_cv <- data.frame(
    Model = c("Mean", "Simple Lag", "Linear Regression", "AR(1)", "GAM"),
    RMSE = rep(0, 5)
  )
  
  for (i in 1:folds) {
    train_data <- data[fold_indices != i, ]
    test_data <- data[fold_indices == i, ]
    
    # 1) Baseline model (predict mean SPV)
    mean_spv <- mean(train_data$SPV)
    test_data$SPV_pred_mean <- mean_spv
    
    # 2) Simple lag (SPV_t = SPV_t-1)
    train_data <- train_data %>% mutate(SPV_lag1 = lag(SPV,1))
    test_data <- test_data %>% mutate(SPV_lag1 = lag(SPV,1))
    
    test_data$SPV_pred_simple_lag <- lag(train_data$SPV,1)[nrow(train_data)]
    test_data$SPV_pred_simple_lag[2:nrow(test_data)] <- test_data$SPV_lag1[2:nrow(test_data)]
    
    # 3) Linear Regression Model using predictors (BK, Ural, NP)
    lm_model <- lm(SPV ~ BK + Ural + NP, data = train_data)
    test_data$SPV_pred_lm <- predict(lm_model, newdata = test_data)
    
    # 4) AR(1) model
    ar1_model <- ar(train_data$SPV, order.max = 1, method = "mle")
    phi1 <- ar1_model$ar
    intercept <- ar1_model$x.mean
    test_data$SPV_pred_ar1 <- intercept + phi1 * lag(c(train_data$SPV, test_data$SPV[1:(nrow(test_data)-1)]),1)[(nrow(train_data)+1):length(c(train_data$SPV, test_data$SPV[1:(nrow(test_data)-1)]))]
    
    # 5) GAM model to allow for non-linear relationships
    gam_model <- gam(SPV ~ s(BK) + s(Ural) + s(NP), data = train_data)
    test_data$SPV_pred_gam <- predict(gam_model, newdata = test_data)
    
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
  return(rmse_results_cv)
}

# Create lagged SPV for simple lag and AR models
pred_df <- raw_df %>%
  mutate(SPV_lag1 = lag(SPV, 1))

# Compute RMSEs using 4-fold cross-validation
rmse_results_cv <- perform_cv(pred_df)

# Add Year and process for plotting
pred_df$Year <- df_avg_bk$year[-length(df_avg_bk$year)]
pred_df_plot <- pred_df[-1, ]  # Remove first row with NA

# Plotting code remains the same
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
  theme_minimal()

print(p)
print(rmse_results_cv)

#===============================================================================
# Create train and test data
#===============================================================================


set.seed(123)

# train test split (90% train, 10% test)
train_indizes <- sample(1:nrow(raw_df), 0.9*nrow(raw_df))
train_df <- raw_df[train_indizes,]
test_df <- raw_df[-train_indizes,]


# create the data object

dgp <- function(n_obs, doX=c(NA, NA, NA, NA), seed=-1, data = NULL) {
  
  # n_obs <- 1000
  # doX <- c(NA, NA, NA, NA)
  # seed=-1
  
  if (seed > 0) {
    set.seed(seed)
    print(paste0("Setting Seed:", seed))
  }
  
  
  if (is.na(doX[1])){
    X_1 = data$NP
  } else{
    X_1 = rep(doX[1], n_obs)
  }
  #hist(X_1)
  
  if (is.na(doX[2])){
    
    X_2 = data$Ural
    # question: why scaled with 1/5.?
    
    
  } else{
    X_2 = rep(doX[2], n_obs)
  }
  #hist(X_2)
  
  if (is.na(doX[3])){
    
    X_3 = data$BK
  } else{
    X_3 = rep(doX[3], n_obs)
  }
  
  if (is.na(doX[4])){
    
    X_4 = data$SPV
  } else{
    X_4 = rep(doX[4], n_obs)
  }
  
  #hist(X_3)
  # par(mfrow=c(1,3))
  # hist(X_1)
  # hist(X_2)
  # hist(X_3)
  
  # Define adjacency matrix
  A <- matrix(c(0, 0, 1, 1, 0,0,1, 1,0,0,0, 1, 0, 0 ,0, 0), nrow = 4, ncol = 4, byrow = TRUE)
  # Put smaples in a dataframe
  dat.orig =  data.frame(x1 = X_1, x2 = X_2, x3 = X_3, x4 = X_4)
  # Put samples in a tensor
  dat.tf = tf$constant(as.matrix(dat.orig), dtype = 'float32')
  
  # calculate 5% quantiles of the sampled variables
  q1 = quantile(dat.orig[,1], probs = c(0.05, 0.95)) 
  q2 = quantile(dat.orig[,2], probs = c(0.05, 0.95))
  q3 = quantile(dat.orig[,3], probs = c(0.05, 0.95))
  q4 = quantile(dat.orig[,4], probs = c(0.05, 0.95))
  
  
  # return samples in a tensor, the original dataframe, the min and max values 
  # of the variables (the quantiles), and the adjacency matrix
  return(list(
    df_orig=dat.tf, 
    df_R = dat.orig,
    #min =  tf$reduce_min(dat.tf, axis=0L),
    #max =  tf$reduce_max(dat.tf, axis=0L),
    min = tf$constant(c(q1[1], q2[1], q3[1], q4[1]), dtype = 'float32'),
    max = tf$constant(c(q1[2], q2[2], q3[2], q4[2]), dtype = 'float32'),
    type = c('c', 'c', 'c', 'c'),
    A=A))
} 

# generate data
train <- dgp(nrow(train_df), seed=ifelse(SEED > 0, SEED, -1), data = train_df)
test <- dgp(nrow(test_df), seed=ifelse(SEED > 0, SEED, -1), data = test_df)

test$df_R


(global_min = train$min) # the lower 5% quantiles
(global_max = train$max) # the upper 5% quantiles
data_type = train$type   # c for continuous, o for ordinal



#==============================================================================
# Define the TRAM-DAG model
#==============================================================================



len_theta_max = len_theta # 20
for (i in 1:nrow(MA)){ #Maximum number of coefficients (BS and Levels - 1 for the ordinal)
  if (train$type[i] == 'o'){  # 'o' for ordinal
    len_theta_max = max(len_theta_max, nlevels(train$df_R[,i]) - 1)
  }
}



param_model = create_param_model(MA, hidden_features_I = hidden_features_I, 
                                 len_theta = len_theta_max, 
                                 hidden_features_CS = hidden_features_CS)

summary(param_model)


#==============================================================================
# set custom weights

# input the samples into the model
h_params = param_model(train$df_orig)

# loss before training  # 2.76
struct_dag_loss(t_i=train$df_orig, h_params=h_params)
param_model = create_param_model(MA, hidden_features_I=hidden_features_I, len_theta=len_theta, hidden_features_CS=hidden_features_CS)
optimizer = optimizer_adam(learning_rate=0.005)

# set weight (betas) according to Colr
# 
# fit.21 = Colr(x2~x1,data = train$df_R, order=len_theta)
# summary(fit.21)
# fit.3 = Colr(x3~x1 + x2, data = train$df_R, order = len_theta)
# summary(fit.3)
# 
# beta_matrix_colr <- matrix(c(0,coef(fit.21), coef(fit.3)[1],
#                              0,0,coef(fit.3)[2],
#                              0,0,0), nrow=3, byrow = TRUE)
# 
# # beta_matrix_colr <- matrix(c(0,1.95, -0.2,
# #                              0,0,0.30,
# #                              0,0,0), nrow=3, byrow = TRUE)
# 
# # Extract current weights (the betas in the Adjacency Matrix)
# param_model$get_layer(name = "beta")$get_weights()[[1]]
# 
# # Set weights to Colr estimates
# param_model$get_layer(name = "beta")$set_weights(list(tf$constant(beta_matrix_colr, dtype = 'float32')))

# Check newly set weights
param_model$get_layer(name = "beta")$get_weights()[[1]]


h_params = param_model(train$df_orig)
# loss before training after beta initialization  # 2.98
struct_dag_loss(t_i=train$df_orig, h_params=h_params)


#==============================================================================


param_model$compile(optimizer, loss=struct_dag_loss)
param_model$evaluate(x = train$df_orig, y=train$df_orig, batch_size = 7L)




#==============================================================================
# Train the TRAM-DAG model
#==============================================================================
num_epochs <- 10000

##### Training or readin of weights if h5 available ####
fnh5 = paste0(fn, '_E', num_epochs, '.h5')
fnRdata = paste0(fn, '_E', num_epochs, '.RData')
if (file.exists(fnh5)){
  param_model$load_weights(fnh5)
  load(fnRdata) #Loading of the workspace causes trouble e.g. param_model is zero
  # Quick Fix since loading global_min causes problem (no tensors as RDS)
  (global_min = train$min)
  (global_max = train$max)
} else {
  if (FALSE){ ### Full Training w/o diagnostics
    hist = param_model$fit(x = train$df_orig, y=train$df_orig, epochs = 200L,verbose = TRUE)
    param_model$save_weights(fn)
    plot(hist$epoch, hist$history$loss)
    plot(hist$epoch, hist$history$loss, ylim=c(1.07, 1.2))
  } else { ### Training with diagnostics
    ws <- data.frame(w13 = numeric())
    train_loss <- numeric()
    val_loss <- numeric()
    
    # Training loop
    for (e in 1:num_epochs) {
      print(paste("Epoch", e))
      hist <- param_model$fit(x = train$df_orig, y = train$df_orig, 
                              epochs = 1L, verbose = TRUE, 
                              validation_data = list(test$df_orig,test$df_orig))
      
      # Append losses to history
      train_loss <- c(train_loss, hist$history$loss)
      val_loss <- c(val_loss, hist$history$val_loss)
      
      # Extract specific weights (the betas in the Adjacency Matrix)
      w <- param_model$get_layer(name = "beta")$get_weights()[[1]]
      
      # Append the directed weights for this epoch
      ws <- rbind(ws, data.frame(w13 = w[1, 3], w14 = w[1, 4], w23 = w[2, 3], w24 = w[2, 4], w34 = w[3, 4]))
    }
    # Save the model
    param_model$save_weights(fnh5)
    f = "OriginalData"
    save(train_loss, val_loss, train_loss, f, MA, len_theta,
         hidden_features_I,
         hidden_features_CS,
         ws,
         #global_min, global_max,
         file = fnRdata)
  }
}

tail(ws)

par(mfrow=c(1,1))

####### FINISHED TRAINING #####
#pdf(paste0('loss_',fn,'.pdf'))
epochs = length(train_loss)
plot(1:length(train_loss), train_loss, type='l', main='Training (black: train, green: valid)')
lines(1:length(train_loss), val_loss, type = 'l', col = 'green')

# Last 50
diff = max(epochs - 50,1)
plot(diff:epochs, val_loss[diff:epochs], type = 'l', col = 'green', main='Last 50 epochs')
lines(diff:epochs, train_loss[diff:epochs], type='l')



ws <- rbind(ws, data.frame(w13 = w[1, 3], w14 = w[1, 4], w23 = w[2, 3], w24 = w[2, 4], w34 = w[3, 4]))


# fit.13 = Colr(x3~x1, data = train$df_R, order = len_theta)
# fit.23 = Colr(x3~x2, data = train$df_R, order = len_theta)
fit.3 = Colr(x3~x1 + x2, data = train$df_R, order = len_theta)
fit.4 = Colr(x4~x1 + x2 + x3, data = train$df_R, order = len_theta)



#### Plotting of the Loss Curve ##########
tail(ws) # betas for the last 5 epochs
p = ggplot(ws, aes(x=1:nrow(ws))) + 
  geom_line(aes(y=w13, color="beta13")) + 
  geom_line(aes(y=w14, color="beta14")) + 
  geom_line(aes(y=w23, color="beta23")) + 
  geom_line(aes(y=w24, color="beta24")) + 
  geom_line(aes(y=w34, color="beta34")) + 
  geom_hline(aes(yintercept=coef(fit.3)[1], color="beta13"), linetype=2) +
  geom_hline(aes(yintercept=coef(fit.3)[2], color="beta23"), linetype=2) +
  geom_hline(aes(yintercept=coef(fit.4)[1], color="beta14"), linetype=2) +
  geom_hline(aes(yintercept=coef(fit.4)[2], color="beta24"), linetype=2) +
  geom_hline(aes(yintercept=coef(fit.4)[3], color="beta34"), linetype=2) +
  scale_color_manual(
    values=c('beta13'='skyblue', 'beta14'='red', 'beta23'='darkgreen',
             'beta24'='green', 'beta34'='purple'),
    labels=c(expression(beta[13]), expression(beta[14]), expression(beta[23]),
             expression(beta[24]), expression(beta[34]))
  ) +
  labs(x='Epoch', y='Coefficients') +
  theme_minimal() +
  theme(
    legend.title = element_blank(),   # Removes the legend title
    legend.position = c(0.85, 0.25),  # Adjust this to position the legend inside the plot (lower-right)
    legend.background = element_rect(fill="white", colour="black")  # Optional: white background with border
  )
# if (F32 == 4){ # We don't have beta23
if (F21 == 4){
  p =  ggplot(ws, aes(x=1:nrow(ws))) + 
    geom_line(aes(y=w23, color="beta23")) + 
    geom_line(aes(y=w13, color="beta13")) + 
    geom_hline(aes(yintercept=coef(fit.3)[2], color="beta23"), linetype=2) +
    geom_hline(aes(yintercept=coef(fit.3)[1], color="beta13"), linetype=2) +
    scale_color_manual(
      values=c('beta23'='skyblue', 'beta13'='red'),
      labels=c(expression(beta[13]), expression(beta[23]))
    ) +
    labs(x='Epoch', y='Coefficients') +
    theme_minimal() +
    theme(
      legend.title = element_blank(),   # Removes the legend title
      legend.position = c(0.85, 0.25),  # Adjust this to position the legend inside the plot (lower-right)
      legend.background = element_rect(fill="white", colour="black")  # Optional: white background with border
    )
}
p
if (FALSE){
  ### NOTE THAT WE RENAMED THE PRODUCED FILE and added the PATH due to a naming conflict (the continuous files are wrongly named mixed)
  # Remove 'mixed' in filename
  file_name <- paste0(fn, "_coef_epoch.pdf")
  file_name <- gsub("mixed", "", file_name)
  file_path <- file.path("runs/experiment_4_real_data_linear/run/", basename(file_name))
  ggsave(file_path, plot = p, width = 8, height = 6/2)  
}


if (FALSE){
  # Creating the figure for the paper 
  # triangle_mixed_DPGLinear_ModelLS_coef_epoch 
  p = ggplot(ws, aes(x=1:nrow(ws))) + 
    geom_line(aes(y=w12, color="beta12")) + 
    geom_line(aes(y=w13, color="beta13")) + 
    geom_line(aes(y=w23, color="beta23")) + 
    geom_hline(aes(yintercept=2, color="beta12"), linetype=2) +
    geom_hline(aes(yintercept=-0.2, color="beta13"), linetype=2) +
    geom_hline(aes(yintercept=+0.3, color="beta23"), linetype=2) +
    scale_color_manual(
      values=c('beta12'='skyblue', 'beta13'='red', 'beta23'='darkgreen'),
      labels=c(expression(beta[12]), expression(beta[13]), expression(beta[23]))
    ) +
    labs(x='Epoch', y='Coefficients') +
    theme_minimal() +
    theme(
      legend.title = element_blank(),   # Removes the legend title
      legend.position = c(0.85, 0.25),  # Adjust this to position the legend inside the plot (lower-right)
      legend.background = element_rect(fill="white", colour="black")  # Optional: white background with border
    )
  
  file_name <- paste0(fn, "_coef_epoch.pdf")
  # Save the plot
  ggsave(file_name, plot = p, width = 8, height = 6)
  file_path <- file.path("runs/experiment_4_real_data_linear/run/", basename(file_name))
  ggsave(file_path, plot = p, width = 8/2, height = 6/2)
}




#param_model$evaluate(x = train$df_orig, y=train$df_scaled) #Does not work, probably TF Eager vs Compiled

# One more step to estimate NLL 
# (make t.test of 10 NLL after fitting to check if model is stable)
if (FALSE){
  vals = NULL
  for (i in 1:10){
    test  = dgp(40000, i+10001)
    hist = param_model$fit(x = train$df_orig, y = train$df_orig, 
                           epochs = 1L, verbose = TRUE, 
                           validation_data = list(test$df_orig,test$df_orig))
    vals = append(vals, hist$history$val_loss)
  }
  t.test(vals)
  M32
  F32
}
fn
len_theta

# Adjacency Matrix with estimated parameters
param_model$get_layer(name = "beta")$get_weights() * param_model$get_layer(name = "beta")$mask


#==============================================================================
# Sample Observational and Interventional Distribution
#==============================================================================


###### Figure for paper Observational and Do intervention ######
if (TRUE){
  
  # observational distribution
  doX=c(NA, NA, NA, NA)
  s_obs_fitted = do_dag_struct(param_model, train$A, doX, num_samples = 5000)$numpy()
  
  
  # do intervention
  dx3 = -1.5
  #dx1 = 1.5
  doX=c(NA, NA, dx3, NA)
  s_do_fitted = do_dag_struct(param_model, train$A, doX=doX)$numpy()
  
  # add the doX to the plot
  df = data.frame(vals=s_obs_fitted[,1], type='Model', X=1, L='L0')
  df = rbind(df, data.frame(vals=s_obs_fitted[,2], type='Model', X=2, L='L0'))
  df = rbind(df, data.frame(vals=s_obs_fitted[,3], type='Model', X=3, L='L0'))
  df = rbind(df, data.frame(vals=s_obs_fitted[,4], type='Model', X=4, L='L0'))
  
  df = rbind(df, data.frame(vals=train$df_R[,1], type='Real Data', X=1, L='L0'))
  df = rbind(df, data.frame(vals=train$df_R[,2], type='Real Data', X=2, L='L0'))
  df = rbind(df, data.frame(vals=train$df_R[,3], type='Real Data', X=3, L='L0'))
  df = rbind(df, data.frame(vals=train$df_R[,4], type='Real Data', X=4, L='L0'))
  
  df = rbind(df, data.frame(vals=s_do_fitted[,1], type='Model', X=1, L='L1'))
  df = rbind(df, data.frame(vals=s_do_fitted[,2], type='Model', X=2, L='L1'))
  df = rbind(df, data.frame(vals=s_do_fitted[,3], type='Model', X=3, L='L1'))
  df = rbind(df, data.frame(vals=s_do_fitted[,4], type='Model', X=4, L='L1'))
  
  # d = dgp(10000, doX=doX)$df_R
  # d = dgp(nrow(train_df), doX=doX, data=train_df)$df_R  # not possible for real data
  # df = rbind(df, data.frame(vals=d[,1], type='DGP', X=1, L='L1'))
  # df = rbind(df, data.frame(vals=d[,2], type='DGP', X=2, L='L1'))
  # df = rbind(df, data.frame(vals=as.numeric(d[,3]), type='DGP', X=3, L='L1'))
  
  p = ggplot() +
    geom_histogram(data = df, 
                   aes(x=vals, col=type, fill=type, y=..density..), 
                   position = "identity", alpha=0.4) +
    facet_grid(L ~ X, scales = 'free_y',
               labeller = as_labeller(c('1' = 'X1', '2' = 'X2', '3' = 'X3', '4' = 'X4', 'L1' = paste0('Do X3=', dx3), 'L0' = 'Obs'))) +
    labs(y = "Density", x='Values') + # Update y-axis label
    theme_minimal() +
    theme(
      legend.title = element_blank(),   # Removes the legend title
      legend.position = c(0.17, 0.25),  # Adjust this to position the legend inside the plot (lower-right)
      legend.background = element_rect(fill="white", colour="white")  # Optional: white background with border
    ) +
    facet_grid(L ~ X, scales = "free",
               labeller = as_labeller(c('1' = 'X1', '2' = 'X2', '3' = 'X3', '4' = 'X4', 'L1' = paste0('Do X3=', dx3), 'L0' = 'Obs'))) +
    coord_cartesian(ylim = c(0, 2), xlim = NULL) # Adjust y-axis zoom for facets 
  p
  
  file_name <- paste0(fn, "_L0_L1.pdf")
  file_name <- gsub("mixed", "", file_name) #We have wrongly mixed in fn
  if (TRUE){
    file_path <- file.path("runs/experiment_4_real_data_linear/run/", basename(file_name))
    print(file_path)
    ggsave(file_path, plot=p, width = 8/2, height = 6/2)
  }
  
}




#### Checking the transformation ####
h_params = param_model(train$df_orig)
r = check_baselinetrafo(h_params)
Xs = r$Xs
h_I = r$h_I

# check max of each row in Xs
apply(Xs, 2, max)


par(mfrow=c(1,4))

##### X1 (added, not sure if correct)

df = data.frame(train$df_orig$numpy())
# check max of each row in Xs
apply(df, 2, max)

fit.1 = Colr(X1~0,df, order=len_theta)
temp = model.frame(fit.1)[1:2,-1, drop=FALSE] #WTF!  -> yields the values for h_I(x2_min), h_I(x2_(min+1))
# plots the transfomration function fitted by Colr()
plot(fit.1, which = 'baseline only', newdata = temp, lwd=2, col='blue', 
     main='h_I(X1) Black: COLR, Red: Our Model', cex.main=0.8)
# add for the range over q0.05 to q0.95 the transformation function (Intercept h_I) fitted by the NN
lines(Xs[,1], h_I[,1], col='red', lty=2, lwd=5)
rug(train$df_orig$numpy()[,1], col='blue')


### X2

fit.2 = Colr(X2~0,df, order=len_theta)
temp = model.frame(fit.2)[1:2,-1, drop=FALSE] #WTF!  -> yields the values for h_I(x2_min), h_I(x2_(min+1))
# plots the transfomration function fitted by Colr()
plot(fit.2, which = 'baseline only', newdata = temp, lwd=2, col='blue', 
     main='h_I(X2) Black: COLR, Red: Our Model', cex.main=0.8)
# add for the range over q0.05 to q0.95 the transformation function (Intercept h_I) fitted by the NN
lines(Xs[,2], h_I[,2], col='red', lty=2, lwd=5)
rug(train$df_orig$numpy()[,2], col='blue')

##### X3
fit.312 = Colr(X3 ~ X1 + X2,df, order=len_theta)
temp = model.frame(fit.312)[1:2, -1, drop=FALSE] #WTF!

plot(fit.312, which = 'baseline only', newdata = temp, lwd=2, col='blue', 
     main='h_I(X3) Colr and Our Model', cex.main=0.8)
lines(Xs[,3], h_I[,3], col='red', lty=2, lwd=5)
rug(train$df_orig$numpy()[,3], col='blue')


##### X4
fit.4123 = Colr(X4 ~ X1 + X2 + X3,df, order=len_theta)
temp = model.frame(fit.4123)[1:2, -1, drop=FALSE] #WTF!

plot(fit.4123, which = 'baseline only', newdata = temp, lwd=2, col='blue', 
     main='h_I(X3) Colr and Our Model', cex.main=0.8)
lines(Xs[,4], h_I[,4], col='red', lty=2, lwd=5)
rug(train$df_orig$numpy()[,4], col='blue')





##### Checking observational distribution ####
library(car)
s = do_dag_struct(param_model, train$A, doX=c(NA, NA, NA, NA), num_samples = 5000)
par(mfrow=c(1,4))
for (i in 1:4){
  d = s[,i]$numpy()
  hist(train$df_orig$numpy()[,i], freq=FALSE, 100,main=paste0("X",i, " red: ours, black: data"), xlab='samples')
  lines(density(train$df_orig$numpy()[,i]), col='blue')
  #hist(train$df_orig$numpy()[,i], freq=FALSE, 100,main=paste0("X_",i))
  lines(density(s[,i]$numpy()), col='red')
  #qqplot(train$df_orig$numpy()[,i], s[,i]$numpy())
  #abline(0,1)
}
par(mfrow=c(1,1))



######### Simulation of do-interventions #####
doX=c(0.2, NA, NA)
dx0.2 = dgp(10000, doX=doX, seed=SEED)
dx0.2$df_orig$numpy()[1:5,]


doX=c(0.7, NA, NA)
dx7 = dgp(10000, doX=doX, seed=SEED)
#hist(dx0.2$df_orig$numpy()[,2], freq=FALSE,100)
mean(dx7$df_orig$numpy()[,2]) - mean(dx0.2$df_orig$numpy()[,2])  
mean(dx7$df_orig$numpy()[,3]) - mean(dx0.2$df_orig$numpy()[,3])  

s_dag = do_dag_struct(param_model, train$A, doX=c(0.2, NA, NA))
hist(dx0.2$df_orig$numpy()[,2], freq=FALSE, 50, main='X2 | Do(X1=0.2)', xlab='samples', 
     sub='Histogram from DGP with do. red:TRAM_DAG')
sample_dag_0.2 = s_dag[,2]$numpy()
lines(density(sample_dag_0.2), col='red', lw=2)
m_x2_do_x10.2 = median(sample_dag_0.2)

doX=c(NA, -1, NA)
s_dag = do_dag_struct(param_model, train$A, doX)
sdgp = dgp(10000, doX=doX, seed=SEED)
hist(sdgp$df_orig$numpy()[,3], freq=FALSE, 50, xlab='samples', 
     sub='Histogram from DGP with do. red:TRAM_DAG')
sample_dag = s_dag[,3]$numpy()
lines(density(sample_dag), col='red', lw=2)

doX=c(1, NA, NA)
s_dag = do_dag_struct(param_model, train$A, doX)
sdgp = dgp(10000, doX=doX, seed=SEED)
hist(sdgp$df_orig$numpy()[,2], freq=FALSE, 50, xlab='samples', 
     sub='Histogram from DGP with do. red:TRAM_DAG')
sample_dag = s_dag[,2]$numpy()
lines(density(sample_dag), col='red', lw=2)



# 
# ###### Comparison of estimated f(x2) vs TRUE f(x2) #######
# shift_23 = shift_13 = shift_13 = cs_12 = xs = seq(-1,1,length.out=41)
# idx0 = which(xs == 0) #Index of 0 xs needs to be odd
# for (i in 1:length(xs)){
#   #i = 1
#   x = xs[i]
#   # Varying x1
#   X = tf$constant(c(x, 0.5, -3), shape=c(1L,3L)) 
#   shift_13[i] =   param_model(X)[1,3,2]$numpy() #2=LS Term X1->X3
#   cs_12[i] = param_model(X)[1,2,1]$numpy() #1=CS Term X1->X2
#   
#   #Varying x2
#   X = tf$constant(c(0.5, x, 3), shape=c(1L,3L)) 
#   shift_23[i] = param_model(X)[1,3,2]$numpy() #2-LS Term X2-->X3 (Mrs. Whites' Notation)
# }
# par(mfrow=c(1,3))
# plot(xs, shift_13, type='l', col='red', lwd=2, xlab='x1', ylab='f(x1)', main='b13 * x1 (linear)')
# plot(xs, cs_12, type='l', col='red', lwd=2, xlab='x1', ylab='f(x1)', main='b12(x1) complex')
# plot(xs, shift_23, type='l', col='red', lwd=2, xlab='x2', ylab='f(x2)', main='b23 * x2 (linear)')
# 



###### Comparison of estimated f(x2) vs TRUE f(x2) #######
shift_13 = shift_23  = shift_14  = shift_34 = shift_24 = xs = seq(-4.5,4.5,length.out=41)
idx0 = which(xs == 0) #Index of 0 xs needs to be odd
for (i in 1:length(xs)){
  #i = 35
  x = xs[i]
  # Varying x1
  X = tf$constant(c(x, 0.5, -3, 1), shape=c(1L,4L)) 
  shift_13[i] =   param_model(X)[1,3,2]$numpy() #2=LS Term X1->X3
  shift_14[i] =   param_model(X)[1,4,2]$numpy() #2=LS Term X1->X4
  
  
  #Varying x2
  X = tf$constant(c(0.5, x, 3, 1), shape=c(1L,4L)) 
  shift_23[i] = param_model(X)[1,3,2]$numpy() #2-LS Term X2-->X3 (Mrs. Whites' Notation)
  shift_24[i] = param_model(X)[1,4,2]$numpy() #2=LS Term X2->X4
  
  #Varying x3
  X = tf$constant(c(0.5, 0.5, x, 1), shape=c(1L,4L))
  shift_34[i] = param_model(X)[1,4,2]$numpy() #2-LS Term X3-->X4
}
par(mfrow=c(1,5))
plot(xs, shift_13, type='l', col='red', lwd=2, xlab='x1', ylab='f(x1)', main='b13 * x1 (linear)')
plot(xs, shift_14, type='l', col='red', lwd=2, xlab='x1', ylab='f(x1)', main='b14 * x1 (linear)')
plot(xs, shift_23, type='l', col='red', lwd=2, xlab='x2', ylab='f(x2)', main='b23 * x2 (linear)')
plot(xs, shift_34, type='l', col='red', lwd=2, xlab='x3', ylab='f(x3)', main='b34 * x3 (linear)')
plot(xs, shift_24, type='l', col='red', lwd=2, xlab='x2', ylab='f(x2)', main='b24 * x2 (linear)')

par(mfrow = c(1,1))

######### Learned Transformation of f(x2) ########
if (FALSE){
  if (MA[1,2] == 'cs' && F12 == 1){
    # Assuming xs, cs_12, and idx0 are predefined vectors
    # Create a data frame for the ggplot
    df <- data.frame(x1 = xs, cs_12 = cs_12)
    # Create the ggplot
    p <- ggplot(df, aes(x = x1, y = cs_12)) +
      geom_line(aes(color = "Complex Shift Estimate"), size = 1) +  
      geom_point(aes(color = "Complex Shift Estimate"), size = 1) + 
      # geom_abline(aes(color = "f"), intercept = cs_12[idx0]), slope = 0.3, size = 1) +  # Black solid line for 'DGP'
      scale_color_manual(
        values = c("Complex Shift Estimate" = "blue", "f" = "black"),  # Set colors
        labels = c("Complex Shift Estimate", "f(x)")  # Custom legend labels with expression for f(X_1)
      ) +
      labs(
        x = expression(x[1]),  # Subscript for x_1
        y = paste("~f(x1)"),  # Optionally leave y-axis label blank
        color = NULL  # Removes the color legend title
      ) +
      theme_minimal() +
      theme(legend.position = "none")  # Correct way to remove the legend
    
    # Display the plot
    p
  } else if (MA[1,2] == 'cs' && F12 != 1){
    # Assuming xs, shift_23, and idx0 are predefined vectors
    # Create a data frame for the ggplot
    df <- data.frame(x1 = xs, 
                     shift_12 = cs_12 + ( -cs_12[idx0] - f(0)),
                     f = -f(xs)
    )
    # Create the ggplot
    p <- ggplot(df, aes(x = x1, y = shift_12)) +
      #geom_line(aes(color = "Shift Estimate"), size = 1) +  # Blue line for 'Shift Estimate'
      geom_point(aes(color = "Shift Estimate"), size = 1) +  # Blue points for 'Shift Estimate'
      geom_line(aes(color = "f", y = f), size=0.5) +  # Black solid line for 'DGP'
      scale_color_manual(
        values = c("Shift Estimate" = "blue", "f" = "black"),  # Set colors
        labels = c("Shift Estimate", "f(x1)")  # Custom legend labels with expression for f(X_2)
      ) +
      labs(
        x = expression(x[1]),  # Subscript for x_2
        y = "~f(x1)",  # Optionally leave y-axis label blank
        color = NULL  # Removes the color legend title
      ) +
      theme_minimal() +
      theme(legend.position = "none")  # Correct way to remove the legend
    
    # Display the plot
    p
  } else{
    print(paste0("Unknown Model ", MA[2,3]))
  }
  
  file_name <- paste0(fn, "_f23_est.pdf")
  # Save the plot
  ggsave(file_name, plot = p, width = 8, height = 8)
  file_path <- file.path("~/Library/CloudStorage/Dropbox/Apps/Overleaf/tramdag/figures", basename(file_name))
  ggsave(file_path, plot = p, width = 1.6*6/3, height = 6/3)
}



par(mfrow=c(2,2))
plot(xs, shift_12, main='LS-Term (black DGP, red Ours)', 
     sub = 'Effect of x1 on x2',
     xlab='x1', col='red')
abline(0, 2)

delta_0 = shift1[idx0] - 0
plot(xs, shift1 - delta_0, main='LS-Term (black DGP, red Ours)', 
     sub = paste0('Effect of x1 on x3, delta_0 ', round(delta_0,2)),
     xlab='x1', col='red')
abline(0, -.2)


if (F32 == 1){ #Linear DGP
  if (MA[2,3] == 'ls'){
    delta_0 = shift_23[idx0] - f(0)
    plot(xs, shift_23 - delta_0, main='LS-Term (black DGP, red Ours)', 
         sub = paste0('Effect of x2 on x3, delta_0 ', round(delta_0,2)),
         xlab='x2', col='red')
    #abline(shift_23[length(shift_23)/2], -0.3)
    abline(0, 0.3)
  } 
  if (MA[2,3] == 'cs'){
    plot(xs, cs_23, main='CS-Term (black DGP, red Ours)', xlab='x2',  
         sub = 'Effect of x2 on x3',col='red')
    
    abline(cs_23[idx0], 0.3)  
  }
} else{ #Non-Linear DGP
  if (MA[2,3] == 'ls'){
    delta_0 = shift_23[idx0] + f(0)
    plot(xs, shift_23 - delta_0, main='LS-Term (black DGP, red Ours)', 
         sub = paste0('Effect of x2 on x3, delta_0 ', round(delta_0,2)),
         xlab='x2', col='red')
    lines(xs, -f(xs))
  } else if (MA[2,3] == 'cs'){
    plot(xs, cs_23 + ( -cs_23[idx0] - f(0) ),
         ylab='CS',
         main='CS-Term (black DGP f2(x), red Ours)', xlab='x2',  
         sub = 'Effect of x2 on x3',col='red')
    lines(xs, -f(xs))
  } else{
    print(paste0("Unknown Model ", MA[2,3]))
  }
}
#plot(xs,f(xs), xlab='x2', main='DGP')
par(mfrow=c(1,1))




if (TRUE){
  ####### Compplete transformation Function #######
  ### Copied from structured DAG Loss
  t_i = train$df_orig
  k_min <- k_constant(global_min)
  k_max <- k_constant(global_max)
  
  # from the last dimension of h_params the first entriy is h_cs1
  # the second to |X|+1 are the LS
  # the 2+|X|+1 to the end is H_I
  h_cs <- h_params[,,1, drop = FALSE]
  h_ls <- h_params[,,2, drop = FALSE]
  #LS
  h_LS = tf$squeeze(h_ls, axis=-1L)#tf$einsum('bx,bxx->bx', t_i, beta)
  #CS
  h_CS = tf$squeeze(h_cs, axis=-1L)
  
  theta_tilde <- h_params[,,3:dim(h_params)[3], drop = FALSE]
  theta = to_theta3(theta_tilde)
  cont_dims = which(data_type == 'c') #1 2
  cont_ord = which(data_type == 'o') #3
  
  ### Continiuous dimensions
  #### At least one continuous dimension exits
  h_I = h_dag_extra(t_i[,cont_dims, drop=FALSE], theta[,cont_dims,1:len_theta,drop=FALSE], k_min[cont_dims], k_max[cont_dims]) 
  
  h = h_I + h_LS[,cont_dims, drop=FALSE] + h_CS[,cont_dims, drop=FALSE]
  
  ####### DGP Transformations #######
  X_1 = t_i[,1]$numpy()
  X_2 = t_i[,2]$numpy()
  X_3 = t_i[,3]$numpy()
  
  #h2 = x_2_dash = 5 * x_2 + 2 * X_1
  h2_DGP = 5 *X_2 + 2 * X_1
  h2_DGP_LS = 2 * X_1
  h2_DGP_CS = rep(0, length(X_2))
  h2_DGP_I = 5 * X_2
  
  #h(x3|x1,x2) = 0.63*x3 - 0.2*x1 - f(x2)
  h3_DGP = 0.63*X_3 - 0.2*X_1 - f(X_2)
  h3_DGP_LS = -0.2*X_1
  h3_DGP_CS = -f(X_2)
  h3_DGP_I = 0.63*X_3
  
  
  par(mfrow=c(2,2))
  plot(h2_DGP, h[,2]$numpy(), main='h2')
  abline(0,1,col='red')
  confint(lm(h[,2]$numpy() ~ h2_DGP))
  
  #Same for Intercept
  plot(h2_DGP_I, h_I[,2]$numpy(), main='h2_I')
  abline(0,1,col='red')
  confint(lm(h_I[,2]$numpy() ~ h2_DGP_I))
  
  plot(h2_DGP_LS, h_LS[,2]$numpy(), main='h2_LS')
  abline(0,1,col='red')
  confint(lm(h_LS[,2]$numpy() ~ h2_DGP_LS))
  
  #Same for CS
  plot(h2_DGP_CS, h_CS[,2]$numpy(), main='h2_CS')
  abline(0,1,col='red')
  confint(lm(h_CS[,2]$numpy() ~ h2_DGP_CS))
  
  par(mfrow=c(1,1))
  
  
  par(mfrow=c(2,2))
  
  plot(h3_DGP, h[,3]$numpy(), main='h3')
  abline(0,1,col='red')
  confint(lm(h[,3]$numpy() ~ h3_DGP))
  
  plot(h3_DGP_I, h_I[,3]$numpy(), main='h3_I')
  abline(0,1,col='red')
  confint(lm(h_I[,3]$numpy() ~ h3_DGP_I))
  
  #same for ls  
  plot(h3_DGP_LS, h_LS[,3]$numpy(), main='h3_LS')
  abline(0,1,col='red')
  confint(lm(h_LS[,3]$numpy() ~ h3_DGP_LS))
  
  #same for CS
  plot(h3_DGP_CS, h_CS[,3]$numpy(), main='h3_CS')
  abline(0,1,col='red')
  confint(lm(h_CS[,3]$numpy() ~ h3_DGP_CS))
  
  par(mfrow=c(1,1))
  
}




#==============================================================================
# Use fitted model for prediction
#==============================================================================


