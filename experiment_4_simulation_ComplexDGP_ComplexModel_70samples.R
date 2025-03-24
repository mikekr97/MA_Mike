
# Experiment 4: Simulation (with 70 observations complex shift)

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
DIR = 'runs/experiment_4_simulation_ComplexDGP_ComplexModel_70samples/run'
if (!dir.exists(DIR)) {
  dir.create(DIR, recursive = TRUE)
}
# Copy this file to the directory DIR
file.copy('MA_Mike/experiment_4_simulation_ComplexDGP_ComplexModel_70samples.R', file.path(DIR, 'experiment_4_simulation_ComplexDGP_ComplexModel_70samples.R'), overwrite=TRUE)


#==============================================================================
# Define DGP and Model Structure
#==============================================================================


# Select desired DGA and type of TRAM-DAG shift

# first entry is the type of DGP (1=simple, 4= complex)
# second entry is the type of TRAM-DAG shift (ls=linear, cs=complex)

# args <- c(1, 'ls') # 
# args <- c(1, 'cs') # 
args <- c(5, 'cs') # 



# example of a connection
F42 <- as.numeric(args[1]) 
M42 <- args[2] 
print(paste("F42:", F42, "M42:", M42))

num_epochs <- 200  # 500
len_theta = 20 # Number of coefficients of the Bernstein polynomials
hidden_features_I = c(2,25,25,2)    #hidden_features_CS=hidden_features_I = c(2,25,25,2)
hidden_features_CS = c(2,6,2)


SEED = 42 #-1 #If seed > 0 then the seed is set

if (F42 == 1){
  FUN_NAME = 'DPGLinear'
  f <- function(x) -0.3 * x
} else if (F42 == 2){
  f = function(x) 2 * x**3 + x
  FUN_NAME = 'DPG2x3+x'
} else if (F42 == 3){
  f = function(x) 0.5*exp(x)
  FUN_NAME = 'DPG0.5exp'
} else if (F42 == 4){
  f = function(x) -1.7*atan(5*(x-0.32))
  FUN_NAME = 'DPGatan'
} else if (F42 == 5){
  f = function(x) -2*sin(3*x)+x
  FUN_NAME = 'DPGSin'
} else {
  stop("Unknown Function F32")
}

# function name for DGP
FUN_NAME
f

# plot f(x) for seq(-1,1)
x = seq(-1,1,0.01)
plot(x, f(x), type='l', col='blue', lwd=2, main=FUN_NAME)



# order: x1, x2, x3, x4
#       NP, Ural, BK, SPV
MA =  matrix(c(
  0, 0, 'ls', 'ls',
  0, 0, 'ls', 'cs',
  0, 0,  0  , 'ls',
  0, 0,  0  ,   0), nrow = 4, ncol = 4, byrow = TRUE)
MODEL_NAME = 'ModelCS'

# adjacency matrix
MA



# model name
MODEL_NAME

if (SEED < 0){
  fn = file.path(DIR, paste0('triangle_mixed_',  '_', MODEL_NAME))
} else{
  fn = file.path(DIR, paste0('triangle_mixed_',  '_', MODEL_NAME, '_SEED', SEED))
}
print(paste0("Starting experiment ", fn))



#===============================================================================
# Create train and test data
#===============================================================================


set.seed(123)


# create the data object

dgp <- function(n_obs, doX=c(NA, NA, NA, NA), seed=-1, data = NULL) {
  
  # n_obs <- 1000
  # doX <- c(NA, NA, NA, NA)
  # seed=-1
  
  if (seed > 0) {
    set.seed(seed)
    print(paste0("Setting Seed:", seed))
  }
  
  
  # X1 represents NP, no parents
  if (is.na(doX[1])){
    X_1_A = rnorm(n_obs, -1, 0.3)
    X_1_B = rnorm(n_obs, 1, 0.8)
    X_1 = ifelse(sample(1:2, replace = TRUE, size = n_obs, prob = c(0.2, 0.8)) == 1, X_1_A, X_1_B)
  } else{
    X_1 = rep(doX[1], n_obs)
  }
  #hist(X_1)
  
  # X2 represents Ural, no parents
  if (is.na(doX[2])){
    X_2_A = rnorm(n_obs, -0.25, 1.1)
    X_2_B = rnorm(n_obs, 2, 0.6)
    X_2 = ifelse(sample(1:2, replace = TRUE, size = n_obs, prob = c(0.8, 0.2)) == 1, X_2_A, X_2_B)
  } else{
    X_2 = rep(doX[2], n_obs)
  }
  #hist(X_2)
  
  
  # X3 represents BK, parents NP (X1) and Ural (X2)
  
  if (is.na(doX[3])){
    
    # Sampling according to colr: 
    # x_3_dash = h_0(x_3) + beta13 * X_1 + beta23 * X_2
    # x_3 = h_0^(-1)(x_3_dash - beta13 * X_1 - beta23 * X_2)
    
    # we define h_0(x3) = 3.2 * x3 + 0.3
    # therefore h_0^(-1)(z) = (z - 0.3) / 3.2  
    
    
    # sample n_obs values from a logistic distribution (latent space)
    U3 = runif(n_obs)
    x_3_dash = qlogis(U3) 
    # hist(x_3_dash)
    
    
    
    # shift (on the log-odds scale)
    # h3 = x_3_dash = h_0(x_3) + beta1 * X_13 + beta23 * X_2
    # h3 = x_3_dash = (3.2 * x_3 + 0.3) + beta13 * X_1 + beta23 * X_2
    # X_3 = h_0^(-1)(x_3_dash - beta13 * X_1 - beta23 * X_2)
    # X_3 = 1/3.2 * (x_3_dash - beta13 * X_1 - beta23 * X_2    - 0.3)
    
    beta13 = 0.2
    beta23 = 0.35
    
    X_3 = 1/3.2 * (x_3_dash - beta13 * X_1 - beta23 * X_2    - 0.3)
    
  } else{
    X_3 = rep(doX[3], n_obs)
  }
  # hist(X_3)
  
  
  
  # X4 represents SPV, parents NP (X1) and Ural (X2) and BK (X3)
  
  if (is.na(doX[4])){
    
    # Sampling according to colr: 
    # x_4_dash = h_0(x_4) + beta14 * X_1 + f(X_2) + beta34 * X_3
    # x_4 = h_0^(-1)(x_4_dash - beta14 * X_1 - f(X_2) - beta34 * X_3)
    
    # we define h_0(x4) = 5 * x4
    # therefore h_0^(-1)(z) = (z) / 5  
    
    
    # sample n_obs values from a logistic distribution (latent space)
    U4 = runif(n_obs)
    x_4_dash = qlogis(U4) 
    
    
    # shift (on the log-odds scale)
    # h4 = x_4_dash = h_0(x_4) + beta14 * X_1 + f(X_2) + beta34 * X_3
    # X_4 = 1/5 * (x_4_dash - beta14 * X_1 - f(X_2)- beta34 * X_3)
    
    beta14 = -0.3
    beta34 = -0.7
    
    X_4 = 1/5 * (x_4_dash - beta14 * X_1 - f(X_2)- beta34 * X_3)
    
  } else{
    X_4 = rep(doX[4], n_obs)
  }
  # hist(X_4)
  
  # par(mfrow=c(1,4))
  # hist(X_1)
  # hist(X_2)
  # hist(X_3)
  # hist(X_4)
  
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
train <- dgp(63, seed=ifelse(SEED > 0, SEED, -1))
#test <- dgp(10000, seed=ifelse(SEED > 0, SEED, -1))
test <- dgp(7, seed=41)

# # plot the data for train
par(mfrow=c(1,4))
hist(train$df_R$x1)
hist(train$df_R$x2)
hist(train$df_R$x3)
hist(train$df_R$x4)

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

# loss before training  # 2.75
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
# loss before training after beta initialization
struct_dag_loss(t_i=train$df_orig, h_params=h_params)


#==============================================================================


param_model$compile(optimizer, loss=struct_dag_loss)
param_model$evaluate(x = train$df_orig, y=train$df_orig, batch_size = 7L)




#==============================================================================
# Train the TRAM-DAG model
#==============================================================================
num_epochs <-20003

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

tram_dag_betas <- ws[nrow(ws),]

# fit.13 = Colr(x3~x1, data = train$df_R, order = len_theta)
# fit.23 = Colr(x3~x2, data = train$df_R, order = len_theta)
fit.3 = Colr(x3~x1 + x2, data = train$df_R, order = len_theta)
fit.4 = Colr(x4~x1 + x2 + x3, data = train$df_R, order = len_theta)

# w13        w14       w23        w24        w34
colr_betas <- c(coef(fit.3)[1], coef(fit.4)[1], coef(fit.3)[2], coef(fit.4)[2], coef(fit.4)[3])

true_betas <- c(0.2, -0.3, 0.35, NA, -0.7)

comparison <- rbind(tram_dag_betas, colr_betas, true_betas)
rownames(comparison) <- c('TRAM-DAG', 'Colr', 'True')
round(comparison,3)

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
if (F42 == 4){
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
  file_path <- file.path("runs/experiment_4_simulation_ComplexDGP_ComplexModel_70samples/run/", basename(file_name))
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
  file_path <- file.path("runs/experiment_4_simulation_ComplexDGP_ComplexModel_70samples/run/", basename(file_name))
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
  dx3 = -3.5
  doX=c(NA, NA, dx3, NA)
  s_do_fitted = do_dag_struct(param_model, train$A, doX=doX)$numpy()
  
  # add the doX to the plot
  df = data.frame(vals=s_obs_fitted[,1], type='Model', X=1, L='L0')
  df = rbind(df, data.frame(vals=s_obs_fitted[,2], type='Model', X=2, L='L0'))
  df = rbind(df, data.frame(vals=s_obs_fitted[,3], type='Model', X=3, L='L0'))
  df = rbind(df, data.frame(vals=s_obs_fitted[,4], type='Model', X=4, L='L0'))
  
  df = rbind(df, data.frame(vals=train$df_R[,1], type='DGP', X=1, L='L0'))
  df = rbind(df, data.frame(vals=train$df_R[,2], type='DGP', X=2, L='L0'))
  df = rbind(df, data.frame(vals=train$df_R[,3], type='DGP', X=3, L='L0'))
  df = rbind(df, data.frame(vals=train$df_R[,4], type='DGP', X=4, L='L0'))
  
  df = rbind(df, data.frame(vals=s_do_fitted[,1], type='Model', X=1, L='L1'))
  df = rbind(df, data.frame(vals=s_do_fitted[,2], type='Model', X=2, L='L1'))
  df = rbind(df, data.frame(vals=s_do_fitted[,3], type='Model', X=3, L='L1'))
  df = rbind(df, data.frame(vals=s_do_fitted[,4], type='Model', X=4, L='L1'))
  
  d = dgp(10000, doX=doX)$df_R
  # d = dgp(nrow(train_df), doX=doX, data=train_df)$df_R  # not possible for real data
  df = rbind(df, data.frame(vals=d[,1], type='DGP', X=1, L='L1'))
  df = rbind(df, data.frame(vals=d[,2], type='DGP', X=2, L='L1'))
  df = rbind(df, data.frame(vals=as.numeric(d[,3]), type='DGP', X=3, L='L1'))
  df = rbind(df, data.frame(vals=d[,4], type='DGP', X=4, L='L1'))
  
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
    file_path <- file.path("runs/experiment_4_simulation_ComplexDGP_ComplexModel_70samples/run/", basename(file_name))
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
     main='h_I(X4) Colr and Our Model', cex.main=0.8)
lines(Xs[,4], h_I[,4], col='red', lty=2, lwd=5)
rug(train$df_orig$numpy()[,4], col='blue')


# xs = seq(-1,1,length.out=41)
# idx0 = which(xs == 0) #Index of 0 xs needs to be odd
# 
# x = xs[idx0]  # x = 0 at index 21
# X = tf$constant(c(x, 0.5, -3, 1), shape=c(1L,4L))  # input c(0, 0.5, -3, 1)
# CS_X2_0 <- param_model(X)[1,4,1]$numpy() #1=CS Term X2->X4 when # x2 = 0
# 
# ##### X4
# fit.4123 = Colr(X4 ~ X1 + X2 + X3,df, order=len_theta)
# temp = model.frame(fit.4123)[1:2, -1, drop=FALSE] #WTF!
# 
# plot(fit.4123, which = 'baseline only', newdata = temp, lwd=2, col='blue', 
#      main='h_I(X4) Colr and Our Model', cex.main=0.8)
# lines(Xs[,4], h_I[,4]+ CS_X2_0, col='red', lty=2, lwd=5)
# rug(train$df_orig$numpy()[,4], col='blue')





##### Checking observational distribution ####
library(car)
s = do_dag_struct(param_model, train$A, doX=c(NA, NA, NA, NA), num_samples = 5000)
par(mfrow=c(1,4))
for (i in 1:4){
  d = s[,i]$numpy()
  hist(train$df_orig$numpy()[,i], freq=FALSE, 100,main=paste0("X",i, " red: ours, black: data"), xlab='samples')
  lines(density(train$df_orig$numpy()[,i]), col='blue', lwd=2)
  #hist(train$df_orig$numpy()[,i], freq=FALSE, 100,main=paste0("X_",i))
  lines(density(s[,i]$numpy()), col='red', lwd=2)
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




###### Comparison of estimated f(x2) vs TRUE f(x2) #######
shift_13 = shift_23  = shift_14  = shift_34 = cs_24 = xs = seq(-4.5,4.5,length.out=41)
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
  cs_24[i] = param_model(X)[1,4,1]$numpy() #1=CS Term X2->X4
  
  #Varying x3
  X = tf$constant(c(0.5, 0.5, x, 1), shape=c(1L,4L))
  shift_34[i] = param_model(X)[1,4,2]$numpy() #2-LS Term X3-->X4
}
par(mfrow=c(1,5))
plot(xs, shift_13, type='l', col='red', lwd=2, xlab='x1', ylab='f(x1)', main='b13 * x1 (linear)')
plot(xs, shift_14, type='l', col='red', lwd=2, xlab='x1', ylab='f(x1)', main='b14 * x1 (linear)')
plot(xs, shift_23, type='l', col='red', lwd=2, xlab='x2', ylab='f(x2)', main='b23 * x2 (linear)')
plot(xs, shift_34, type='l', col='red', lwd=2, xlab='x3', ylab='f(x3)', main='b34 * x3 (linear)')
plot(xs, cs_24, type='l', col='red', lwd=2, xlab='x2', ylab='f(x2)', main='b24(x2) complex')

par(mfrow = c(1,1))

library(mgcv)
gam_model <- gam(x4 ~ s(x1) + s(x2) + s(x3), data = train$df_R)
plot(gam_model)


######### Learned Transformation of f(x2) ########
if (FALSE){
  if (MA[2,4] == 'cs' && F42 == 1){
    # Assuming xs, cs_24, and idx0 are predefined vectors
    # Create a data frame for the ggplot
    df <- data.frame(x2 = xs, cs_24 = cs_24)
    # Create the ggplot
    p <- ggplot(df, aes(x = x2, y = cs_24)) +
      geom_line(aes(color = "Complex Shift Estimate"), size = 1) +  
      geom_point(aes(color = "Complex Shift Estimate"), size = 1) + 
      geom_abline(aes(color = "f"), intercept = cs_24[idx0], slope = -0.3, size = 1) +  # Black solid line for 'DGP'
      scale_color_manual(
        values = c("Complex Shift Estimate" = "blue", "f" = "black"),  # Set colors
        labels = c("Complex Shift Estimate", "f(x)")  # Custom legend labels with expression for f(X_1)
      ) +
      labs(
        x = expression(x[2]),  # Subscript for x_1
        y = paste("~f(x2)"),  # Optionally leave y-axis label blank
        color = NULL  # Removes the color legend title
      ) +
      theme_minimal() + 
      theme(legend.position = "none")  # Correct way to remove the legend 
    
    
    # Display the plot
    p
  } else if (MA[2,4] == 'cs' && F42 != 1){
    # Assuming xs, shift_24, and idx0 are predefined vectors
    # Create a data frame for the ggplot
    df <- data.frame(x2 = xs, 
                     shift_24 = cs_24 + ( -cs_24[idx0] + f(0)), # cs_24 + ( -cs_24[idx0] - f(0))
                     f = f(xs)   # -f(xs)
    )
    # Create the ggplot
    p <- ggplot(df, aes(x = x2, y = shift_24)) +
      geom_line(aes(color = "Shift Estimate"), size = 1) +  # Blue line for 'Shift Estimate'
      geom_point(aes(color = "Shift Estimate"), size = 1) +  # Blue points for 'Shift Estimate'
      geom_line(aes(color = "f", y = f), size=0.5) +  # Black solid line for 'DGP'
      scale_color_manual(
        values = c("Shift Estimate" = "blue", "f" = "black"),  # Set colors
        labels = c("Shift Estimate", "f(x1)")  # Custom legend labels with expression for f(X_2)
      ) +
      labs(
        x = expression(x[2]),  # Subscript for x_2
        y = "~f(x2)",  # Optionally leave y-axis label blank
        color = NULL  # Removes the color legend title
      ) +
      theme_minimal() +
      theme(legend.position = "none")  # Correct way to remove the legend
    
    # Display the plot
    p
  } else{
    print(paste0("Unknown Model ", MA[2,4]))
  }
  
  file_name <- paste0(fn, "_f24_est.pdf")
  # Save the plot
  ggsave(file_name, plot = p, width = 8, height = 8)
  file_path <- file.path("~/Library/CloudStorage/Dropbox/Apps/Overleaf/tramdag/figures", basename(file_name))
  ggsave(file_path, plot = p, width = 1.6*6/3, height = 6/3)
}



par(mfrow=c(2,2))
plot(xs, shift_14, main='LS-Term (black DGP, red Ours)', 
     sub = 'Effect of x1 on x4',
     xlab='x1', col='red')
abline(0, -0.3)

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
    plot(xs, cs_24, main='CS-Term (black DGP, red Ours)', xlab='x2',  
         sub = 'Effect of x2 on x3',col='red')
    
    abline(cs_24[idx0], -0.3)  
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
  
  ### Continuous dimensions
  #### At least one continuous dimension exits
  h_I = h_dag_extra(t_i[,cont_dims, drop=FALSE], theta[,cont_dims,1:len_theta,drop=FALSE], k_min[cont_dims], k_max[cont_dims]) 
  
  h = h_I + h_LS[,cont_dims, drop=FALSE] + h_CS[,cont_dims, drop=FALSE]
  
  ####### DGP Transformations #######
  X_1 = t_i[,1]$numpy()
  X_2 = t_i[,2]$numpy()
  X_3 = t_i[,3]$numpy()
  X_4 = t_i[,4]$numpy()
  
  # #h2 = x_2_dash = 5 * x_2 + 2 * X_1
  # h2_DGP = 5 *X_2 + 2 * X_1
  # h2_DGP_LS = 2 * X_1
  # h2_DGP_CS = rep(0, length(X_2))
  # h2_DGP_I = 5 * X_2
  # 
  # #h(x3|x1,x2) = 0.63*x3 - 0.2*x1 - f(x2)
  # h3_DGP = 0.63*X_3 - 0.2*X_1 - f(X_2)
  # h3_DGP_LS = -0.2*X_1
  # h3_DGP_CS = -f(X_2)
  # h3_DGP_I = 0.63*X_3
  
  # h3 = x_3_dash = (3.2 * x_3 + 0.3) + beta13 * X_1 + beta23 * X_2
  h3_DGP = (3.2 * X_3 + 0.3) + 0.2 * X_1 + 0.35 * X_2
  h3_DGP_LS = 0.2 * X_1 + 0.35 * X_2
  h3_DGP_I = (3.2 * X_3 + 0.3) 
  
  # h4 = x_4_dash = 5*x_4 + beta14 * X_1 + f(X_2) + beta34 * X_3
  h4_DGP = 5*X_4 + (-0.3) * X_1 + f(X_2) + (-0.7) * X_3
  h4_DGP_LS = (-0.3) * X_1 + (-0.7) * X_3
  h4_DGP_CS = f(X_2) 
  h4_DGP_I = 5*X_4
  
  par(mfrow=c(2,2))
  plot(h3_DGP, h[,3]$numpy(), main='h3')
  abline(0,1,col='red')
  confint(lm(h[,3]$numpy() ~ h3_DGP))
  
  #Same for Intercept
  plot(h3_DGP_I, h_I[,3]$numpy(), main='3_I')
  abline(0,1,col='red')
  confint(lm(h_I[,3]$numpy() ~ h3_DGP_I))
  
  plot(h3_DGP_LS, h_LS[,3]$numpy(), main='h3_LS')
  abline(0,1,col='red')
  confint(lm(h_LS[,3]$numpy() ~ h3_DGP_LS))
  
  #Same for CS
  plot(h_DGP_CS, h_CS[,2]$numpy(), main='h2_CS')
  abline(0,1,col='red')
  confint(lm(h_CS[,2]$numpy() ~ h2_DGP_CS))
  
  par(mfrow=c(1,1))
  
  
  par(mfrow=c(2,2))
  
  plot(h4_DGP, h[,4]$numpy(), main='h4')
  abline(0,1,col='red')
  confint(lm(h[,4]$numpy() ~ h4_DGP))
  
  plot(h4_DGP_I, h_I[,4]$numpy(), main='h4_I')
  abline(0,1,col='red')
  confint(lm(h_I[,4]$numpy() ~ h4_DGP_I))
  
  #same for ls  
  plot(h4_DGP_LS, h_LS[,4]$numpy(), main='h4_LS')
  abline(0,1,col='red')
  confint(lm(h_LS[,4]$numpy() ~ h4_DGP_LS))
  
  #same for CS
  plot(h4_DGP_CS, h_CS[,4]$numpy(), main='h4_CS')
  abline(cs_24[idx0],1,col='red')
  confint(lm(h_CS[,4]$numpy() ~ h4_DGP_CS))
  
  par(mfrow=c(1,1))
  
}



