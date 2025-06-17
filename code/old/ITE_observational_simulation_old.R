##### When starting a new R Session ####
if (FALSE){
  reticulate::use_python("C:/ProgramData/Anaconda3/python.exe", required = TRUE)
}




#### A mixture of discrete and continuous variables ####
library(tensorflow)
library(keras)
library(mlt)
library(tram)
library(MASS)
library(tidyverse)
library(dplyr)


#### For ITE
source('code/utils/ITE_utils.R')

#### For TF
source('code/utils/utils_tf.R')

#### For TFP
library(tfprobability)
source('code/utils/utils_tfp.R')

##### Flavor of experiment ######

#### Saving the current version of the script into runtime
DIR = 'runs/ITE_observational_simulation/run'
if (!dir.exists(DIR)) {
  dir.create(DIR, recursive = TRUE)
}
# Copy this file to the directory DIR
file.copy('/code/ITE_observational_simulation.R', file.path(DIR, 'ITE_observational_simulation.R'), overwrite=TRUE)



# X1, X2, X3, X4 (Tr), X5, X6, X7 (Y)
MA =  matrix(c(
  0,   0,  0, 'ci', 0, 0, 'ci',    ## X1 impacts X4 (Tr) and X7 (Y) directly
  0,   0,  0, 'ci', 0, 0, 'ci',    ## X2 impacts X4 (Tr) and X7 (Y) directly
  0,   0,  0,  0  , 0, 0, 'ci',    ## X3 impacts X7 (Y) directly
  0,   0,  0,  0, 'ci',0, 'ci',    ## X4 (Tr) impacts X5 and X7 (Y) directly
  0,   0,  0,  0,  0,'ci','ci',    ## X5 impacts X6 and X7 (Y) directly
  0,   0,  0,  0,  0,  0, 'ci',    ## X6  X7 (Y) directly
  0,   0,  0,  0,  0,  0, 0),    ## X7 (Y) terminal node
  nrow = 7, ncol = 7, byrow = TRUE)


MODEL_NAME = 'ModelCI'

fn = file.path(DIR, paste0(MODEL_NAME))
print(paste0("Starting experiment ", fn))






###############################################################################
# DGP for ITE in observational setting
###############################################################################



##### DGP ########
dgp_simulation <- function(n_obs=20000, 
                           SEED=123, 
                           rho = 0.1, 
                           doX=c(NA, NA, NA, NA, NA, NA, NA)) {
  #n_obs = 1e5 n_obs = 10
  set.seed(SEED)
  
  # Data simulation
  
  # Define sample size
  n <- n_obs
  
  
  
  ### Independent continuous source nodes (X1, X2, X3)
  
  # Define the mean vector (all zeros for simplicity)
  mu <- rep(0, 3)  
  
  # Define the covariance matrix (compound symmetric for simplicity)
  rho <- rho  # Correlation coefficient
  Sigma <- matrix(rho, nrow = 3, ncol = 3)  # Start with all elements as rho
  diag(Sigma) <- 1  # Set diagonal elements to 1 (variances)
  
  # Generate n samples from the multivariate normal distribution
  data <- MASS::mvrnorm(n, mu = mu, Sigma = Sigma)
  colnames(data) <- paste0("X", 1:3)
  
  # doX=c(NA, NA, NA, NA, NA, NA, NA)
  if (!is.na(doX[1])){
    data[,1] <- doX[1]  # Set X1 to a fixed value if specified
  }
  if (!is.na(doX[2])) {
    data[,2] <- doX[2]  # Set X2 to a fixed value if specified
  }
  if (!is.na(doX[3])) {
    data[,3] <- doX[3]  # Set X3 to a fixed value if specified
  }
  
  
  
  ### Binary treatment X4 (Tr) (depends on X1 and X2)
  
  if (is.na(doX[4])) {
    
    beta_0 <- 0.5
    beta_X <- c(-0.5, 0.3)  # main effects of X1 and X2
    
    logit_Tr <- beta_0 + data[,1:2] %*% beta_X
    
    # Convert logit to probability of outcome
    prob_Tr <- plogis(logit_Tr)
    
    # Generate binary outcome Y based on the probability
    Tr <- rbinom(n, size = 1, prob = prob_Tr)
  } else {
    Tr <- rep(doX[4], n_obs)
  }
  # mean(Tr)
  
  
  
  ### X5 dependent on Tr
  # Sampling according to colr
  if (is.na(doX[5])){
    U5 = runif(n_obs)
    x_5_dash = qlogis(U5)
    
    beta15 <- 0.8
    #x_5_dash = h_0(x_5) + beta15 * Tr
    #x_5_dash = 2.5*x_5 + beta15 * Tr
    X5 =  1/2.5 * (x_5_dash - beta15 * Tr)
    
    # under Tr = 0
    X5_ct <- 1/2.5 * (x_5_dash - beta15 * 0)
    
    # under Tr = 1
    X5_tx <- 1/2.5 * (x_5_dash - beta15 * 1)
    
    
  } else{
    X5 = rep(doX[5], n_obs)
  }
  
  
  ### X6 dependent on X5
  # Sampling according to colr
  if (is.na(doX[6])){
    U6 = runif(n_obs)
    x_6_dash = qlogis(U6)
    
    beta56 <- (-0.5)
    #x_6_dash = h_0(x_6) + beta56 * X5
    #x_6_dash = 0.7*x_6 + beta56 * X5
    X6 = 1/0.7 * (x_6_dash - beta56 * X5)
    
    # under Tr = 0
    X6_ct <- 1/0.7 * (x_6_dash - beta56 * X5_ct)
    
    # under Tr = 1
    X6_tx <- 1/0.7 * (x_6_dash - beta56 * X5_tx)
    
    
  } else{
    X6 = rep(doX[6], n_obs)
  }
  
  
  
  #### Y (X7): continuous outcome dependent on Tr, X2-X6 (with Tr interactions X2, X3)
  
  data_X <- as.matrix(data.frame(X1 = data[,1], X2 = data[,2], X3 = data[,3], X5 = X5, X6 = X6))
  
  
  if (is.na(doX[7])){
    U7 = runif(n_obs)
    x_7_dash = qlogis(U7)
    
    beta0 <- 0.45
    beta_t <- 1.5
    beta_X <- c(-0.5, 0.1, 0.2, -0.6, 0.3)  # main effects of X1, X2, X3   , X5, X6
    beta_TX <- c(-0.9, 0.7)  # interaction effects of X2, X3 with treatment
    
    logit_X7 <- beta0 + beta_t * Tr + data_X %*% beta_X + (data_X[,c("X2", "X3")] %*% beta_TX) * Tr
    
    #x_7_dash = h_0(x_7) + logit_X7
    #x_7_dash = 4.5*x_7 + logit_X7
    X7 = 1/4.5 * (x_7_dash - logit_X7)
    
    # under Tr = 0
    data_X_ct <- as.matrix(data.frame(X1 = data[,1], X2 = data[,2], X3 = data[,3], X5 = X5_ct, X6 = X6_ct))
    logit_X7_ct <- beta0 + data_X_ct %*% beta_X
    X7_ct <- 1/4.5 * (x_7_dash - logit_X7_ct)
    
    # under Tr = 1
    data_X_tx <- as.matrix(data.frame(X1 = data[,1], X2 = data[,2], X3 = data[,3], X5 = X5_tx, X6 = X6_tx))
    logit_X7_tx <- beta0 + beta_t + data_X_tx %*% beta_X + (data_X_tx[,c("X2", "X3")] %*% beta_TX)
    X7_tx <- 1/4.5 * (x_7_dash - logit_X7_tx)
    
    
    ### ITE with median of potential outcomes (median X_7_dash is 0)
    ITE_median <- (1/4.5 * (0 - logit_X7_tx)) - 
      (1/4.5 * (0 - logit_X7_ct))
    
    
  } else{
    X7 = rep(doX[7], n_obs)
  }
  
  
  # Calculate the individual treatment effect (with the effective potential outcome)
  # however, in my example I assume that Y was not yet observed. Better compare estimate
  # to ITE_median that was specified above
  
  ITE_true <- X7_tx - X7_ct
  
  
  
  ### Note: ITE_true (with observed Z_i) and ITE_median (with Z_i = 0) are equal
  ### in this simulation with additive shift (incl. interaction) because the
  ### latent value cancels out when calculating the ITE:
  
  # Y(0) = 1/4.5 * (Z - logit_X7_ct)
  # Y(1) = 1/4.5 * (Z - logit_X7_tx)
  # ITE = Y(1) - Y(0) = 1/4.5 * (Z - logit_X7_tx) - 1/4.5 * (Z - logit_X7_ct)
  # ITE = 1/4.5 * (Z - logit_X7_tx - Z + logit_X7_ct)
  # ITE = 1/4.5 * (logit_X7_tx + logit_X7_ct)
  
  ### Therefore in this setting the latent variable observed for the outcome does 
  ### not matter, but in other settings with more complex dgp it might be df
  
  # plot(density((1/4.5 * (0 - logit_X7_tx))))
  # lines(density(X7_tx))
  # 
  # plot(density((1/4.5 * (0 - logit_X7_ct))))
  # lines(density(X7_ct))
  # 
  # 
  # plot(density(ITE_true), main = "ITE True vs ITE Median", lwd=2)
  # lines(density(ITE_median), col = 'red', lty = 3)
  


  # Combine all variables into a single data frame (observed)
  simulated_full_data <- data.frame(X1 = data_X[,1],
                                    X2 = data_X[,2],
                                    X3 = data_X[,3],
                                    Tr=Tr, 
                                    X5 = X5,
                                    X6 = X6,
                                    Y=X7, 
                                    ITE_true = ITE_true, 
                                    ITE_median = ITE_median)
  
  # Data for testing ITE models
  # simulated_data <- data.frame(ID =1:n, Y=X7, Tr=Tr, Treatment=Tr, data_X, ITE_true = ITE_true) %>% 
  #   # add Treatment variable Tr=Treatment
  #   mutate(Treatment = ifelse(Treatment==1,"Y", "N")) %>% 
  #   mutate(Treatment = factor(Treatment, levels = c("N", "Y")))
  
  
  
  # Combine all variables into a single data frame (Under Tr = 0)
  # simulated_full_data_ct <- data.frame(ID = 1:n, Y=X7_ct, Tr=0, data_X_ct) %>% 
  #   mutate(Treatment = "N") %>% 
  #   mutate(Treatment = factor(Treatment, levels = c("N", "Y")))
  # 
  # # Data for testing ITE models (Under Tr = 0)
  # simulated_data_ct <- data.frame(ID =1:n, Y=X7_ct, Tr=0, Treatment=0, data_X_ct) %>% 
  #   # add Treatment variable Tr=Treatment
  #   mutate(Treatment = ifelse(Treatment==1,"Y", "N")) %>% 
  #   mutate(Treatment = factor(Treatment, levels = c("N", "Y")))
  
  
  
  # # Combine all variables into a single data frame (Under Tr = 1)
  # simulated_full_data_tx <- data.frame(ID = 1:n, Y=X7_tx, Tr=1, data_X_tx) %>% 
  #   mutate(Treatment = "Y") %>% 
  #   mutate(Treatment = factor(Treatment, levels = c("N", "Y")))
  # 
  # # Data for testing ITE models (Under Tr = 1)
  # simulated_data_tx <- data.frame(ID =1:n, Y=X7_tx, Tr=1, Treatment=1, data_X_tx) %>% 
  #   # add Treatment variable Tr=Treatment
  #   mutate(Treatment = ifelse(Treatment==1,"Y", "N")) %>% 
  #   mutate(Treatment = factor(Treatment, levels = c("N", "Y")))
  # 
  # 
  

  # set.seed(12345)  
  # val_idx <- sample(1:n, size = floor(0.5 * n))  
  # train_idx <- setdiff(1:n, val_idx)
  # 
  # data.dev <- simulated_full_data[train_idx, ]
  # data.val <- simulated_full_data[val_idx, ]
  
  
  # convert to tensorflow objects
  dat_temp <- as.matrix(simulated_full_data[, 1:7]) # without ITE_true
  # dat_temp[,4] <- dat_temp[,4] + 1  # binary treatment ordinal coded
  dat.tf = tf$constant(dat_temp, dtype = 'float32')


  

  q1 = quantile(simulated_full_data[,1], probs = c(0.05, 0.95))
  q2 = quantile(simulated_full_data[,2], probs = c(0.05, 0.95))
  q3 = quantile(simulated_full_data[,3], probs = c(0.05, 0.95))
  q4 = c(0, 1) #No Quantiles for ordinal data
  q5 = quantile(simulated_full_data[,5], probs = c(0.05, 0.95))
  q6 = quantile(simulated_full_data[,6], probs = c(0.05, 0.95))
  q7 = quantile(simulated_full_data[,7], probs = c(0.05, 0.95))
# 
#   # train
#   dat_temp <- as.matrix(data.dev[, 1:7]) # without ITE_true
#   dat_temp[,4] <- dat_temp[,4] + 1  # binary treatment ordinal coded
#   dat.train.tf = tf$constant(dat_temp, dtype = 'float32')
#   
#   
#   # test
#   dat_temp <- as.matrix(data.val[, 1:7]) # without ITE_true
#   dat_temp[,4] <- dat_temp[,4] + 1  # binary treatment ordinal coded
#   dat.test.tf = tf$constant(dat_temp, dtype = 'float32')
#   
#   
#   
#   q1 = quantile(data.dev[,1], probs = c(0.05, 0.95))
#   q2 = quantile(data.dev[,2], probs = c(0.05, 0.95))
#   q3 = quantile(data.dev[,3], probs = c(0.05, 0.95))
#   q4 = c(1, 2) #No Quantiles for ordinal data
#   q5 = quantile(data.dev[,5], probs = c(0.05, 0.95))
#   q6 = quantile(data.dev[,6], probs = c(0.05, 0.95))
#   q7 = quantile(data.dev[,7], probs = c(0.05, 0.95))
# 
#   
  
  # X1, X2, X3, X4 (Tr), X5, X6, X7 (Y)
  A =  matrix(c(
    0,   0,  0,  1,  0,  0,  1,    ## X1 impacts X4 (Tr) and X7 (Y) directly
    0,   0,  0,  1,  0,  0,  1,    ## X2 impacts X4 (Tr) and X7 (Y) directly
    0,   0,  0,  0 , 0,  0,  1,    ## X3 impacts X7 (Y) directly
    0,   0,  0,  0,  1,  0,  1,    ## X4 (Tr) impacts X5 and X7 (Y) directly
    0,   0,  0,  0,  0,  1,  1,    ## X5 impacts X6 and X7 (Y) directly
    0,   0,  0,  0,  0,  0,  1,    ## X6  X7 (Y) directly
    0,   0,  0,  0,  0,  0,  0),    ## X7 (Y) terminal node
    nrow = 7, ncol = 7, byrow = TRUE)

  return(list(
    dat.tf=dat.tf, 
    simulated_full_data = simulated_full_data,
    min =  tf$constant(c(q1[1], q2[1], q3[1], q4[1], q5[1], q6[1], q7[1]), dtype = 'float32'),
    max = tf$constant(c(q1[2], q2[2], q3[2], q4[2], q5[2], q6[2], q7[2]), dtype = 'float32'),
    type = c('c', 'c', 'c', 'o', 'c', 'c', 'c'),
    A=A
  ))
} 


train <- dgp_simulation(n_obs = 10000, SEED = 123, rho = 0.1, doX = c(NA, NA, NA, NA, NA, NA, NA))
global_min = train$min
global_max = train$max
data_type = train$type

## binary treatment is 0,1 encoded, but in the loss treated as ordinal variable 1,2


############################
# Check simulated Train Data
############################


# | Variable | Meaning                                        | Type       | Role       |
#   | -------- | ---------------------------------------------- | ---------- | ---------- |
#   | T        | Weight loss drug participation                 | Binary     | Treatment  |
#   | X₁       | Age                                            | Continuous | Confounder |
#   | X₂       | Baseline BMI                                   | Continuous | Confounder |
#   | X₃       |Physical activity level      (e.g., hours/week) | Continuous | Covariate  |
#   | X₅       | Body fat percentage                            | Continuous | Mediator   |
#   | X₆       | Insulin resistance (e.g., HOMA-IR)             | Continuous | Mediator   |
#   | Y        | Cardiovascular risk score                      | Continuous | Outcome    |
#   


par(mfrow = c(3,3))
hist(train$simulated_full_data$X1, main = "X1")
hist(train$simulated_full_data$X2, main = "X2")
hist(train$simulated_full_data$X3, main = "X3")
hist(train$simulated_full_data$X5, main = "X5")
hist(train$simulated_full_data$X6, main = "X6")
hist(train$simulated_full_data$Y, main = "Y (X7)")
barplot(table(train$simulated_full_data$Tr), main = "Tr (X4)")
boxplot(train$simulated_full_data$Y ~ train$simulated_full_data$Tr, main = "Y (X7) by Tr")



hist(train$simulated_full_data$ITE_true, main = "ITE True", xlab = "ITE", breaks = 50)


############################
# Fit TRAM-DAG on training data
############################



len_theta_max = 20 # max number for intercept (ordinal)
len_theta = 20 # number of coefficients of the Bernstein polynomials

# Attention, number of nodes in each layer shouldbe grater than the number of 
# variable that have a complex influence (CI or CS) on other variables, because
# with masking, some connections are "cut", so for 7 influencing variables, we 
# need at least 7 nodes in the first layer, to ensure there are enough 
# connections available. --> this is not an issue anymore, if the NN's were made 
# separately for each node, compared to this masking approach here
hidden_features_I = c(10, 10, 10) 
hidden_features_CS = c(2, 5, 5, 2)


# t_i <- train$dat.tf
param_model = create_param_model(MA, hidden_features_I=hidden_features_I, len_theta=len_theta, hidden_features_CS=hidden_features_CS,
                                 dropout = TRUE, batchnorm = TRUE, activation = "relu")
optimizer = optimizer_adam(learning_rate = 0.001)
param_model$compile(optimizer, loss=struct_dag_loss_ITE_observational)

h_params <- param_model(train$dat.tf)

param_model$evaluate(x = train$dat.tf, y=train$dat.tf, batch_size = 7L)
summary(param_model)

# show activation function activation_68 --> Relu is used (before it was sigmoid)
param_model$get_layer("activation_761")$get_config()






# generate validation set for early stopping to prevent overfitting
validation <- dgp_simulation(n_obs = 10000, SEED = 3, rho = 0.1, doX = c(NA, NA, NA, NA, NA, NA, NA))



num_epochs <- 1000 

# fnh5 = paste0(fn, '_E', num_epochs, 'early_stopping_CI.h5')   # saved as "best_model.h5" with early stopping
# fnRdata = paste0(fn, '_E', num_epochs, 'early_stopping_CI.RData')   #


fnh5 = paste0(fn, '_E', num_epochs, 'best_model.h5')   # saved as "best_model.h5" with early stopping
fnRdata = paste0(fn, '_E', num_epochs, 'best_model.RData')   #

if (file.exists(fnh5)) {
  param_model$load_weights(fnh5)
  load(fnRdata)
  # (global_min = min)
  # (global_max = min)
} else {
  if (FALSE) { ### Full Training w/o diagnostics
    hist = param_model$fit(x = train$dat.tf, y = train$dat.tf, epochs = 200L, verbose = TRUE,
                           validation_data = list(validation$dat.tf, validation$dat.tf))
    param_model$save_weights(fnh5)
    plot(hist$epoch, hist$history$loss)
    plot(hist$epoch, hist$history$loss, ylim = c(1.07, 1.2))
  } else { ### Training with diagnostics and early stopping
    
    # Early stopping parameters
    patience <- 20
    best_val_loss <- Inf
    epochs_no_improve <- 0
    early_stop_epoch <- NULL
    
    # Initialize loss history
    train_loss <- numeric()
    val_loss <- numeric()
    
    for (e in 1:num_epochs) {
      cat(sprintf("Epoch %d\n", e))
      
      hist <- param_model$fit(
        x = train$dat.tf, y = train$dat.tf,
        epochs = 1L, verbose = TRUE,
        validation_data = list(validation$dat.tf, validation$dat.tf)
      )
      
      # Append current epoch losses
      train_loss <- c(train_loss, hist$history$loss)
      val_loss <- c(val_loss, hist$history$val_loss)
      
      # Early stopping logic
      current_val_loss <- val_loss[length(val_loss)]
      
      if (current_val_loss < best_val_loss - 1e-5) {
        best_val_loss <- current_val_loss
        epochs_no_improve <- 0
        param_model$save_weights("best_model.h5")
      } else {
        epochs_no_improve <- epochs_no_improve + 1
      }
      
      if (epochs_no_improve >= patience) {
        early_stop_epoch <- e
        cat(sprintf("Early stopping triggered at epoch %d\n", e))
        break
      }
    }
    
    # Load best weights
    if (!is.null(early_stop_epoch)) {
      param_model$load_weights("best_model.h5")
    }
    
    # Save final model and training history
    param_model$save_weights(fnh5)
    save(train_loss, val_loss,
         MA, len_theta,
         hidden_features_I,
         hidden_features_CS,
         file = fnRdata)
  }
}



par(mfrow=c(1,1))

epochs = length(train_loss)

# png("C:/Users/kraeh/OneDrive/Dokumente/Desktop/UZH_Biostatistik/Masterarbeit/MA_Mike/presentation_report/intermediate_presentation/img/Loss_Example.png",
#     res = 150)

plot(1:epochs, train_loss, type='l', main='', ylab='Loss', xlab='Epochs')#, ylim = c(1, 1.5)
lines(1:epochs, val_loss, type = 'l', col = 'blue')
legend('topright', legend=c('training', 'validation'), col=c('black', 'blue'), lty=1:1, cex=0.8, bty='n')

# dev.off()

# Last 50
diff = max(epochs - 100,0)
plot(diff:epochs, val_loss[diff:epochs], type = 'l', col = 'green', main='Last 50 epochs')
lines(diff:epochs, train_loss[diff:epochs], type='l')




############################
# Check Observational and Interventional distributions
############################


if (TRUE){
  doX = c(NA, NA, NA, NA, NA, NA, NA)
  s_obs_fitted = do_dag_struct_ITE_observational(param_model, train$A, doX, num_samples = 5000)$numpy()

  ### Do X4 = Control
  dx4_ct = 0
  doX = c(NA, NA, NA, dx4_ct, NA, NA, NA)
  s_do_fitted_ct = do_dag_struct_ITE_observational(param_model, train$A, doX, num_samples = 5000)$numpy()
  
  
  ### Do X4 = Treatment
  
  dx4_tx = 1
  doX = c(NA, NA, NA, dx4_tx, NA, NA, NA)
  s_do_fitted_tx = do_dag_struct_ITE_observational(param_model, train$A, doX, num_samples = 5000)$numpy()
  
  
  # observational
  df = data.frame(vals=s_obs_fitted[,1], type='Model', X=1, L='L0')
  df = rbind(df, data.frame(vals=s_obs_fitted[,2], type='Model', X=2, L='L0'))
  df = rbind(df, data.frame(vals=s_obs_fitted[,3], type='Model', X=3, L='L0'))
  df = rbind(df, data.frame(vals=s_obs_fitted[,4]-1, type='Model', X=4, L='L0')) # do X4 = 0 (binary coded)
  df = rbind(df, data.frame(vals=s_obs_fitted[,5], type='Model', X=5, L='L0'))
  df = rbind(df, data.frame(vals=s_obs_fitted[,6], type='Model', X=6, L='L0'))
  df = rbind(df, data.frame(vals=s_obs_fitted[,7], type='Model', X=7, L='L0'))
  
  df = rbind(df, data.frame(vals=as.numeric(train$simulated_full_data[,1]), type='DGP', X=1, L='L0'))
  df = rbind(df, data.frame(vals=as.numeric(train$simulated_full_data[,2]), type='DGP', X=2, L='L0'))
  df = rbind(df, data.frame(vals=as.numeric(train$simulated_full_data[,3]), type='DGP', X=3, L='L0'))
  df = rbind(df, data.frame(vals=as.numeric(train$simulated_full_data[,4]), type='DGP', X=4, L='L0'))
  df = rbind(df, data.frame(vals=as.numeric(train$simulated_full_data[,5]), type='DGP', X=5, L='L0'))
  df = rbind(df, data.frame(vals=as.numeric(train$simulated_full_data[,6]), type='DGP', X=6, L='L0'))
  df = rbind(df, data.frame(vals=as.numeric(train$simulated_full_data[,7]), type='DGP', X=7, L='L0'))
  
  
  # control
  df = rbind(df, data.frame(vals=s_do_fitted_ct[,1], type='Model', X=1, L='L1'))
  df = rbind(df, data.frame(vals=s_do_fitted_ct[,2], type='Model', X=2, L='L1'))
  df = rbind(df, data.frame(vals=s_do_fitted_ct[,3], type='Model', X=3, L='L1'))
  df = rbind(df, data.frame(vals=s_do_fitted_ct[,4], type='Model', X=4, L='L1'))
  df = rbind(df, data.frame(vals=s_do_fitted_ct[,5], type='Model', X=5, L='L1'))
  df = rbind(df, data.frame(vals=s_do_fitted_ct[,6], type='Model', X=6, L='L1'))
  df = rbind(df, data.frame(vals=s_do_fitted_ct[,7], type='Model', X=7, L='L1'))
  
  
  d_ct = dgp_simulation(n_obs = 5000, SEED = 123, rho = 0.1, doX = c(NA, NA, NA, 0, NA, NA, NA))$simulated_full_data
  df = rbind(df, data.frame(vals=d_ct[,1], type='DGP', X=1, L='L1'))
  df = rbind(df, data.frame(vals=d_ct[,2], type='DGP', X=2, L='L1'))
  df = rbind(df, data.frame(vals=d_ct[,3], type='DGP', X=3, L='L1'))
  df = rbind(df, data.frame(vals=d_ct[,4], type='DGP', X=4, L='L1')) # do X4 = 0 (binary coded)
  df = rbind(df, data.frame(vals=d_ct[,5], type='DGP', X=5, L='L1'))
  df = rbind(df, data.frame(vals=d_ct[,6], type='DGP', X=6, L='L1'))
  df = rbind(df, data.frame(vals=d_ct[,7], type='DGP', X=7, L='L1'))
  

  # treatment
  df = rbind(df, data.frame(vals=s_do_fitted_tx[,1], type='Model', X=1, L='L2'))
  df = rbind(df, data.frame(vals=s_do_fitted_tx[,2], type='Model', X=2, L='L2'))
  df = rbind(df, data.frame(vals=s_do_fitted_tx[,3], type='Model', X=3, L='L2'))
  df = rbind(df, data.frame(vals=s_do_fitted_tx[,4], type='Model', X=4, L='L2'))
  df = rbind(df, data.frame(vals=s_do_fitted_tx[,5], type='Model', X=5, L='L2'))
  df = rbind(df, data.frame(vals=s_do_fitted_tx[,6], type='Model', X=6, L='L2'))
  df = rbind(df, data.frame(vals=s_do_fitted_tx[,7], type='Model', X=7, L='L2'))
  
  
  d_tx = dgp_simulation(n_obs = 5000, SEED = 123, rho = 0.1, doX = c(NA, NA, NA, 1, NA, NA, NA))$simulated_full_data
  df = rbind(df, data.frame(vals=d_tx[,1], type='DGP', X=1, L='L2'))
  df = rbind(df, data.frame(vals=d_tx[,2], type='DGP', X=2, L='L2'))
  df = rbind(df, data.frame(vals=d_tx[,3], type='DGP', X=3, L='L2'))
  df = rbind(df, data.frame(vals=d_tx[,4], type='DGP', X=4, L='L2')) # do X4 = 1 (binary coded)
  df = rbind(df, data.frame(vals=d_tx[,5], type='DGP', X=5, L='L2'))
  df = rbind(df, data.frame(vals=d_tx[,6], type='DGP', X=6, L='L2'))
  df = rbind(df, data.frame(vals=d_tx[,7], type='DGP', X=7, L='L2'))
  
  
  
  p <- ggplot() +
    # For all X except 4: show as histogram
    geom_histogram(data = subset(df, X != 4), 
                   aes(x = vals, col = type, fill = type, y = ..density..), 
                   position = "identity", alpha = 0.2, bins = 30) +
    
    # For X == 4: show as bar plot (binary treatment variable)
    geom_bar(data = subset(df, X == 4),
             aes(x = vals, y = ..prop.. , col = type, fill = type),
             position = "dodge", alpha = 0.4, size = 0.5) +
    
    facet_grid(L ~ X, scales = 'free_y',
               labeller = as_labeller(c(
                 '1' = 'X1', '2' = 'X2', '3' = 'X3',
                 '4' = 'X4 (Tr)', '5' = 'X5', '6' = 'X6',
                 '7' = 'X7 (Y)',
                 'L0' = 'Obs',
                 'L1' = 'Do X4 = 0',
                 'L2' = 'Do X4 = 1'
               ))) +
    labs(y = "Density / Probability", x = "Values") +
    theme_minimal() +
    theme(
      legend.title = element_blank(),
      legend.position = c(0.17, 0.25),
      legend.background = element_rect(fill = "white", colour = "white")
    ) +
    coord_cartesian(ylim = c(0, 1), xlim = NULL)
  
  p
  
  
  # file_name <- paste0(fn, "_L0_L1.pdf")
  # ggsave(file_name, plot=p, width = 8, height = 6)
  # if (FALSE){
  #   file_path <- file.path("~/Library/CloudStorage/Dropbox/Apps/Overleaf/tramdag/figures", basename(file_name))
  #   print(file_path)
  #   ggsave(file_path, plot=p, width = 8/2, height = 6/2)
  # }
  
}


### analyze distributions



# X5 under Tr = 0 and Tr = 1 (estimated)
par(mfrow = c(1,1))
plot(density(s_do_fitted_ct[,5]), main = "X5 under Tr=0 and Tr=1", xlab = "X5", ylab = "Density", ylim=c(0,0.7))
lines(density(s_do_fitted_tx[,5]), col = 'red')


# X5 under Tr = 0 and Tr = 1 (dgp)
lines(density(d_ct[,5]), xlab = "X5", lty= 2)
lines(density(d_tx[,5]), col = 'red', lty=2)
legend("topright", legend=c("Tr=0 (TRAM-DAG)", "Tr=1 (TRAM-DAG)", "Tr=0 (DGP)", "Tr=1 (DGP)"),
       col=c("black", "red", "black", "red"), lty=c(1, 1, 2, 2), cex=0.8, bty = 'n')


# X6 under Tr = 0 and Tr = 1 (estimated)
par(mfrow = c(1,1))
plot(density(s_do_fitted_ct[,6]), main = "X6 under Tr=0 and Tr=1", xlab = "X6", ylab = "Density")
lines(density(s_do_fitted_tx[,6]), col = 'red')
# X6 under Tr = 0 and Tr = 1 (dgp)
lines(density(d_ct[,6]), xlab = "X6", lty= 2)
lines(density(d_tx[,6]), col = 'red', lty=2)
legend("topright", legend=c("Tr=0 (TRAM-DAG)", "Tr=1 (TRAM-DAG)", "Tr=0 (DGP)", "Tr=1 (DGP)"),
       col=c("black", "red", "black", "red"), lty=c(1, 1, 2, 2), cex=0.8, bty = 'n')


# X7 under Tr = 0 and Tr = 1 (estimated)
par(mfrow = c(1,1))
plot(density(s_do_fitted_ct[,7]), main = "X7 under Tr=0 and Tr=1", xlab = "X7", ylab = "Density")
lines(density(s_do_fitted_tx[,7]), col = 'red')
# X7 under Tr = 0 and Tr = 1 (dgp)
lines(density(d_ct[,7]), xlab = "X7", lty= 2)
lines(density(d_tx[,7]), col = 'red', lty=2)
legend("topright", legend=c("Tr=0 (TRAM-DAG)", "Tr=1 (TRAM-DAG)", "Tr=0 (DGP)", "Tr=1 (DGP)"),
       col=c("black", "red", "black", "red"), lty=c(1, 1, 2, 2), cex=0.8, bty = 'n')






############################
# Estimate ITE
############################

# A scenario of observaitonal variables used to fit the model, medical

# | Variable | Meaning                                        | Type       | Role       |
#   | -------- | ---------------------------------------------- | ---------- | ---------- |
#   | T        | Weight loss drug participation                 | Binary     | Treatment  |
#   | X₁       | Age                                            | Continuous | Confounder |
#   | X₂       | Baseline BMI                                   | Continuous | Confounder |
#   | X₃       |Physical activity level      (e.g., hours/week) | Continuous | Covariate  |
#   | X₅       | Body fat percentage                            | Continuous | Mediator   |
#   | X₆       | Insulin resistance (e.g., HOMA-IR)             | Continuous | Mediator   |
#   | Y        | Cardiovascular risk score (after 6 months)       | Continuous | Outcome    |



# another example, marketing.


# | Variable | Description                                         | Role                              |
#   | -------- | --------------------------------------------------- | --------------------------------- |
#   | **T**    | Received marketing email campaign (1 = yes, 0 = no) | Binary treatment                  |
#   | **X₁**   | Prior total spend (last 6 months)                   | Confounder (affects both T and Y) |
#   | **X₂**   | Customer engagement score (website/app activity)    | Confounder (affects both T and Y) |
#   | **X₃**   | Customer satisfaction score from recent survey      | Covariate (affects Y only)        |
#   | **X₅**   | Time spent on website after email                   | Mediator (T → X₅ → Y and X₆)      |
#   | **X₆**   | Number of product pages viewed                      | Downstream mediator (X₅ → X₆ → Y) |
#   | **Y**    | Customer total spend in the next 30 days            | Outcome (continuous)              |
  


##############################

### estimate ITE on test set with patients that received T=0 and T=1


## Train ITE:

# was generated with seed 123
results.dev <- calculate_ITE_median(train)

res.df.train <- results.dev$data$simulated_full_data



## Test ITE:

# generate similar as train but with seed = 1
test <- dgp_simulation(n_obs = 10000, SEED = 1, rho = 0.1, doX = c(NA, NA, NA, NA, NA, NA, NA))

results.val <- calculate_ITE_median(test)

res.df.val <- results.val$data$simulated_full_data



### Plot the results for the ITE based on Median Potential Y


# density ITE (median)

par(mfrow = c(1,2))
plot(density(res.df.train$ITE_median), main = "ITE (median) Density Train", 
     xlab = "ITE", ylab = "Density", ylim=c(0, 2))
lines(density(res.df.train$ITE_median_pred), col = 'red')
legend("topright", legend=c("ITE DGP", "ITE TRAM-DAG"), col=c("black", "red"), 
       lty=1, cex=0.7, bty='n')

plot(density(res.df.val$ITE_median), main = "ITE (median) Density Test",
     xlab = "ITE", ylab = "Density", ylim=c(0, 2))
lines(density(res.df.val$ITE_median_pred), col = 'red')
legend("topright", legend=c("ITE DGP", "ITE TRAM-DAG"), col=c("black", "red"), 
       lty=1, cex=0.7, bty='n')


# scatterplot ITE_median vs. ITE_median_pred

par(mfrow = c(1,2))
plot(res.df.train$ITE_median, res.df.train$ITE_median_pred, 
     main = "ITE Train: True vs. Predicted (median)", 
     xlab = "ITE_median", ylab = "ITE_median_pred")
abline(0, 1, col = 'red', lty = 2)

plot(res.df.val$ITE_median, res.df.val$ITE_median_pred, 
     main = "ITE Test: True vs. Predicted (median)", 
     xlab = "ITE_median", ylab = "ITE_median_pred")
abline(0, 1, col = 'red', lty = 2)



# ITE (median) vs. cATE plot

# STEP 1: Define bin breaks based on training data
breaks <- round(quantile(res.df.train$ITE_median_pred, probs = seq(0, 1, length.out = 7), na.rm = TRUE), 3)

# STEP 2: Group training data and compute ATE per bin
data.dev.grouped.ATE <- res.df.train %>% 
  mutate(ITE.Group = cut(ITE_median_pred, breaks = breaks, include.lowest = TRUE)) %>%
  filter(!is.na(ITE.Group)) %>%
  group_by(ITE.Group) %>%
  group_modify(~ calc.ATE.Continuous(.x)) %>%
  ungroup()

# STEP 3: Group test data using same breaks
data.val.grouped.ATE <- res.df.val %>% 
  mutate(ITE.Group = cut(ITE_median_pred, breaks = breaks, include.lowest = TRUE)) %>%
  filter(!is.na(ITE.Group)) %>%
  group_by(ITE.Group) %>%
  group_modify(~ calc.ATE.Continuous(.x)) %>%
  ungroup()

plot_CATE_vs_ITE_group(
  dev.data = data.dev.grouped.ATE,
  val.data = data.val.grouped.ATE)



###  Note: as result we get 1) the ITE_median_pred which is uses the median for
### the potential outcomes to calculate the ITE and we also get 2) ITE_obsZ_pred 
### which was estimated, using the observed latent sample Z_i, this is used here
### to validate  ITE_true vs. (ITE_obsZ_pred). 

### Note: with linear h as in this DGP, ITE_obsZ_pred and ITE_median_pred
### in the DGP are equal, but in the TRAM-DAG due to CI we will probably not have 
### a straight h. I obesrved, that for the observed Latent variable, estimated ITEs are
### quite off for low or high latent values. I assume that extrapolation 
### is not ideal in some cases.


# maybe check coloured for where the latent_obsZ was outside the 5% and 95% quantiles
par(mfrow = c(1,2))
plot(res.df.train$ITE_true , res.df.train$ITE_obsZ_pred, 
     main = "ITE Train: True vs. Predicted (obsZ)", 
     xlab = "ITE_true", ylab = "ITE_obsZ_pred")
abline(0, 1, col = 'red', lty = 2)
plot(res.df.train$ITE_median , res.df.train$ITE_median_pred,
     main = "ITE Train: True vs. Predicted (median)", 
     xlab = "ITE_true", ylab = "ITE_median_pred")
abline(0, 1, col = 'red', lty = 2)




h_y_tx = h_y_ct = h_X_tx = h_X_ct = h_params_X_tx = h_params_X_ct = xs = seq(-2,1.5,length.out=41)
for (i in 1:length(xs)){
  #i = 1
  x = xs[i]
  
  # Varying y (x7) for control group
  X_ct = tf$constant(c(0, 0, 0, 0, 0, 0, x), shape=c(1L,7L))
  h_params_X_ct =   param_model(X_ct)
  h_X_ct = as.numeric(construct_h(X_ct, h_params_X_ct )$h_combined)
  h_y_ct[i] <- h_X_ct[7]
  
  # Varying y (x7) for treatment group
  X_tx = tf$constant(c(0, 0, 0, 1, 0, 0, x), shape=c(1L,7L))
  h_params_X_tx =   param_model(X_tx)
  h_X_tx = as.numeric(construct_h(X_tx, h_params_X_tx )$h_combined)
  h_y_tx[i] <- h_X_tx[7]
  
  
}

dgp_h_y <- function(y, Tr, data_X, 
                    beta0, beta_t, beta_X, beta_TX) {
  # Function to calculate h_y for varying y
  h_y = y * 4.5 + beta0 + beta_t * Tr + data_X %*% beta_X + (data_X[2:3] %*% beta_TX) * Tr
  return(h_y)
}

par(mfrow=c(1,1))
plot(xs, h_y_tx, type='l', main='h_y for varying y', xlab='y', ylab='h_y', lwd=2)
lines(xs, h_y_ct, col='blue', lwd=2)
curve(dgp_h_y(y=x, Tr=0, data_X =c(0, 0, 0, 0, 0), 
              beta0=0.45, beta_t=1.5, beta_X=c(-0.5, 0.1, 0.2, -0.6, 0.3) , beta_TX=c(-0.9, 0.7)), 
      from=-3, to=3, add=TRUE, col='black', lty=2)
curve(dgp_h_y(y=x, Tr=1, data_X =c(0, 0, 0, 0, 0), 
              beta0=0.45, beta_t=1.5, beta_X=c(-0.5, 0.1, 0.2, -0.6, 0.3) , beta_TX=c(-0.9, 0.7)), 
      from=-3, to=3, add=TRUE, col='blue', lty=2)
legend("topleft", legend=c("h_y (Tr=1)", "h_y (Tr=0)", "DGP h_y (Tr=1)", "DGP h_y (Tr=0)"), 
       col=c("black", "blue", "black", "blue"), lty=c(1, 1, 2, 2), cex=0.8, bty='n')




### Trafos are not entirely parallel (but should be according to the DGP)



### in the trafo h above, i see that it is not really linear between y_dash (-4.5, -1.5)
check_latent <- as.numeric(results.dev$latent_obs[,7]) < -1.5 &
  as.numeric(results.dev$latent_obs[,7]) > -4.5

library(ggplot2)
ggplot(data = results.dev$data$simulated_full_data, aes(x = ITE_true, y = ITE_obsZ_pred)) +
  geom_point(aes(color = check_latent), alpha = 0.5) +
  scale_color_manual(values = c("black", "red"), labels = c("Inside Quantiles", "Outside Quantiles")) +
  labs(title = "Pred ITE Median vs. ITE ObsZ",
       x = "ITE Median", y = "ITE ObsZ") +
  theme_minimal() +
  theme(legend.title = element_blank())




ggplot(data = results.dev$data$simulated_full_data, aes(x = ITE_true, y = ITE_obsZ_pred)) +
  geom_point(aes(color = as.numeric(results.dev$outcome_tx[,7])), alpha = 0.5) +
  labs(title = "Pred ITE Median vs. ITE ObsZ",
       x = "ITE Median", y = "ITE ObsZ") +
  theme_minimal() +
  theme(legend.title = element_blank())

obs_latent_value <- as.numeric(results.dev$latent_obs[,7])

ggplot(data = results.dev$data$simulated_full_data, aes(x = ITE_true, y = ITE_obsZ_pred)) +
  geom_point(aes(color = obs_latent_value), alpha = 0.5) +
  labs(title = "Pred ITE Median vs. ITE ObsZ",
       x = "ITE Median", y = "ITE ObsZ") +
  scale_color_gradient(low = "blue", high = "red", name = "Latent Value") +
  theme_minimal() +
  theme(legend.title = element_blank())


h_y_tx = h_y_ct = h_X_tx = h_X_ct = h_params_X_tx = h_params_X_ct = xs = seq(-2,1.5,length.out=41)
for (i in 1:length(xs)){
  #i = 1
  x = xs[i]
  
  # Varying y (x7) for control group
  X_ct = tf$constant(c(0.5, 1, 1, 0, -0.5, 1, x), shape=c(1L,7L))
  h_params_X_ct =   param_model(X_ct)
  h_X_ct = as.numeric(construct_h(X_ct, h_params_X_ct )$h_combined)
  h_y_ct[i] <- h_X_ct[7]
  
  # Varying y (x7) for treatment group
  X_tx = tf$constant(c(0.5, 1, 1, 1, -0.5, 1, x), shape=c(1L,7L))
  h_params_X_tx =   param_model(X_tx)
  h_X_tx = as.numeric(construct_h(X_tx, h_params_X_tx )$h_combined)
  h_y_tx[i] <- h_X_tx[7]
  
  
}

dgp_h_y <- function(y, Tr, data_X, 
                    beta0, beta_t, beta_X, beta_TX) {
  # Function to calculate h_y for varying y
  h_y = y * 4.5 + beta0 + beta_t * Tr + data_X %*% beta_X + (data_X[2:3] %*% beta_TX) * Tr
  return(h_y)
}

plot(xs, h_y_tx, type='l', main='h_y for varying y', xlab='y', ylab='h_y', lwd=2)
lines(xs, h_y_ct, col='blue', lwd=2)
curve(dgp_h_y(y=x, Tr=0, data_X =c(0.5, 1, 1, -0.5, 1), 
              beta0=0.45, beta_t=1.5, beta_X=c(-0.5, 0.1, 0.2, -0.6, 0.3) , beta_TX=c(-0.9, 0.7)), 
      from=-3, to=3, add=TRUE, col='blue', lty=2)
curve(dgp_h_y(y=x, Tr=1, data_X =c(0.5, 1, 1, -0.5, 1), 
              beta0=0.45, beta_t=1.5, beta_X=c(-0.5, 0.1, 0.2, -0.6, 0.3) , beta_TX=c(-0.9, 0.7)), 
      from=-3, to=3, add=TRUE, col='black', lty=2)

legend("topright", legend=c("h_y (Tr=1)", "h_y (Tr=0)", "DGP h_y (Tr=1)", "DGP h_y (Tr=0)"), 
       col=c("black", "blue", "red", "blue"), lty=c(1, 1, 2, 2), cex=0.8, bty='n')






x_dash <- rlogis(1000)
par(mfrow = c(1,2))
plot(density(tan((x_dash*1/7))))
max(tan(x_dash*1/7))
min(tan(x_dash*1/7))
xx <- seq(-2, 2, by=0.01)
plot(xx, 5*atan(xx))

# my suggestion:
## ITE_median_pred: for ITE prediction on new patients, who did not yet have an observed outcome
## ITE_obsZ_pred: for patients who already had an observed outcome






outside_quantiles <- as.numeric(results.dev$latent_obs[,7]) < quantile(as.numeric(results.dev$latent_obs[,7]), 0.1) |
  as.numeric(results.dev$latent_obs[,7]) > quantile(as.numeric(results.dev$latent_obs[,7]), 0.9)
# plot results.dev$data$simulated_full_data$ITE_median_pred vs results.dev$data$simulated_full_data$ITE_obsZ_pred,
# colored where outside_quantiles with ggplot


library(ggplot2)
ggplot(data = results.dev$data$simulated_full_data, aes(x = ITE_true, y = ITE_obsZ_pred)) +
  geom_point(aes(color = outside_quantiles), alpha = 0.5) +
  scale_color_manual(values = c("black", "red"), labels = c("Inside Quantiles", "Outside Quantiles")) +
  labs(title = "Pred ITE Median vs. ITE ObsZ",
       x = "ITE Median", y = "ITE ObsZ") +
  theme_minimal() +
  theme(legend.title = element_blank())




### in the trafo h above, i see that it is not really linear between y_dash (-4.5, -1.5)
check_latent <- as.numeric(results.dev$latent_obs[,7]) < -1.5 &
  as.numeric(results.dev$latent_obs[,7]) > -4.5

library(ggplot2)
ggplot(data = results.dev$data$simulated_full_data, aes(x = ITE_true, y = ITE_obsZ_pred)) +
  geom_point(aes(color = check_latent), alpha = 0.5) +
  scale_color_manual(values = c("black", "red"), labels = c("Inside Quantiles", "Outside Quantiles")) +
  labs(title = "Pred ITE Median vs. ITE ObsZ",
       x = "ITE Median", y = "ITE ObsZ") +
  theme_minimal() +
  theme(legend.title = element_blank())


df <- data.frame(y_z_ct = as.numeric(results.dev$outcome_ct[,7]), y_z_tx = as.numeric(results.dev$outcome_tx[,7]),
                 y_median_ct = as.numeric(results.dev$outcome_ct_median),
                 y_median_tx = as.numeric(results.dev$outcome_tx_median))

ggplot(data = df, aes(x = y_z_ct, y = y_median_ct)) +
  geom_point(aes(color = outside_quantiles), alpha = 0.5) +
  scale_color_manual(values = c("black", "red"), labels = c("Inside Quantiles", "Outside Quantiles")) +
  #add line 0,1
  geom_abline(slope = 1, intercept = 0, color = 'red', linetype = 'dashed') +
  labs(title = "Outcome Ct: ObsZ vs. Median",
       x = "Outcome Ct (ObsZ)", y = "Outcome Ct (Median)") +
  theme_minimal() +
  theme(legend.title = element_blank())











# to calculate the ITE, i assume that we have new (unseen patients), with following characteristics recorded:

# | Variable | Meaning                                        | Type       | Role       |
#   | -------- | ---------------------------------------------- | ---------- | ---------- |
#   | T        | Weight loss drug participation = 0 (not yet received) | Binary     | Treatment  |
#   | X₁       | Age                                            | Continuous | Confounder |
#   | X₂       | Baseline BMI                                   | Continuous | Confounder |
#   | X₃       |Physical activity level      (e.g., hours/week) | Continuous | Covariate  |
#   | X₅       | Body fat percentage(under T=0)                 | Continuous | Mediator   |
#   | X₆       | Insulin resistance (e.g., HOMA-IR) (under T=0) | Continuous | Mediator   |
# outcome not yet observed


# we have this test data (new patients not yet treated Treatment=0): 
test2 <- dgp_simulation(n_obs = 10000, SEED = 1, rho = 0.1, doX = c(NA, NA, NA, 0, NA, NA, NA))

# this is the full dataset
test$simulated_full_data


# we want to estimate the ITE for these patients, to determine if they should be treated or not
# so we need to calculate the expected outcome under T=0 and T=1


# I estimate the ITE as the difference in 50% quantiles of Y under T=1 and T=0
# ITE_i = Q50_i(T=1) - Q50_i(T=0)



ITE_sampling <- calculate_ITE_median(test2)


  
  
  
  
### Plot the results for the ITE based on Median Potential Y


# density ITE (median)


par(mfrow = c(1,1))
plot(density(ITE_sampling$data$simulated_full_data$ITE_median), main = "ITE (median) Density Train", 
     xlab = "ITE", ylab = "Density", ylim=c(0, 2))
lines(density(ITE_sampling$data$simulated_full_data$ITE_median_pred), col = 'red')
legend("topright", legend=c("ITE DGP", "ITE TRAM-DAG"), col=c("black", "red"), 
       lty=1, cex=0.7, bty='n')




# scatterplot ITE_median vs. ITE_median_pred

par(mfrow = c(1,1), pty = "s")
plot(ITE_sampling$data$simulated_full_data$ITE_median, ITE_sampling$data$simulated_full_data$ITE_median_pred, 
     main = "ITE Train: True vs. Predicted (median)", 
     xlab = "ITE_median", ylab = "ITE_median_pred")
abline(0, 1, col = 'red', lty = 2)










#### Check if the Sampled values for T=0 are equal to the observed values (when T was assumed == 0)

# --> Yes the sampling for control group is correct because it is equal to the observed:

df_sampled <- as.matrix(ITE_sampling$outcome_ct)
df_true <- as.matrix(test2$dat.tf)

# Compare the density of X1 when T=0 between observed and sampled
plot(density(df_sampled[,1]), main = "X1", ylab = "Density", col = 'red', 
     lty = 1, lwd = 2)
lines(density(df_true[,1]), col = 'blue', lty = 2, lwd = 2)
legend("topright", legend = c("Sampled", "True"), 
       col = c("red", "blue"), lty = c(1, 2), lwd = 2, cex = 0.8)

# Compare the density of X2 when T=0 between observed and sampled
plot(density(df_sampled[,2]), main = "X2", ylab = "Density", col = 'red', 
     lty = 1, lwd = 2)
lines(density(df_true[,2]), col = 'blue', lty = 2, lwd = 2)
legend("topright", legend = c("Sampled", "True"), 
       col = c("red", "blue"), lty = c(1, 2), lwd = 2, cex = 0.8)

# Compare the density of X3 when T=0 between observed and sampled
plot(density(df_sampled[,3]), main = "X3", ylab = "Density", col = 'red', 
     lty = 1, lwd = 2)
lines(density(df_true[,3]), col = 'blue', lty = 2, lwd = 2)
legend("topright", legend = c("Sampled", "True"), 
       col = c("red", "blue"), lty = c(1, 2), lwd = 2, cex = 0.8)

# Compare the fractions of X4 when T=0 between observed and sampled
table(df_sampled[,4])
table(df_true[,4])

# Compare the density of X5 when T=0 between observed and sampled
plot(density(df_sampled[,5]), main = "X5", ylab = "Density", col = 'red', 
     lty = 1, lwd = 2)
lines(density(df_true[,5]), col = 'blue', lty = 2, lwd = 2)
legend("topright", legend = c("Sampled", "True"), 
       col = c("red", "blue"), lty = c(1, 2), lwd = 2, cex = 0.8)

# Compare the density of X6 when T=0 between observed and sampled
plot(density(df_sampled[,6]), main = "X6", ylab = "Density", col = 'red', 
     lty = 1, lwd = 2)
lines(density(df_true[,6]), col = 'blue', lty = 2, lwd = 2)
legend("topright", legend = c("Sampled", "True"), 
       col = c("red", "blue"), lty = c(1, 2), lwd = 2, cex = 0.8)





