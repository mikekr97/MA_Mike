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
# source('code/utils/ITE_utils.R')

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
# ITE in a simple RCT
###############################################################################

# 
# 
# ##### DGP ########
# dgp <- function(n_obs=20000, doX=c(NA, NA, NA, NA), SEED=123, 
#                 p=2, p0=0, confounder=FALSE, main_effect = -0.85,
#                 interaction_effect = 0.7) {
#   #n_obs = 1e5 n_obs = 10
#   set.seed(SEED)
#   
#   # Data simulation
#   
#   ## Case 1: continuous random variables
#   
#   # Define sample size
#   n <- n_obs
#   
#   # Generate random binary treatment T
#   Tr <- rbinom(n, size = 1, prob = 0.5)
#   
#   # p <- 2  # number of variables 
#   
#   # Define the mean vector (all zeros for simplicity)
#   mu <- rep(0, p+p0)  # Mean vector of length p
#   
#   # Define the covariance matrix (compound symmetric for simplicity)
#   rho <- 0.1  # Correlation coefficient
#   Sigma <- matrix(rho, nrow = (p+p0), ncol = (p+p0))  # Start with all elements as rho
#   diag(Sigma) <- 1  # Set diagonal elements to 1 (variances)
#   
#   # Generate n samples from the multivariate normal distribution
#   data <- MASS::mvrnorm(n, mu = mu, Sigma = Sigma)
#   colnames(data) <- paste0("X", 1:(p+p0))
#   
#   beta_0 <- 0.45
#   beta_t <- main_effect # -0.85 default
#   beta_X <- c(-0.5, 0.1, rep(0, p0))  # p0 variables with no effect on outcome
#   beta_TX <- interaction_effect # 0.7 default
#   
#   if(confounder) {
#     # Add use the first variable X1 as confounder to affect Tr
#     Tr <- rbinom(n, size = 1, prob = plogis(0.5 * data[,1]))
#   }
#   
#   # Calculate the linear predictor (logit)
#   logit_Y <- beta_0 + beta_t * Tr + data %*% beta_X + (data[,1] * beta_TX) * Tr
#   
#   
#   # Convert logit to probability of outcome
#   Y_prob <- plogis(logit_Y)
#   
#   # Generate binary outcome Y based on the probability
#   Y <- rbinom(n, size = 1, prob = Y_prob)
#   
#   # Potential outcome for treated and untreated
#   Y1 <- plogis(beta_0 + beta_t + data %*% beta_X + data[,1] * beta_TX)
#   # Y1 <- plogis(beta_0 + beta_t + data %*% beta_X + data %*% beta_TX)
#   Y0 <- plogis(beta_0 + data %*% beta_X)
#   
#   # Calculate the individual treatment effect
#   ITE_true <- Y1 - Y0
#   # summary(data)
#   # sd(data[,1])
#   # mean(data[,1])
#   # 
#   # data[,1] <- scale(data[,1])
#   # data[,2] <- (data[,2]-mean(data[,2]))/sd(data[,2])
#   
#   # Combine all variables into a single data frame
#   simulated_full_data <- data.frame(ID = 1:n, Y=Y, Treatment=Tr, data, Y1, Y0, ITE_true, Y_prob)
#   
#   # Data for testing ITE models
#   simulated_data <- data.frame(ID =1:n, Y=Y, Treatment=Tr, Tr=Tr, data, ITE_true = ITE_true, Y_prob=Y_prob) %>% 
#     # add Treatment variable Tr=Treatment
#     mutate(Treatment = ifelse(Treatment==1,"Y", "N")) %>% 
#     mutate(Treatment = factor(Treatment, levels = c("N", "Y")))
#   
#   
#   set.seed(12345)
#   test.data <- split_data(simulated_data, 1/2)
#   test.compl.data <- remove_NA_data(test.data)
#   
#   
#   # for two-model structure we only need the 2 patient specific variables (no Tr)
#   A <- matrix(c(0, 0, 0, 1, 
#                 0, 0, 0, 1,
#                 0, 0, 0, 1,
#                 0, 0, 0, 0), nrow = 4, ncol = 4, byrow = TRUE)
#   
#   # Full dataset
#   dat.orig =  data.frame(x1 = simulated_full_data$Treatment, 
#                          x2 = simulated_full_data$X1, 
#                          x3 = simulated_full_data$X2, 
#                          x4 = simulated_full_data$Y)
#   dat_temp <- as.matrix(dat.orig)
#   # dat_temp[,4] <- dat_temp[,4] + 1
#   dat_temp[,c(1,4)] <- dat_temp[,c(1,4)] + 1
#   dat.tf = tf$constant(as.matrix(dat_temp), dtype = 'float32')
#   
#   # train dataset
#   dat.train <- data.frame(x1 = test.compl.data$data.dev$Tr, 
#                           x2 = test.compl.data$data.dev$X1, 
#                           x3 = test.compl.data$data.dev$X2, 
#                           x4 = test.compl.data$data.dev$Y)
#   dat_temp <- as.matrix(dat.train)
#   # dat_temp[,4] <- dat_temp[,4] + 1
#   dat_temp[,c(1,4)] <- dat_temp[,c(1,4)] + 1
#   dat.train.tf = tf$constant(as.matrix(dat_temp), dtype = 'float32')
#   
#   dat.test <- data.frame(x1 = test.compl.data$data.val$Tr, 
#                          x2 = test.compl.data$data.val$X1, 
#                          x3 = test.compl.data$data.val$X2, 
#                          x4 = test.compl.data$data.val$Y)
#   dat_temp <- as.matrix(dat.test)
#   # dat_temp[,4] <- dat_temp[,4] + 1
#   dat_temp[,c(1,4)] <- dat_temp[,c(1,4)] + 1
#   dat.test.tf = tf$constant(as.matrix(dat_temp), dtype = 'float32')
#   
#   
#   q1 = c(1, 2)
#   q2 = quantile(dat.orig[,2], probs = c(0.05, 0.95))
#   q3 = quantile(dat.orig[,3], probs = c(0.05, 0.95))
#   q4 = c(1, 2) #No Quantiles for ordinal data
#   # q1 = quantile(dat.orig[,2], probs = c(0.05, 0.95)) 
#   # q2 = quantile(dat.orig[,3], probs = c(0.05, 0.95))
#   # q3 = c(0, 1) #No Quantiles for ordinal data
#   
#   
#   return(list(
#     df_orig=dat.tf, 
#     df_R = dat.orig,
#     min =  tf$reduce_min(dat.tf, axis=0L),
#     max =  tf$reduce_max(dat.tf, axis=0L),
#     min = tf$constant(c(q1[1], q2[1], q3[1], q4[1]), dtype = 'float32'),
#     max = tf$constant(c(q1[2], q2[2], q3[2], q4[2]), dtype = 'float32'),
#     
#     # min = tf$constant(c(q1[1], q2[1], q3[1]), dtype = 'float32'),
#     # max = tf$constant(c(q1[2], q2[2], q3[2]), dtype = 'float32'),
#     type = c('o', 'c', 'c', 'o'),
#     A=A,
#     
#     #train
#     df_R_train = dat.train,
#     df_orig_train = dat.train.tf,
#     
#     
#     # df_orig_train_ct = dat.train.ct.tf,
#     # df_R_train_ct = dat.train.ct,
#     # 
#     # df_orig_train_tx = dat.train.tx.tf,
#     # df_R_train_tx = dat.train.tx,
#     
#     #test
#     df_R_test = dat.test,
#     df_orig_test = dat.test.tf,
#     
#     # df_orig_test_ct = dat.test.ct.tf,
#     # df_R_test_ct = dat.test.ct,
#     # 
#     # df_orig_test_tx = dat.test.tx.tf,
#     # df_R_test_tx = dat.test.tx,
#     # 
#     #full
#     simulated_full_data = simulated_full_data,
#     simulated_data = simulated_data,
#     test.compl.data = test.compl.data,
#     dgp_params = list(
#       beta_0 = beta_0,
#       beta_t = beta_t,
#       beta_X = beta_X,
#       beta_TX = beta_TX
#     )
#   ))
# } 
# 
# 
# 
# 
# 

# n_obs <- 20000
# 
# # specify number of predictor variables
# p <- 2
# 
# # specify number of variables without effect
# p0 <- 0
# 
# 
# dgp_data = dgp(n_obs, p=p, p0=p0, SEED=123, confounder=FALSE)
# 
# # percentage of patients with Y=1
# mean(dgp_data$simulated_full_data$Y)
# 
# # percentage of patients with Y=1 in Control (train)
# mean(dgp_data$test.compl.data$data.dev.ct$Y)
# 
# # percentage of patients with Y=1 in Treatment (train)
# mean(dgp_data$test.compl.data$data.dev.tx$Y)
# 
# dgp_data$df_orig_test
# 
# dgp_data$simulated_full_data
# 
# boxplot(Y_prob ~ Y, data = dgp_data$simulated_full_data)




#### new dgp for simulation (for choosing different scenarios)


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
  
  
  
  #### Y (X7): continuous outcome dependent on Tr, X2-X6 (with Tr interactions X3, X4)
  
  data_X <- as.matrix(data.frame(X1 = data[,1], X2 = data[,2], X3 = data[,3], X5 = X5, X6 = X6))
  
  
  if (is.na(doX[6])){
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
    
    
  } else{
    X7 = rep(doX[7], n_obs)
  }
  
  
  # Calculate the individual treatment effect
  ITE_true <- X7_tx - X7_ct

  
  
  
  
  # Combine all variables into a single data frame (observed)
  simulated_full_data <- data.frame(X1 = data_X[,1],
                                    X2 = data_X[,2],
                                    X3 = data_X[,3],
                                    Tr=Tr, 
                                    X5 = X5,
                                    X6 = X6,
                                    Y=X7, 
                                    ITE_true)
  
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
(global_min = train$min)
(global_max = train$max)
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
  (global_min = min)
  (global_max = min)
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
  
  
  p = ggplot() +
    geom_histogram(data = df, 
                   aes(x=vals, col=type, fill=type, y=..density..), 
                   position = "identity", alpha=0.2) +
    facet_grid(L ~ X, scales = 'free_y',
               labeller = as_labeller(c(
                 '1' = 'X1', '2' = 'X2', '3' = 'X3',
                 '4' = 'X4 (Tr)', '5' = 'X5', '6' = 'X6',
                 '7' = 'X7 (Y)',
                 'L0' = 'Obs',
                 'L1' = 'Do X4 = 0',
                 'L2' = 'Do X4 = 1'
               ))) +
    labs(y = "Density", x = "Values") +
    theme_minimal() +
    theme(
      legend.title = element_blank(),
      legend.position = c(0.17, 0.25),
      legend.background = element_rect(fill = "white", colour = "white")
    ) +
    coord_cartesian(ylim = c(0, 2), xlim = NULL)
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
plot(density(s_do_fitted_ct[,5]), main = "X5 under Tr=0 and Tr=1", xlab = "X5", ylab = "Density")
lines(density(s_do_fitted_tx[,5]), col = 'red')

# X5 under Tr = 0 and Tr = 1 (dgp)
lines(density(d_ct[,5]), xlab = "X5", lty= 2)
lines(density(d_tx[,5]), col = 'red', lty=2)
legend("topright", legend=c("Tr=0 (TRAM-DAG)", "Tr=1 (TRAM-DAG)", "Tr=0 (DGP)", "Tr=1 (DGP)"),
       col=c("black", "red", "black", "red"), lty=c(1, 1, 2, 2), cex=0.8)


# X6 under Tr = 0 and Tr = 1 (estimated)
par(mfrow = c(1,1))
plot(density(s_do_fitted_ct[,6]), main = "X6 under Tr=0 and Tr=1", xlab = "X6", ylab = "Density")
lines(density(s_do_fitted_tx[,6]), col = 'red')
# X6 under Tr = 0 and Tr = 1 (dgp)
lines(density(d_ct[,6]), xlab = "X6", lty= 2)
lines(density(d_tx[,6]), col = 'red', lty=2)
legend("topright", legend=c("Tr=0 (TRAM-DAG)", "Tr=1 (TRAM-DAG)", "Tr=0 (DGP)", "Tr=1 (DGP)"),
       col=c("black", "red", "black", "red"), lty=c(1, 1, 2, 2), cex=0.8)


# X7 under Tr = 0 and Tr = 1 (estimated)
par(mfrow = c(1,1))
plot(density(s_do_fitted_ct[,7]), main = "X7 under Tr=0 and Tr=1", xlab = "X7", ylab = "Density")
lines(density(s_do_fitted_tx[,7]), col = 'red')
# X7 under Tr = 0 and Tr = 1 (dgp)
lines(density(d_ct[,7]), xlab = "X7", lty= 2)
lines(density(d_tx[,7]), col = 'red', lty=2)
legend("topright", legend=c("Tr=0 (TRAM-DAG)", "Tr=1 (TRAM-DAG)", "Tr=0 (DGP)", "Tr=1 (DGP)"),
       col=c("black", "red", "black", "red"), lty=c(1, 1, 2, 2), cex=0.8)






############################
# Estimate ITE
############################

# these were the observaitonal variables used to fit the model

# | Variable | Meaning                                        | Type       | Role       |
#   | -------- | ---------------------------------------------- | ---------- | ---------- |
#   | T        | Weight loss drug participation                 | Binary     | Treatment  |
#   | X₁       | Age                                            | Continuous | Confounder |
#   | X₂       | Baseline BMI                                   | Continuous | Confounder |
#   | X₃       |Physical activity level      (e.g., hours/week) | Continuous | Covariate  |
#   | X₅       | Body fat percentage                            | Continuous | Mediator   |
#   | X₆       | Insulin resistance (e.g., HOMA-IR)             | Continuous | Mediator   |
#   | Y        | Cardiovascular risk score (after 6 months)       | Continuous | Outcome    |


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
test <- dgp_simulation(n_obs = 10000, SEED = 1, rho = 0.1, doX = c(NA, NA, NA, 0, NA, NA, NA))

# this is the full dataset
test$simulated_full_data


# we want to estimate the ITE for these patients, to determine if they should be treated or not
# so we need to calculate the expected outcome under T=0 and T=1


# I estimate the ITE as the difference in 50% quantiles of Y under T=1 and T=0
# ITE_i = Q50_i(T=1) - Q50_i(T=0)

calculate_ITE_median <- function(data){
  # data <- test
  
  # from the observed patient characteristics determine the latent value

  
  # NN outputs (CS, LS, theta') at the observed values
  h_params_obs <- param_model(data$dat.tf)
  
  # combine outputs to the transformation function
  h_obs <- construct_h(t_i = data$dat.tf, h_params = h_params_obs)
  
  # this is the cut point for the Treatment (X4), it is not used because we will intervene on this variable
  # h_obs$h_ord_vars
  
  # these are the latent values for the continuous variables (X1, X2, X3, X5, X6, X7 (outcome, not used))
  # h_obs$h_cont_vars
  
  ### Note that we only use the latent values for the observed patient characteristics
  ### (X1, X2, X3, X5, X6) , where X5 and X6 depended on the treatment received (here T=0), 
  ### the received treatment was considered when h was constructed with construct_h
  ### because I assume that the patients were not yet treated, all is constructed with T=0
  
  ### in ITE estimation, the outcome is not known already, we only 
  
  h_obs$h_cont_vars
  
  # prepare tensor for control
  
  # set data$dat.tf[,4] <- 0
  
  
  
}




construct_h <- function (t_i, h_params){

  #t_i <- data$dat.tf   # original data x1, x2, x3 for each obs
  #h_params = h_params_obs    # NN outputs (CS, LS, theta') for each obs
  k_min <- k_constant(global_min)
  k_max <- k_constant(global_max)
  
  # from the last dimension of h_params the first entry is h_cs1
  # the second to |X|+1 are the LS
  # the 2+|X|+1 to the end is H_I
  
  # complex shifts for each observation
  h_cs <- h_params[,,1, drop = FALSE]
  
  # linear shifts for each observation
  h_ls <- h_params[,,2, drop = FALSE]
  #LS
  h_LS = tf$squeeze(h_ls, axis=-1L) # throw away last dimension
  #CS
  h_CS = tf$squeeze(h_cs, axis=-1L)
  theta_tilde <- h_params[,,3:dim(h_params)[3], drop = FALSE]
  #Thetas for intercept -> to_theta3 to make them increasing
  theta = to_theta3(theta_tilde)
  
  if (!exists('data_type')){ #Defaulting to all continuous 
    cont_dims = 1:dim(theta_tilde)[2]
    cont_ord = c()
  } else{ 
    cont_dims = which(data_type == 'c')
    cont_ord = which(data_type == 'o')
  }
  if (len_theta == -1){ 
    len_theta = dim(theta_tilde)[3]
  }

  ### Continiuous dimensions
  #### At least one continuous dimension exits
  if (length(cont_dims) != 0){
    
    # inputs in h_dag_extra:
    # data=(40000, 3), 
    # theta=(40000, 3, 20), k_min=(3), k_max=(3))
    
    # creates the value of the Bernstein at each observation
    # and current parameters: output shape=(40000, 3)
    # h_I = h_dag_extra(t_i[,cont_dims, drop=FALSE], theta[,cont_dims,1:len_theta,drop=FALSE], k_min[cont_dims], k_max[cont_dims])
    h_I = h_dag_extra(tf$gather(t_i, as.integer(cont_dims-1L), axis = 1L), 
                      tf$gather(theta, as.integer(cont_dims-1L), axis = 1L)[,,1:len_theta,drop=FALSE],
                      tf$gather(k_min, as.integer(cont_dims-1L)),
                      tf$gather(k_max, as.integer(cont_dims-1L)))
    
    
    # adding the intercepts and shifts: results in shape=(40000, 3)
    # basically the estimated value of the latent variable
    h_cont_vars = h_I + tf$gather(h_LS, as.integer(cont_dims-1L), axis = 1L) + 
      tf$gather(h_CS, as.integer(cont_dims-1L), axis = 1L)
    
    
  }
  
  ### Ordinal dimensions
  if (length(cont_ord) != 0){
    B = tf$shape(t_i)[1]
    for (col in cont_ord){
      # col=4
      # nol = tf$cast(k_max[col] - 1L, tf$int32) # Number of cut-points in respective dimension
      nol = tf$cast(k_max[col], tf$int32) # Number of cut-points in respective dimension (binary encoded)
      
      theta_ord = theta[,col,1:nol,drop=TRUE] # Intercept (2 values per observation if 2 cutpoints)
      
      
      h_ord_vars = theta_ord + h_LS[,col, drop=FALSE] + h_CS[,col, drop=FALSE]
    }
  }
  
  ### DEBUG 
  #if (sum(is.infinite(log_lik$numpy())) > 0){
  #  print("Hall")
  #}
  return (list(
    h_cont_vars = h_cont_vars, 
    h_ord_vars = h_ord_vars))
}

