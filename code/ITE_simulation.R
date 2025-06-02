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
DIR = 'runs/ITE_simulation/run'
if (!dir.exists(DIR)) {
  dir.create(DIR, recursive = TRUE)
}
# Copy this file to the directory DIR
file.copy('/code/ITE_simulation.R', file.path(DIR, 'ITE_simulation.R'), overwrite=TRUE)
# file.copy('/code/TRAM_DAG_ITE_simulation_pt2_single_model_newDGP_CS_Relu_CI.R', file.path(DIR, 'TRAM_DAG_ITE_simulation_pt2_single_model_newDGP_CS_Relu_CI.R'), overwrite=TRUE)

# 
# len_theta = 20 # Number of coefficients of the Bernstein polynomials
# hidden_features_I = c(4, 5, 5, 4) # c(3,3,3,3) 
# hidden_features_CS = c(2,5,5,2) # c(4,8,10,8,4)#c(2,5,5,2) #c(4,8,8,4)# c(2,5,5,2)
# 
# 
# MA =  matrix(c(
#   0,   0,  0, 'cs',
#   0,   0,  0, 'cs',
#   0,   0,  0, 'ls',
#   0,   0,  0,   0), nrow = 4, ncol = 4, byrow = TRUE)
# MODEL_NAME = 'ModelCS'
# # 



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
dgp_simulation <- function(n_obs=20000, doX=c(NA, NA, NA, NA), SEED=123,
                rho = 0.1,     # correlation coeff between Xs
                beta_0 = 0.45, # intercept
                beta_t = -0.85, # main treatment effect
                beta_X = c(-0.5, 0.1), # main effects of Xs
                beta_TX = c(0.7), # interaction effects of Xs with treatment  
                p0 = 0, # number of variables without effect
                confounder=FALSE,  #index of confounder (only single possible)
                drop=FALSE) {   # eg. drop = c("X1", "X3") these are dropped from the final dataset
  #n_obs = 1e5 n_obs = 10
  set.seed(SEED)
  
  # Data simulation
  
  # Define sample size
  n <- n_obs
  
  p <- length(beta_X) # number of variables with effect
  
  # Define the mean vector (all zeros for simplicity)
  mu <- rep(0, p+p0)  # Mean vector of length p
  
  # Define the covariance matrix (compound symmetric for simplicity)
  rho <- rho  # Correlation coefficient
  Sigma <- matrix(rho, nrow = (p+p0), ncol = (p+p0))  # Start with all elements as rho
  diag(Sigma) <- 1  # Set diagonal elements to 1 (variances)
  
  # Generate n samples from the multivariate normal distribution
  data <- MASS::mvrnorm(n, mu = mu, Sigma = Sigma)
  colnames(data) <- paste0("X", 1:(p+p0))
  
  beta_0 <- beta_0
  beta_t <- beta_t # -0.85 default
  beta_X <- c(beta_X, rep(0, p0))  # p0 variables with no effect on outcome
  beta_TX <- beta_TX # 0.7 default
  
  if(confounder != FALSE) {
    # Add use the first variable X1 as confounder to affect Tr
    Tr <- rbinom(n, size = 1, prob = plogis(0.5 * data[,confounder]))
  } else {
    # Generate random binary treatment T
    Tr <- rbinom(n, size = 1, prob = 0.5)
  }
  
  # Calculate the linear predictor (logit)
  logit_Y <- beta_0 + beta_t * Tr + data %*% beta_X + (as.matrix(data[,c(1:length(beta_TX))]) %*% beta_TX) * Tr
  
  
  # Convert logit to probability of outcome
  Y_prob <- plogis(logit_Y)
  
  # Generate binary outcome Y based on the probability
  Y <- rbinom(n, size = 1, prob = Y_prob)
  
  # Potential outcome for treated and untreated
  Y1 <- plogis(beta_0 + beta_t + data %*% beta_X + (as.matrix(data[,c(1:length(beta_TX))]) %*% beta_TX))
  # Y1 <- plogis(beta_0 + beta_t + data %*% beta_X + data %*% beta_TX)
  Y0 <- plogis(beta_0 + data %*% beta_X)
  
  # Calculate the individual treatment effect
  ITE_true <- Y1 - Y0
  # summary(data)
  # sd(data[,1])
  # mean(data[,1])
  # 
  # data[,1] <- scale(data[,1])
  # data[,2] <- (data[,2]-mean(data[,2]))/sd(data[,2])
  
  # Combine all variables into a single data frame
  simulated_full_data <- data.frame(ID = 1:n, Y=Y, Treatment=Tr, data, Y1, Y0, ITE_true, Y_prob)
  
  # Data for testing ITE models
  simulated_data <- data.frame(ID =1:n, Y=Y, Treatment=Tr, Tr=Tr, data, ITE_true = ITE_true, Y_prob=Y_prob) %>% 
    # add Treatment variable Tr=Treatment
    mutate(Treatment = ifelse(Treatment==1,"Y", "N")) %>% 
    mutate(Treatment = factor(Treatment, levels = c("N", "Y")))
  
  
  if (!isFALSE(drop) && length(drop) > 0) {
    valid_drop <- intersect(drop, colnames(simulated_data))  # Keep original `drop` unchanged
    simulated_data <- simulated_data %>% dplyr::select(-all_of(valid_drop))
    
    # Identify remaining X columns and rename them sequentially
    x_cols <- grep("^X", colnames(simulated_data), value = TRUE)
    new_x_names <- paste0("X", seq_along(x_cols))
    
    names(simulated_data)[match(x_cols, names(simulated_data))] <- new_x_names
  }
  
  
  set.seed(12345)
  test.data <- split_data(simulated_data, 1/2)
  test.compl.data <- remove_NA_data(test.data)
  
  # 
  # # for two-model structure we only need the 2 patient specific variables (no Tr)
  # A <- matrix(c(0, 0, 0, 1, 
  #               0, 0, 0, 1,
  #               0, 0, 0, 1,
  #               0, 0, 0, 0), nrow = 4, ncol = 4, byrow = TRUE)
  # 
  # # Full dataset
  # dat.orig =  data.frame(x1 = simulated_full_data$Treatment, 
  #                        x2 = simulated_full_data$X1, 
  #                        x3 = simulated_full_data$X2, 
  #                        x4 = simulated_full_data$Y)
  # dat_temp <- as.matrix(dat.orig)
  # # dat_temp[,4] <- dat_temp[,4] + 1
  # dat_temp[,c(1,4)] <- dat_temp[,c(1,4)] + 1
  # dat.tf = tf$constant(as.matrix(dat_temp), dtype = 'float32')
  # 
  # # train dataset
  # dat.train <- data.frame(x1 = test.compl.data$data.dev$Tr, 
  #                         x2 = test.compl.data$data.dev$X1, 
  #                         x3 = test.compl.data$data.dev$X2, 
  #                         x4 = test.compl.data$data.dev$Y)
  # dat_temp <- as.matrix(dat.train)
  # # dat_temp[,4] <- dat_temp[,4] + 1
  # dat_temp[,c(1,4)] <- dat_temp[,c(1,4)] + 1
  # dat.train.tf = tf$constant(as.matrix(dat_temp), dtype = 'float32')
  # 
  # dat.test <- data.frame(x1 = test.compl.data$data.val$Tr, 
  #                        x2 = test.compl.data$data.val$X1, 
  #                        x3 = test.compl.data$data.val$X2, 
  #                        x4 = test.compl.data$data.val$Y)
  # dat_temp <- as.matrix(dat.test)
  # # dat_temp[,4] <- dat_temp[,4] + 1
  # dat_temp[,c(1,4)] <- dat_temp[,c(1,4)] + 1
  # dat.test.tf = tf$constant(as.matrix(dat_temp), dtype = 'float32')
  # 
  # 
  # q1 = c(1, 2)
  # q2 = quantile(dat.orig[,2], probs = c(0.05, 0.95))
  # q3 = quantile(dat.orig[,3], probs = c(0.05, 0.95))
  # q4 = c(1, 2) #No Quantiles for ordinal data
  # # q1 = quantile(dat.orig[,2], probs = c(0.05, 0.95)) 
  # # q2 = quantile(dat.orig[,3], probs = c(0.05, 0.95))
  # # q3 = c(0, 1) #No Quantiles for ordinal data
  # 
  # 
  return(list(
    # df_orig=dat.tf, 
    # df_R = dat.orig,
    # min =  tf$reduce_min(dat.tf, axis=0L),
    # max =  tf$reduce_max(dat.tf, axis=0L),
    # min = tf$constant(c(q1[1], q2[1], q3[1], q4[1]), dtype = 'float32'),
    # max = tf$constant(c(q1[2], q2[2], q3[2], q4[2]), dtype = 'float32'),
    # 
    # # min = tf$constant(c(q1[1], q2[1], q3[1]), dtype = 'float32'),
    # # max = tf$constant(c(q1[2], q2[2], q3[2]), dtype = 'float32'),
    # type = c('o', 'c', 'c', 'o'),
    # A=A,
    # 
    # #train
    # df_R_train = dat.train,
    # df_orig_train = dat.train.tf,
    
    
    # df_orig_train_ct = dat.train.ct.tf,
    # df_R_train_ct = dat.train.ct,
    # 
    # df_orig_train_tx = dat.train.tx.tf,
    # df_R_train_tx = dat.train.tx,
    
    # #test
    # df_R_test = dat.test,
    # df_orig_test = dat.test.tf,
    # 
    # df_orig_test_ct = dat.test.ct.tf,
    # df_R_test_ct = dat.test.ct,
    # 
    # df_orig_test_tx = dat.test.tx.tf,
    # df_R_test_tx = dat.test.tx,
    # 
    #full
    simulated_full_data = simulated_full_data,
    simulated_data = simulated_data,
    test.compl.data = test.compl.data,
    dgp_params = list(
      beta_0 = beta_0,
      beta_t = beta_t,
      beta_X = beta_X,
      beta_TX = beta_TX
    )
  ))
} 




#################################################
# ITE Utils (supporting functions)
#################################################


calc.ATE.Risks <- function(data) {
  data <- as.data.frame(data)
  
  # Calculate proportions
  p1 <- mean(data$Y[data$Tr ==  1])  # treated risk
  p0 <- mean(data$Y[data$Tr == 0])  # control risk
  
  # Risk difference
  ATE.RiskDiff <- p1 - p0
  
  # Standard error for RD (Wald)
  n1 <- sum(data$Tr == 1)
  n0 <- sum(data$Tr == 0)
  se <- sqrt((p1 * (1 - p1)) / n1 + (p0 * (1 - p0)) / n0)
  
  # 95% CI using normal approximation
  z <- qnorm(0.975)
  ATE.lb <- ATE.RiskDiff - z * se
  ATE.ub <- ATE.RiskDiff + z * se
  
  return(data.frame(
    ATE.RiskDiff = ATE.RiskDiff,
    ATE.lb = ATE.lb,
    ATE.ub = ATE.ub,
    n.total = nrow(data),
    n.tr = n1,
    n.ct = n0
  ))
}



plot_ATE_ITE_in_group_risks <- function(dev.data = data.dev.rs, val.data = data.val.rs, ylb=0, yub=2){
  data <- rbind(dev.data %>% mutate(sample = "derivation"), val.data %>%  mutate(sample = "validation"))
  result <- ggplot(data, aes(x = ITE.Group, y = ATE.RiskDiff)) +
    geom_line(aes(group = sample, color = sample), linewidth = 1, 
              position = position_dodge(width = 0.2)) +
    geom_point(aes(color = sample), size = 1.5, 
               position = position_dodge(width = 0.2)) +
    geom_errorbar(aes(ymin = ATE.lb, ymax = ATE.ub, color = sample), width = 0.2,
                  position = position_dodge(width = 0.2))+
    geom_hline(yintercept = 0, linetype = "dashed", color = "black") +
    scale_color_manual(name = "Group",
                       labels = c("derivation" = "Training Data", "validation" = "Test Data"),
                       values = c("orange", "#36648B"))+
    scale_x_discrete(guide = guide_axis(n.dodge = 2))+
    ylim(min(dev.data$ATE.lb)-0.1, max(dev.data$ATE.ub)+0.1)+
    xlab("ITE Group")+
    ylab("ATE in Risk Difference")+
    theme_minimal()+
    theme(
      legend.position.inside = c(0.9, 0.9),
      legend.justification = c("right", "top"),
      legend.box.just = "right",
      panel.grid.major = element_blank(),  # Removes major grid lines
      panel.grid.minor = element_blank(),  # Removes minor grid lines
      panel.background = element_blank(),  # Removes panel background
      plot.background = element_blank(),
      text = element_text(size = 14),
      axis.line = element_line(color = "black"),
      axis.ticks = element_line(color = "black")
    )
  
  return(result)
}





library(gridExtra)
library(ggpubr)
plot_pred_ite <- function(model.results, ate_ite = FALSE){
  # train
  p_dev_plot <- ggplot(model.results$data.dev.rs, aes(x = Y_prob, y = Y_pred, color = Treatment)) +
    geom_point() +
    geom_abline(slope = 1, intercept = 0, color = "red") +
    labs(x = "True Probabilities", y = "Estimated Probabilities", title = "Prob (Train)") +
    theme_minimal() +
    theme(legend.position = "top")
  
  # test
  p_val_plot <- ggplot(model.results$data.val.rs, aes(x = Y_prob, y = Y_pred, color = Treatment)) +
    geom_point() +
    geom_abline(slope = 1, intercept = 0, color = "red") +
    labs(x = "True Probabilities", y = "Estimated Probabilities", title = "Prob (Test)") +
    theme_minimal() +
    theme(legend.position = "top")
  
  
  
  ite_dev_plot <- ggplot(model.results$data.dev.rs, aes(x=ITE_true, y=ITE, color=Treatment)) +
    geom_point() +
    geom_abline(slope = 1, intercept = 0, color = "red") +
    labs(title = "ITE (Train)", x = "True ITE", y = "Estimated ITE") +
    theme_minimal() +
    theme(legend.position = "top")
  
  ite_val_plot <- ggplot(model.results$data.val.rs, aes(x=ITE_true, y=ITE, color=Treatment)) +
    geom_point() +
    geom_abline(slope = 1, intercept = 0, color = "red") +
    labs(title = "ITE (Test)", x = "True ITE", y = "Estimated ITE") +
    theme_minimal() +
    theme(legend.position = "top")
  
  outcome_ITE_plot <- plot_outcome_ITE(data.dev.rs = model.results$data.dev.rs, data.val.rs = model.results$data.val.rs, x_lim = c(-0.9,0.9))
  
  
  # Define layout matrix: 3 columns x 2 rows
  layout_matrix <- rbind(
    c(1, 2, 5),
    c(3, 4, 5)
  )
  
  grid.arrange(
    p_dev_plot, p_val_plot,
    ite_dev_plot, ite_val_plot,
    outcome_ITE_plot,
    layout_matrix = layout_matrix,
    widths = c(1, 1, 1.3) 
  )
  
  if (ate_ite){
    # ATE as risk difference
    breaks <- round(quantile(model.results$data.dev.rs$ITE, probs = seq(0, 1, length.out = 7), na.rm = TRUE), 3)
    data.dev.grouped.ATE <- model.results$data.dev.rs %>% 
      mutate(ITE.Group = cut(ITE, breaks = breaks, include.lowest = T)) %>%
      dplyr::filter(!is.na(ITE.Group)) %>%
      group_by(ITE.Group) %>% 
      group_modify(~ calc.ATE.Risks(.x)) %>% ungroup()
    data.val.grouped.ATE <- model.results$data.val.rs %>% 
      mutate(ITE.Group = cut(ITE, breaks = breaks, include.lowest = T)) %>%
      dplyr::filter(!is.na(ITE.Group)) %>%
      group_by(ITE.Group) %>%
      group_modify(~ calc.ATE.Risks(.x)) %>% ungroup() 
    
    outcome_ATE_ITE_plot <- plot_ATE_ITE_in_group_risks(dev.data = data.dev.grouped.ATE, val.data = data.val.grouped.ATE)
    
    # Define layout matrix: 3 columns x 2 rows
    layout_matrix <- rbind(
      c(1, 2, 5),
      c(3, 4, 6)
    )
    
    grid.arrange(
      p_dev_plot, p_val_plot,
      ite_dev_plot, ite_val_plot,
      outcome_ATE_ITE_plot,
      outcome_ITE_plot,
      layout_matrix = layout_matrix,
      widths = c(1, 1, 1.3) 
    )
  }
}


check_ate <- function(model.results) {
  dev_ate_est <- mean(model.results$data.dev.rs$ITE)
  val_ate_est <- mean(model.results$data.val.rs$ITE)
  
  dev_ate_obs <- mean(model.results$data.dev.rs[model.results$data.dev.rs$Tr==1,]$Y) - 
    mean(model.results$data.dev.rs[model.results$data.dev.rs$Tr==0,]$Y)
  
  val_ate_obs <- mean(model.results$data.val.rs[model.results$data.val.rs$Tr==1,]$Y) -
    mean(model.results$data.val.rs[model.results$data.val.rs$Tr==0,]$Y)
  
  dev_ate_true <- mean(model.results$data.dev.rs$ITE_true)
  val_ate_true <- mean(model.results$data.val.rs$ITE_true)
  
  # RMSE of ITE
  dev_rmse <- sqrt(mean((model.results$data.dev.rs$ITE_true - model.results$data.dev.rs$ITE)^2))
  val_rmse <- sqrt(mean((model.results$data.val.rs$ITE_true - model.results$data.val.rs$ITE)^2))
  
  return(round(data.frame(
    ATE_Estimated = c(dev_ate_est, val_ate_est),
    ATE_Observed = c(dev_ate_obs, val_ate_obs),
    ATE_True = c(dev_ate_true, val_ate_true),
    RMSE = c(dev_rmse, val_rmse),
    row.names = c("Train (Risk Diff)", "Test (Risk Diff)")
  ),4))
}



#################################################
# Benchmark (GLM T-learner)
#################################################

# functions for fitting model and plotting results

fit.glm <- function(df) {
  p <- sum(grepl("^X", colnames(df$data.dev)))
  variable_names <- paste0("X", 1:p)
  form <- as.formula(paste("Y ~", paste(variable_names, collapse = " + ")))
  
  # Fit GLM for treatment and control groups
  fit.dev.tx <- glm(form, data = df$data.dev.tx, family = binomial(link = "logit"))
  fit.dev.ct <- glm(form, data = df$data.dev.ct, family = binomial(link = "logit"))
  
  # Predict outcome for observed T and X on derivation sample
  df$data.dev$Y_pred <- predict(fit.dev.tx, newdata = df$data.dev, type = "response") * 
    df$data.dev$Tr + 
    predict(fit.dev.ct, newdata = df$data.dev, type = "response") * 
    (1 - df$data.dev$Tr)
  
  # Predict outcome for observed T and X on validation sample
  df$data.val$Y_pred <- predict(fit.dev.tx, newdata = df$data.val, type = "response") * 
    df$data.val$Tr + 
    predict(fit.dev.ct, newdata = df$data.val, type = "response") * 
    (1 - df$data.val$Tr)
  
  # Predict ITE on derivation sample
  pred.data.dev <- df$data.dev %>% dplyr::select(variable_names)
  df$data.dev$Y_pred_tx <- predict(fit.dev.tx, newdata = pred.data.dev, type = "response") 
  df$data.dev$Y_pred_ct <- predict(fit.dev.ct, newdata = pred.data.dev, type = "response")
  pred.dev <- df$data.dev$Y_pred_tx - df$data.dev$Y_pred_ct 
  
  
  # Predict ITE on validation sample
  pred.data.val <- df$data.val %>% dplyr::select(variable_names)
  df$data.val$Y_pred_tx <- predict(fit.dev.tx, newdata = pred.data.val, type = "response")
  df$data.val$Y_pred_ct <- predict(fit.dev.ct, newdata = pred.data.val, type = "response")
  pred.val <- df$data.val$Y_pred_tx  - df$data.val$Y_pred_ct 
  
  # generate data
  data.dev.rs <- df$data.dev %>% 
    mutate(ITE = pred.dev, RS = ifelse(ITE < 0, "benefit", "harm")) %>%
    mutate(RS = as.factor(RS))
  
  data.val.rs <- df$data.val %>% 
    mutate(ITE = pred.val, RS = ifelse(ITE < 0, "benefit", "harm")) %>%
    mutate(RS = as.factor(RS))
  
  
  return(list(data.dev.rs = data.dev.rs, data.val.rs = data.val.rs, 
              model.dev.tx = fit.dev.tx, model.dev.ct = fit.dev.ct))
  
}


#### Model 1: no unnecessary variables, no confounder  ####

dgp_model1 <-dgp_simulation(n_obs=20000, SEED=123,
                beta_0 = 0.45, # intercept
                beta_t = -0.85, # main treatment effect
                beta_X = c(-0.5, 0.1), # main effects of Xs
                beta_TX = c(0.7), # interaction effects of Xs with treatment (here X1:T)
                p0 = 0,  # number of variables without effect
                confounder=FALSE,  # index of confounder (only single index specified)
                drop=FALSE) # e.g. drop = c("X1", "X2") these are dropped from the final dataset


glm.results1 <- fit.glm(dgp_model1$test.compl.data)


# plot the results
plot_pred_ite(glm.results1, ate_ite = FALSE) # ate_ite = TRUE adds the cATE plot 

check_ate(glm.results1)


#### Model 2: 4 unnecessary variables, no confounder  ####

dgp_model2 <-dgp_simulation(n_obs=20000, SEED=123,
                            beta_0 = 0.45,
                            beta_t = -0.85,
                            beta_X = c(-0.5, 0.1),
                            beta_TX = c(0.7),
                            p0 = 4, 
                            confounder=FALSE, 
                            drop=FALSE) 


glm.results2 <- fit.glm(dgp_model2$test.compl.data)


# plot the results
plot_pred_ite(glm.results2)

check_ate(glm.results2)


#### Model 3: 4 unnecessary variables, 1 confounder  ####

dgp_model3 <-dgp_simulation(n_obs=20000, SEED=123,
                            beta_0 = 0.45,
                            beta_t = -0.85,
                            beta_X = c(-0.5, 0.1),
                            beta_TX = c(0.7),
                            p0 = 4, 
                            confounder=1, 
                            drop=FALSE)  

glm.results3 <- fit.glm(dgp_model3$test.compl.data)


# plot the results
plot_pred_ite(glm.results3)

check_ate(glm.results3)


#### Model 4: 4 unnecessary variables, 1 confounder, small treatment effect  ####



dgp_model4 <-dgp_simulation(n_obs=20000, SEED=123,
                            beta_0 = 0.45,
                            beta_t = 0.02,        # small main effect
                            beta_X = c(-0.5, 0.1),
                            beta_TX = c(-0.01),   # small interaction effect
                            p0 = 4, 
                            confounder=1, 
                            drop=FALSE)  

glm.results4 <- fit.glm(dgp_model4$test.compl.data)


# plot the results
plot_pred_ite(glm.results4)

check_ate(glm.results4)



#### Model 5: 4 unnecessary variables, no confounder, small treatment effect  ####

dgp_model5 <-dgp_simulation(n_obs=20000, SEED=123,
                              beta_0 = 0.45,
                              beta_t = 0.02,        # small main effect
                              beta_X = c(-0.5, 0.1),
                              beta_TX = c(-0.01),   # small interaction effect
                              p0 = 4, 
                              confounder=FALSE, 
                              drop=FALSE)

glm.results5 <- fit.glm(dgp_model5$test.compl.data)


# plot the results
plot_pred_ite(glm.results5)

check_ate(glm.results5)


#### Model 6: 8 predictors, 4 unnecessary variables, no confounder, small treatment effect  ####

dgp_model6 <-dgp_simulation(n_obs=20000, SEED=123,
                            beta_0 = 0.45,
                            beta_t = 0.02,        # small main effect
                            beta_X = c(-0.5, 0.1, 0.2, -0.7, -0.15, 0.3, 0.05, -0.2),
                            beta_TX = c(-0.01, 0.04, 0.001),   # small interaction effect
                            p0 = 4, 
                            confounder=FALSE, 
                            drop=FALSE)

glm.results6 <- fit.glm(dgp_model6$test.compl.data)


# plot the results
plot_pred_ite(glm.results6)

check_ate(glm.results6)




#### Model 7: 8 predictors, 4 unnecessary variables, no confounder, small main effect, large interaction effect  ####

dgp_model7 <-dgp_simulation(n_obs=20000, SEED=123,
                            beta_0 = 0.45,
                            beta_t = 0.02,        # small main effect
                            beta_X = c(-0.5, 0.1, 0.2, -0.7, -0.15, 0.3, 0.05, -0.2),
                            beta_TX = c(-0.5, 0.04, 0.001),   # small interaction effect
                            p0 = 4, 
                            confounder=FALSE, 
                            drop=FALSE) 

glm.results7 <- fit.glm(dgp_model7$test.compl.data)


# plot the results
plot_pred_ite(glm.results7)

check_ate(glm.results7)





#### Model 8: 8 predictors, 4 unnecessary variables, no confounder, large main effect, small interaction effect  ####

dgp_model8 <-dgp_simulation(n_obs=20000, SEED=123,
                            beta_0 = 0.45,
                            beta_t = 0.7,        # small main effect
                            beta_X = c(-0.5, 0.1, 0.2, -0.7, -0.15, 0.3, 0.05, -0.2),
                            beta_TX = c(-0.005, 0.04, 0.001),   # small interaction effect
                            p0 = 4, 
                            confounder=FALSE, 
                            drop=FALSE)

glm.results8 <- fit.glm(dgp_model8$test.compl.data)


# plot the results
plot_pred_ite(glm.results8)

check_ate(glm.results8)


#### Model 9: 8 predictors, 4 unnecessary variables, no confounder, small main effect, large interaction effect


dgp_model9 <-dgp_simulation(n_obs=20000, SEED=123,
                            beta_0 = 0.45,
                            beta_t = 0.02,        # small main effect
                            beta_X = c(-0.5, 0.1, 0.2, -0.7, -0.15, 0.3, 0.05, -0.2),
                            beta_TX = c(-0.5, 0.04, 0.001),   # large interaction effect
                            p0 = 4, 
                            confounder=FALSE, 
                            drop=FALSE)

glm.results9 <- fit.glm(dgp_model9$test.compl.data)


# plot the results
plot_pred_ite(glm.results9, ate_ite = FALSE)

check_ate(glm.results9)



#### Model 10: 8 predictors, 4 unnecessary variables, no confounder, small main effect, large interaction effect but dropped  ####

# looks like real data of IST Trial! -> good on train data, but bad on test data


dgp_model10 <-dgp_simulation(n_obs=20000, SEED=123,
                            rho=0.1,
                            beta_0 = 0.45,
                            beta_t = 0.02,        # small main effect
                            beta_X = c(-0.5, 0.1, 0.2, -0.7, -0.15, 0.3, 0.05, -0.2),
                            beta_TX = c(-0.5, 0.04, 0.001),   # large interaction effect
                            p0 = 4, 
                            confounder=FALSE, 
                            drop=c("X1"))

glm.results10 <- fit.glm(dgp_model10$test.compl.data)


# plot the results
plot_pred_ite(glm.results10, ate_ite = TRUE)

check_ate(glm.results10)



#### Model 11: 8 predictors, 4 unnecessary variables, no confounder, small main effect, large interaction effect 
# dropped X4 which has no interaction effect

# --> seems like if you drop a variable with no treatment interaction effect, it does not matter for ITE estimation

dgp_model11 <-dgp_simulation(n_obs=20000, SEED=123,
                             rho=0.1,
                             beta_0 = 0.45,
                             beta_t = 0.02,        # small main effect
                             beta_X = c(-0.5, 0.1, 0.2, -0.7, -0.15, 0.3, 0.05, -0.2),
                             beta_TX = c(-0.5, 0.04, 0.001),   # small interaction effect
                             p0 = 4, 
                             confounder=FALSE, 
                             drop=c("X4")) 

glm.results11 <- fit.glm(dgp_model11$test.compl.data)


# plot the results
plot_pred_ite(glm.results11)

check_ate(glm.results11)



#### Model 12: 8 predictors, 4 unnecessary variables, no confounder, small main effect, large interaction effect but dropped  ####
#### rho (correlation) set to 0, compared to 0.1 before

# to compare with model 10, if correlation might be a reason for bad generalization

## --> compared to model 10, it doesn't genearalize worse anymore, but it is already
## quite bad in the train set. --> in next experiment, with rho=0.6, the 
## result is good on both the train and test set...


dgp_model12 <-dgp_simulation(n_obs=20000, SEED=123,
                             rho = 0,
                             beta_0 = 0.45,
                             beta_t = 0.02,        # small main effect
                             beta_X = c(-0.5, 0.1, 0.2, -0.7, -0.15, 0.3, 0.05, -0.2),
                             beta_TX = c(-0.5, 0.04, 0.001),   # large interaction effect
                             p0 = 4, 
                             confounder=FALSE, 
                             drop=c("X1"))

glm.results12 <- fit.glm(dgp_model12$test.compl.data)


# plot the results
plot_pred_ite(glm.results12)

check_ate(glm.results12)






#### Model 13: 8 predictors, 4 unnecessary variables, no confounder, small main effect, large interaction effect but dropped  ####
#### (high correlation) set to 0.6, compared to 0.1 before


dgp_model13 <-dgp_simulation(n_obs=20000, SEED=123,
                             rho = 0.6,
                             beta_0 = 0.45,
                             beta_t = 0.02,        # small main effect
                             beta_X = c(-0.5, 0.1, 0.2, -0.7, -0.15, 0.3, 0.05, -0.2),
                             beta_TX = c(-0.5, 0.04, 0.001),   # large interaction effect
                             p0 = 4, 
                             confounder=FALSE, 
                             drop=c("X1"))

glm.results13 <- fit.glm(dgp_model13$test.compl.data)


# plot the results
plot_pred_ite(glm.results13)

check_ate(glm.results13)





#### Model 14: to play around ####


dgp_model14 <-dgp_simulation(n_obs=20000, SEED=123,
                             rho = 0.1,
                             beta_0 = 0.45,
                             beta_t = 0.01,        # small main effect
                             beta_X = c(-0.5, 0.1, 0.2, -0.7, -0.15, 0.3, 0.05, -0.2),
                             beta_TX = c(-0.8, -0.7, 0.001),   # large interaction effect
                             p0 = 4, 
                             confounder=FALSE, 
                             drop=FALSE)  # c("X1")  FALSE c("X1", "X2")

glm.results14 <- fit.glm(dgp_model14$test.compl.data)


# plot the results
plot_pred_ite(glm.results14)

check_ate(glm.results14)






#################################################
# glmnet T-learner (lasso regression)
#################################################


### MODEL WITH LASSO
library(glmnet)
fit.glmnet <- function(df) {
  # Extract predictor matrix (X) and response (Y)
  X_vars <- grep("^X", names(df$data.dev), value = TRUE)
  
  # Training data for treated and control
  X_tx <- as.matrix(df$data.dev.tx[, X_vars])
  Y_tx <- df$data.dev.tx$Y
  
  X_ct <- as.matrix(df$data.dev.ct[, X_vars])
  Y_ct <- df$data.dev.ct$Y
  
  # Fit Lasso with cross-validation
  cv_tx <- cv.glmnet(X_tx, Y_tx, family = "binomial", alpha = 1)
  cv_ct <- cv.glmnet(X_ct, Y_ct, family = "binomial", alpha = 1)
  
  # Final models
  fit.dev.tx <- glmnet(X_tx, Y_tx, family = "binomial", lambda = cv_tx$lambda.min)
  fit.dev.ct <- glmnet(X_ct, Y_ct, family = "binomial", lambda = cv_ct$lambda.min)
  
  # Prediction on dev data
  X_dev <- as.matrix(df$data.dev[, X_vars])
  df$data.dev$Y_pred <- predict(fit.dev.tx, newx = X_dev, type = "response") * df$data.dev$Tr +
    predict(fit.dev.ct, newx = X_dev, type = "response") * (1 - df$data.dev$Tr)
  
  # Prediction on val data
  X_val <- as.matrix(df$data.val[, X_vars])
  df$data.val$Y_pred <- predict(fit.dev.tx, newx = X_val, type = "response") * df$data.val$Tr +
    predict(fit.dev.ct, newx = X_val, type = "response") * (1 - df$data.val$Tr)
  
  # ITE prediction on dev
  df$data.dev$Y_pred_tx <- predict(fit.dev.tx, newx = X_dev, type = "response")
  df$data.dev$Y_pred_ct <- predict(fit.dev.ct, newx = X_dev, type = "response")
  pred.dev <- df$data.dev$Y_pred_tx - df$data.dev$Y_pred_ct
  
  # ITE prediction on val
  df$data.val$Y_pred_tx <- predict(fit.dev.tx, newx = X_val, type = "response")
  df$data.val$Y_pred_ct <- predict(fit.dev.ct, newx = X_val, type = "response")
  pred.val <- df$data.val$Y_pred_tx - df$data.val$Y_pred_ct
  
  # Generate RS labels
  data.dev.rs <- df$data.dev %>%
    mutate(ITE = pred.dev, RS = ifelse(ITE < 0, "benefit", "harm")) %>%
    mutate(RS = as.factor(RS))
  
  data.val.rs <- df$data.val %>%
    mutate(ITE = pred.val, RS = ifelse(ITE < 0, "benefit", "harm")) %>%
    mutate(RS = as.factor(RS))
  
  return(list(
    data.dev.rs = data.dev.rs,
    data.val.rs = data.val.rs,
    model.dev.tx = fit.dev.tx,
    model.dev.ct = fit.dev.ct
  ))
}


#### Model 1: 8 predictors, 4 unnecessary variables, no confounder, small main effect, large interaction effect


dgp_model1 <-dgp_simulation(n_obs=20000, SEED=123,
                            beta_0 = 0.45,
                            beta_t = 0.02,        # small main effect
                            beta_X = c(-0.5, 0.1, 0.2, -0.7, -0.15, 0.3, 0.05, -0.2),
                            beta_TX = c(-0.5, 0.04, 0.001),   # large interaction effect
                            p0 = 4, 
                            confounder=FALSE, 
                            drop=FALSE) 

glmnet.results1 <- fit.glmnet(dgp_model1$test.compl.data)

# plot the results
plot_pred_ite(glmnet.results1)

check_ate(glmnet.results1)


#### Model 2: 8 predictors, 4 unnecessary variables, no confounder, small main effect, large interaction effect
# drop interaction

dgp_model2 <-dgp_simulation(n_obs=20000, SEED=123,
                            beta_0 = 0.45,
                            beta_t = 0.02,        # small main effect
                            beta_X = c(-0.5, 0.1, 0.2, -0.7, -0.15, 0.3, 0.05, -0.2),
                            beta_TX = c(-0.5, 0.04, 0.001),   # large interaction effect
                            p0 = 4, 
                            confounder=FALSE, 
                            drop=c("X1")) 

glmnet.results2 <- fit.glmnet(dgp_model2$test.compl.data)


# plot the results
plot_pred_ite(glmnet.results2)

check_ate(glmnet.results2)




#################################################
# glmnet S-learner (lasso regression with all interactions)
#################################################


#### single model lasso regression with all interactions(S-learner)
fit.glmnet.slearner <- function(df) {
  # df <- glmnet.slearner.results1 # debugging
  
  # Extract variable names
  X_vars <- grep("^X", names(df$data.dev), value = TRUE)
  Tr <- df$data.dev$Tr
  Y <- df$data.dev$Y
  
  # Build interaction terms manually
  X_main <- as.matrix(df$data.dev[, X_vars])
  X_interactions <- X_main * Tr  # element-wise multiplication for interactions
  colnames(X_interactions) <- paste0(X_vars, "_Tr")
  
  # Combine into one design matrix: Xs + treatment + interactions
  X_all <- cbind(X_main, Tr = Tr, X_interactions)
  
  # Fit Lasso-penalized logistic regression with cross-validation
  cv_fit <- cv.glmnet(X_all, Y, family = "binomial", alpha = 1)
  fit <- glmnet(X_all, Y, family = "binomial", lambda = cv_fit$lambda.min)
  
  
  # Predict with treatment = 1 on derivation data
  X_dev_main <- as.matrix(df$data.dev[, X_vars])
  X_dev_tx <- cbind(
    X_dev_main,
    Tr = 1,
    X_dev_main * 1
  )
  colnames(X_dev_tx) <- colnames(X_all)
  
  # Predict with treatment = 0 on derivation data
  X_dev_ct <- cbind(
    X_dev_main,
    Tr = 0,
    X_dev_main * 0
  )
  colnames(X_dev_ct) <- colnames(X_all)
  
  pred_dev_tx <- predict(fit, newx = X_dev_tx, type = "response")
  pred_dev_ct <- predict(fit, newx = X_dev_ct, type = "response")
  pred_dev <- pred_dev_tx - pred_dev_ct
  
  
  
  # Prepare validation data for prediction
  X_val_main <- as.matrix(df$data.val[, X_vars])
  
  # Predict with treatment = 1
  X_val_tx <- cbind(
    X_val_main,
    Tr = 1,
    X_val_main * 1
  )
  colnames(X_val_tx) <- colnames(X_all)
  
  # Predict with treatment = 0
  X_val_ct <- cbind(
    X_val_main,
    Tr = 0,
    X_val_main * 0
  )
  colnames(X_val_ct) <- colnames(X_all)
  
  # Predict ITE on validation data
  pred_val_tx <- predict(fit, newx = X_val_tx, type = "response")
  pred_val_ct <- predict(fit, newx = X_val_ct, type = "response")
  pred_val <- pred_val_tx - pred_val_ct
  
  
  
  # Predict observed outcome for derivation sample
  df$data.dev$Y_pred <- df$data.dev$Tr * pred_dev_tx + (1 - df$data.dev$Tr) * pred_dev_ct
  df$data.dev$Y_pred_tx <- pred_dev_tx
  df$data.dev$Y_pred_ct <- pred_dev_ct
  
  # Predict observed outcome for validation sample
  df$data.val$Y_pred <- df$data.val$Tr * pred_val_tx + (1 - df$data.val$Tr) * pred_val_ct
  df$data.val$Y_pred_tx <- pred_val_tx
  df$data.val$Y_pred_ct <- pred_val_ct
  
  # Generate RS labels
  data.dev.rs <- df$data.dev %>%
    mutate(ITE = pred_dev, RS = ifelse(ITE < 0, "benefit", "harm")) %>%
    mutate(RS = as.factor(RS))
  
  data.val.rs <- df$data.val %>%
    mutate(ITE = pred_val, RS = ifelse(ITE < 0, "benefit", "harm")) %>%
    mutate(RS = as.factor(RS))
  
  return(list(
    data.dev.rs = data.dev.rs,
    data.val.rs = data.val.rs,
    model = fit
  ))
}




dgp_model1 <-dgp_simulation(n_obs=20000, SEED=123,
                            beta_0 = 0.45,
                            beta_t = 0.02,        # small main effect
                            beta_X = c(-0.5, 0.1, 0.2, -0.7, -0.15, 0.3, 0.05, -0.2),
                            beta_TX = c(-0.5, 0.04, 0.001),   # large interaction effect
                            p0 = 4, 
                            confounder=FALSE, 
                            drop=FALSE) 

glmnet.slearner.results1 <- fit.glmnet.slearner(dgp_model1$test.compl.data)

# plot the results
plot_pred_ite(glmnet.slearner.results1)

check_ate(glmnet.slearner.results1)




#### Model 2: 8 predictors, 4 unnecessary variables, no confounder, small main effect, large interaction effect
# drop interaction

# same issue as with other models, dropping interaction variable causes problem with generalization

dgp_model2 <-dgp_simulation(n_obs=20000, SEED=123,
                            beta_0 = 0.45,
                            beta_t = 0.02,        # small main effect
                            beta_X = c(-0.5, 0.1, 0.2, -0.7, -0.15, 0.3, 0.05, -0.2),
                            beta_TX = c(-0.5, 0.04, 0.001),   # large interaction effect
                            p0 = 4, 
                            confounder=FALSE, 
                            drop=c("X1"))

glmnet.slearner.results2 <- fit.glmnet.slearner(dgp_model2$test.compl.data)


# plot the results
plot_pred_ite(glmnet.slearner.results2)

check_ate(glmnet.slearner.results2)










#################################################
# Complex Model (randomForest)
#################################################


library(randomForest)
library(dplyr)

fit.rf <- function(df, ntrees = 100) {
  p <- sum(grepl("^X", colnames(df$data.dev)))
  variable_names <- paste0("X", 1:p)
  form <- as.formula(paste("Y ~", paste(variable_names, collapse = " + ")))
  
  df$data.dev.tx$Y <- as.factor(df$data.dev.tx$Y)  # Ensure Y is a factor for classification
  df$data.dev.ct$Y <- as.factor(df$data.dev.ct$Y)  # Ensure Y is a factor for classification
  
  # Fit random forest for treatment and control groups
  fit.dev.tx <- randomForest(form, data = df$data.dev.tx, ntree = ntrees)
  fit.dev.ct <- randomForest(form, data = df$data.dev.ct, ntree = ntrees)
  
  # Predict outcome for observed T and X on derivation sample
  df$data.dev$Y_pred <- predict(fit.dev.tx, newdata = df$data.dev, type="prob")[,2] * df$data.dev$Tr +
    predict(fit.dev.ct, newdata = df$data.dev, type="prob")[,2] * (1 - df$data.dev$Tr)
  
  # Predict outcome for observed T and X on validation sample
  df$data.val$Y_pred <- predict(fit.dev.tx, newdata = df$data.val, type="prob")[,2] * df$data.val$Tr +
    predict(fit.dev.ct, newdata = df$data.val, type="prob")[,2] * (1 - df$data.val$Tr)
  
  # Predict ITE on derivation sample
  pred.data.dev <- df$data.dev %>% dplyr::select(variable_names)
  df$data.dev$Y_pred_tx <- predict(fit.dev.tx, newdata = pred.data.dev, type="prob")[,2]
  df$data.dev$Y_pred_ct <- predict(fit.dev.ct, newdata = pred.data.dev, type="prob")[,2]
  pred.dev <- df$data.dev$Y_pred_tx - df$data.dev$Y_pred_ct
  
  # Predict ITE on validation sample
  pred.data.val <- df$data.val %>% dplyr::select(variable_names)
  df$data.val$Y_pred_tx <- predict(fit.dev.tx, newdata = pred.data.val, type="prob")[,2]
  df$data.val$Y_pred_ct <- predict(fit.dev.ct, newdata = pred.data.val, type="prob")[,2]
  pred.val <- df$data.val$Y_pred_tx - df$data.val$Y_pred_ct
  
  
  
  
  # check binary predictions on the train set
  train_y_pred_tx <- predict(fit.dev.tx, newdata = df$data.dev.tx, type="response")
  train_y_pred_ct <- predict(fit.dev.ct, newdata = df$data.dev.ct, type="response")
  
  mean(df$data.dev.tx$Y == train_y_pred_tx)
  mean(df$data.dev.ct$Y == train_y_pred_ct)
  # combined accuracy (train)
  acc_train <- mean(c(df$data.dev.tx$Y == train_y_pred_tx, df$data.dev.ct$Y == train_y_pred_ct))
  
  
  # check binary predictions on the validation set
  val_y_pred_tx <- predict(fit.dev.tx, newdata = df$data.val.tx, type="response")
  val_y_pred_ct <- predict(fit.dev.ct, newdata = df$data.val.ct, type="response")
  
  mean(df$data.val.tx$Y == val_y_pred_tx)
  mean(df$data.val.ct$Y == val_y_pred_ct)
  
  # combined accuracy (validation)
  acc_test <- mean(c(df$data.val.tx$Y == val_y_pred_tx, df$data.val.ct$Y == val_y_pred_ct))
  
  
  
  
  # Generate result sets
  data.dev.rs <- df$data.dev %>%
    mutate(ITE = pred.dev, RS = ifelse(ITE < 0, "benefit", "harm")) %>%
    mutate(RS = as.factor(RS))
  
  data.val.rs <- df$data.val %>%
    mutate(ITE = pred.val, RS = ifelse(ITE < 0, "benefit", "harm")) %>%
    mutate(RS = as.factor(RS))
  
  # Print accuracy directly inside the function
  cat(paste0("Train Accuracy: ", round(acc_train, 3), 
             ", Test Accuracy: ", round(acc_test, 3), "\n"))
  
  
  return(list(data.dev.rs = data.dev.rs, data.val.rs = data.val.rs,
              model.dev.tx = fit.dev.tx, model.dev.ct = fit.dev.ct))
}

dgp_rf <-dgp_simulation(n_obs=20000, SEED=123,
                            beta_0 = 0.45,
                            beta_t = -0.85,
                            beta_X = c(-0.5, 0.8),
                            beta_TX = c(0.7),
                            p0 = 0, 
                            confounder=FALSE, 
                            drop=FALSE)  


rf.results <- fit.rf(dgp_rf$test.compl.data, ntrees = 100)

# plot the results
plot_pred_ite(rf.results)

# glm to compare for same dgp
res.glm <- fit.glm(df)
plot_pred_ite(res.glm)




dgp_rf <-dgp_simulation(n_obs=20000, SEED=123,
                        beta_0 = 0.45,
                        beta_t = -0.85,
                        beta_X = c(-0.5, 0.8, 0.3, 0.6, -0.9),
                        beta_TX = c(0.7, 0.3, -0.3),
                        p0 = 0, 
                        confounder=FALSE, 
                        drop=FALSE)  



rf.results <- fit.rf(dgp_rf$test.compl.data, ntrees = 100)

# plot the results
plot_pred_ite(rf.results)





#################################################
# Complex Model (Random Forest comets package, tuned)
#################################################


library(comets)
library(dplyr)

# extract the tuned_rf function
comets_tuned_rf <- comets:::tuned_rf

fit.tuned_rf <- function(df) {
  p <- sum(grepl("^X", colnames(df$data.dev)))
  variable_names <- paste0("X", 1:p)
  form <- as.formula(paste("Y ~", paste(variable_names, collapse = " + ")))
  
  df$data.dev.tx$Y <- as.factor(df$data.dev.tx$Y)  # Ensure Y is a factor for classification
  df$data.dev.ct$Y <- as.factor(df$data.dev.ct$Y)  # Ensure Y is a factor for classification
  
  # Fit random forest for treatment and control groups
  fit.dev.tx <- comets_tuned_rf(y=as.matrix(df$data.dev.tx$Y), x=as.matrix(df$data.dev.tx %>% dplyr::select(variable_names)))
  fit.dev.ct <- comets_tuned_rf(y=as.matrix(df$data.dev.ct$Y), x=as.matrix(df$data.dev.ct %>% dplyr::select(variable_names)))
  
  # Feature set of derivation sample
  X_dev <- as.matrix(df$data.dev %>% dplyr::select(variable_names))
  
  # Predict probabilities on derivation sample
  pred_tx_dev <- predict(fit.dev.tx, data = X_dev)
  pred_ct_dev <- predict(fit.dev.ct, data = X_dev)
  
  # Predict outcome for observed T and X on derivation sample
  df$data.dev$Y_pred <- pred_tx_dev * df$data.dev$Tr + pred_ct_dev * (1 - df$data.dev$Tr)
  
  # Predict ITE on derivation sample
  df$data.dev$Y_pred_tx <- pred_tx_dev
  df$data.dev$Y_pred_ct <- pred_ct_dev
  pred.dev <- df$data.dev$Y_pred_tx - df$data.dev$Y_pred_ct
  
  
  # Feature set of validation sample
  X_val <- as.matrix(df$data.val %>% dplyr::select(variable_names))
  
  # Predict probabilities on derivation sample
  pred_tx_val <- predict(fit.dev.tx, data = X_val)
  pred_ct_val <- predict(fit.dev.ct, data = X_val)
  
  # Predict outcome for observed T and X on validation sample
  df$data.val$Y_pred <- pred_tx_val * df$data.val$Tr + pred_ct_val * (1 - df$data.val$Tr)
  
  # Predict ITE on validation sample
  df$data.val$Y_pred_tx <- pred_tx_val
  df$data.val$Y_pred_ct <- pred_ct_val
  pred.val <- df$data.val$Y_pred_tx - df$data.val$Y_pred_ct
  
  
  
  
  # check binary predictions on the train set
  # train_y_pred_tx <- predict(fit.dev.tx, newdata = df$data.dev.tx, type="response")
  # train_y_pred_ct <- predict(fit.dev.ct, newdata = df$data.dev.ct, type="response")
  
  # mean(df$data.dev.tx$Y == train_y_pred_tx)
  # mean(df$data.dev.ct$Y == train_y_pred_ct)
  # # combined accuracy (train)
  # acc_train <- mean(c(df$data.dev.tx$Y == train_y_pred_tx, df$data.dev.ct$Y == train_y_pred_ct))
  
  
  # check binary predictions on the validation set
  # val_y_pred_tx <- predict(fit.dev.tx, newdata = df$data.val.tx, type="response")
  # val_y_pred_ct <- predict(fit.dev.ct, newdata = df$data.val.ct, type="response")
  # 
  # mean(df$data.val.tx$Y == val_y_pred_tx)
  # mean(df$data.val.ct$Y == val_y_pred_ct)
  # 
  # combined accuracy (validation)
  # acc_test <- mean(c(df$data.val.tx$Y == val_y_pred_tx, df$data.val.ct$Y == val_y_pred_ct))
  
  
  
  
  # Generate result sets
  data.dev.rs <- df$data.dev %>%
    mutate(ITE = pred.dev, RS = ifelse(ITE < 0, "benefit", "harm")) %>%
    mutate(RS = as.factor(RS))
  
  data.val.rs <- df$data.val %>%
    mutate(ITE = pred.val, RS = ifelse(ITE < 0, "benefit", "harm")) %>%
    mutate(RS = as.factor(RS))
  
  # Print accuracy directly inside the function
  # cat(paste0("Train Accuracy: ", round(acc_train, 3), 
  #            ", Test Accuracy: ", round(acc_test, 3), "\n"))
  # 
  
  return(list(data.dev.rs = data.dev.rs, data.val.rs = data.val.rs,
              model.dev.tx = fit.dev.tx, model.dev.ct = fit.dev.ct))
}




# case 1: # 2 predictors, 1 interaction, large effect

dgp_tuned_rf1 <-dgp_simulation(n_obs=20000, SEED=123,
                        beta_0 = 0.45,
                        beta_t = -0.85,
                        beta_X = c(-0.5, 0.8),
                        beta_TX = c(0.7),
                        p0 = 0, 
                        confounder=FALSE, 
                        drop=FALSE)  

# plot the results
plot_pred_ite(fit.tuned_rf(dgp_tuned_rf1$test.compl.data))
check_ate(fit.tuned_rf(dgp_tuned_rf1$test.compl.data))

# case 2: # 5 predictors, 1 interaction, large effect

dgp_tuned_rf2 <-dgp_simulation(n_obs=20000, SEED=123,
                              beta_0 = 0.45,
                              beta_t = -0.85,
                              beta_X = c(-0.5, 0.8, 0.2, 0.6, -0.4),
                              beta_TX = c(0.7),
                              p0 = 0, 
                              confounder=FALSE, 
                              drop=FALSE)  

# plot the results
plot_pred_ite(fit.tuned_rf(dgp_tuned_rf2$test.compl.data))



# case 3: # 5 predictors, 2 interactions, large effect

dgp_tuned_rf3 <-dgp_simulation(n_obs=20000, SEED=123,
                              beta_0 = 0.45,
                              beta_t = -0.85,
                              beta_X = c(-0.5, 0.8, 0.2, 0.6, -0.4),
                              beta_TX = c(0.7, 0.3),
                              p0 = 0, 
                              confounder=FALSE, 
                              drop=FALSE)  
# plot the results
plot_pred_ite(fit.tuned_rf(dgp_tuned_rf3$test.compl.data))



# case 4: # 5 predictors, 4 interactions, large effect

dgp_tuned_rf4 <-dgp_simulation(n_obs=20000, SEED=123,
                               beta_0 = 0.45,
                               beta_t = -0.85,
                               beta_X = c(-0.5, 0.8, 0.2, 0.6, -0.4),
                               beta_TX = c(0.7, 0.3, 0.1, -0.6),
                               p0 = 0, 
                               confounder=FALSE, 
                               drop=FALSE)  
# plot the results
plot_pred_ite(fit.tuned_rf(dgp_tuned_rf4$test.compl.data))




# case 5: # 2 predictors, 1 interaction, large effect, 3 unnecessary variables

dgp_tuned_rf5 <-dgp_simulation(n_obs=20000, SEED=123,
                               beta_0 = 0.45,
                               beta_t = -0.85,
                               beta_X = c(-0.5, 0.8),
                               beta_TX = c(0.7),
                               p0 = 3, 
                               confounder=FALSE, 
                               drop=FALSE)  

# plot the results
plot_pred_ite(fit.tuned_rf(dgp_tuned_rf5$test.compl.data))



#case 6: # 2 predictors, 1 interaction, small main effect


dgp_tuned_rf6 <-dgp_simulation(n_obs=20000, SEED=123,
                               beta_0 = 0.45,
                               beta_t = -0.05,
                               beta_X = c(-0.5, 0.8),
                               beta_TX = c(0.7),
                               p0 = 3, 
                               confounder=FALSE, 
                               drop=FALSE)  

# plot the results
plot_pred_ite(fit.tuned_rf(dgp_tuned_rf6$test.compl.data))



#case 7: # 2 predictors, 1 interaction, small interaction effect


dgp_tuned_rf7 <-dgp_simulation(n_obs=20000, SEED=123,
                               beta_0 = 0.45,
                               beta_t = -0.85,
                               beta_X = c(-0.5, 0.8),
                               beta_TX = c(0.03),
                               p0 = 3, 
                               confounder=FALSE, 
                               drop=FALSE)  

# plot the results
plot_pred_ite(fit.tuned_rf(dgp_tuned_rf7$test.compl.data))



#case 8: # 2 predictors, 1 interaction, small main + interaction effect


dgp_tuned_rf8 <-dgp_simulation(n_obs=20000, SEED=123,
                               beta_0 = 0.45,
                               beta_t = -0.05,
                               beta_X = c(-0.5, 0.8),
                               beta_TX = c(0.03),
                               p0 = 3, 
                               confounder=FALSE, 
                               drop=FALSE)

# plot the results
plot_pred_ite(fit.tuned_rf(dgp_tuned_rf8$test.compl.data))



#case 9: # 7 predictors, 1 interaction, large effect


dgp_tuned_rf9 <-dgp_simulation(n_obs=20000, SEED=123,
                               beta_0 = 0.45,
                               beta_t = -0.85,
                               beta_X = c(-0.5, 0.8, 0.3, 0.3, -0.7, 0.1, 0.2),
                               beta_TX = c(0.7),
                               p0 = 0, 
                               confounder=FALSE, 
                               drop=FALSE)

# plot the results
plot_pred_ite(fit.tuned_rf(dgp_tuned_rf9$test.compl.data))



#case 10: # 7 predictors, 1 interaction, large effect, drop X2, X4 and X5

dgp_tuned_rf10 <-dgp_simulation(n_obs=20000, SEED=123,
                               beta_0 = 0.45,
                               beta_t = -0.85,
                               beta_X = c(-0.5, 0.8, 0.3, 0.3, -0.7, 0.1, 0.2),
                               beta_TX = c(0.7),
                               p0 = 0, 
                               confounder=FALSE, 
                               drop= c("X2", "X4", "X5"))

# plot the results
plot_pred_ite(fit.tuned_rf(dgp_tuned_rf10$test.compl.data))





#### Model 11: 8 predictors, 4 unnecessary variables, no confounder, small main effect, large interaction effect but dropped  ####

# looks like real data of IST Trial! very strong discrimination in train but not in test

dgp_model11 <-dgp_simulation(n_obs=20000, SEED=123,
                            beta_0 = 0.45,                    # intercept
                            beta_t = 0.02,                    # small main effect
                            beta_X = c(-0.5, 0.1, 0.2, -0.7, -0.15, 0.3, 0.05, -0.2), # predictor betas X1-X8
                            beta_TX = c(-0.5, 0.04, 0.001),   # interaction betas X1 large, X2 and X3 small
                            p0 = 4,                           # 4 unnecessary variables
                            confounder=FALSE,                 # no confounder
                            drop=c("X1"))                     # use X1 for DGP but drop from Dataset (unobserved)

tuned_rf.results11 <- fit.tuned_rf(dgp_model11$test.compl.data)


# plot the results
plot_pred_ite(tuned_rf.results11, ate_ite = TRUE)

check_ate(tuned_rf.results11)




#### Model 12: 8 predictors, 4 unnecessary variables, no confounder, small main effect, large interaction effect but dropped  ####
#### rho (correlation) set to 0, compared to 0.1 before

# to compare with model 11, if correlation might be a reason for bad generalization



dgp_model12 <-dgp_simulation(n_obs=20000, SEED=123,
                             rho = 0,
                             beta_0 = 0.45,
                             beta_t = 0.02,        # small main effect
                             beta_X = c(-0.5, 0.1, 0.2, -0.7, -0.15, 0.3, 0.05, -0.2),
                             beta_TX = c(-0.5, 0.04, 0.001),   # large interaction effect
                             p0 = 4, 
                             confounder=FALSE, 
                             drop=c("X1"))  

tuned_rf.results12 <- fit.tuned_rf(dgp_model12$test.compl.data)


# plot the results
plot_pred_ite(tuned_rf.results12)

check_ate(tuned_rf.results12)




#### Model 13: 8 predictors, 4 unnecessary variables, no confounder, small main effect, large interaction effect ####
#### rho (correlation) set to 0, compared to 0.1 before
#### but dropped X1, X2, X3




dgp_model13 <-dgp_simulation(n_obs=20000, SEED=123,
                             rho = 0,
                             beta_0 = 0.45,
                             beta_t = 0.02,        # small main effect
                             beta_X = c(-0.5, 0.1, 0.2, -0.7, -0.15, 0.3, 0.05, -0.2),
                             beta_TX = c(-0.5, 0.04, 0.001),   # large interaction effect
                             p0 = 4, 
                             confounder=FALSE, 
                             drop=c("X1", "X2", "X3"))

tuned_rf.results13 <- fit.tuned_rf(dgp_model13$test.compl.data)


# plot the results
plot_pred_ite(tuned_rf.results13)

check_ate(tuned_rf.results13)






#### Model 14: 8 predictors, 4 unnecessary variables, no confounder, small main effect, large interaction effect ####
#### rho (correlation) set to 0, compared to 0.1 before
# to compare: when we drop no variables:

dgp_model14 <-dgp_simulation(n_obs=20000, SEED=123,
                             rho = 0,
                             beta_0 = 0.45,                    # intercept
                             beta_t = 0.02,                    # small main effect
                             beta_X = c(-0.5, 0.1, 0.2, -0.7, -0.15, 0.3, 0.05, -0.2), # predictor betas X1-X8
                             beta_TX = c(-0.5, 0.04, 0.001),   # interaction betas X1 large, X2 and X3 small
                             p0 = 4,                           # 4 unnecessary variables
                             confounder=FALSE,                 # no confounder
                             drop=FALSE)                     # use X1 for DGP but drop from Dataset (unobserved)

tuned_rf.results14 <- fit.tuned_rf(dgp_model14$test.compl.data)


# plot the results
plot_pred_ite(tuned_rf.results14)

check_ate(tuned_rf.results14)





#################################################
# Analysis as Holly T-learner GLM
#################################################


dgp_data$simulated_full_data %>% ggplot(aes(x=ITE_true)) +
  geom_density(color="gray8",fill="skyblue",alpha=.6) + 
  theme_minimal() + 
  xlab("True ITE") + 
  geom_vline(xintercept = 0, linetype="dashed") + coord_cartesian(xlim = c(-.8,.6))

str(dgp_data$simulated_data)

# average treatment effects
mean(dgp_data$simulated_full_data$ITE_true)

# percentage of patients with Y=1
mean(dgp_data$simulated_data$Y)


# Calculate ITE with logistic T-learner
test.results <- logis.ITE(dgp_data$test.compl.data , p=2)


data.dev.rs = test.results[["data.dev.rs"]] %>%  as.data.frame()
data.val.rs = test.results[["data.val.rs"]] %>%  as.data.frame()

library(ggpubr)
plot_outcome_ITE(data.dev.rs = data.dev.rs, data.val.rs = data.val.rs, x_lim = c(-0.8,0.8))


plot_ITE_density(test.results = test.results, true.data = dgp_data$simulated_full_data)


plot_ITE_density_tx_ct(data = data.dev.rs)
plot_ITE_density_tx_ct(data = data.val.rs)

par(mfrow=c(1,2))
plot(ITE ~ ITE_true, data = data.dev.rs, col = "orange", pch = 19, cex = 0.5
     , main = "Training Data")
abline(0,1)
plot(ITE ~ ITE_true, data = data.val.rs, col = "#36648B", pch = 19, cex = 0.5
     , main = "Test Data")
abline(0,1)


ggplot(data.dev.rs, aes(x=ITE_true, y=ITE, color=Treatment)) +
  geom_point() +
  geom_abline(slope = 1, intercept = 0, color = "red") +
  labs(title = "Training Data", x = "True ITE", y = "Estimated ITE") +
  theme_minimal() +
  theme(legend.position = "top")

ggplot(data.val.rs, aes(x=ITE_true, y=ITE, color=Treatment)) +
  geom_point() +
  geom_abline(slope = 1, intercept = 0, color = "red") +
  labs(title = "Test Data", x = "True ITE", y = "Estimated ITE") +
  theme_minimal() +
  theme(legend.position = "top")



# compare estimated ITE and true ITE
mean(abs(data.dev.rs$ITE_true - data.dev.rs$ITE))
mean(abs(data.val.rs$ITE_true - data.val.rs$ITE))



breaks <- c(-0.75, -0.4, -0.2, 0.1, 0.5)
log.odds <- F
data.dev.grouped.ATE <- data.dev.rs %>% 
  mutate(ITE.Group = cut(ITE, breaks = breaks, include.lowest = T)) %>%
  dplyr::filter(!is.na(ITE.Group)) %>%
  group_by(ITE.Group) %>% 
  group_modify(~ calc.ATE.Odds(.x, log.odds = log.odds)) %>% ungroup()
data.val.grouped.ATE <- data.val.rs %>% 
  mutate(ITE.Group = cut(ITE, breaks = breaks, include.lowest = T)) %>%
  dplyr::filter(!is.na(ITE.Group)) %>%
  group_by(ITE.Group) %>%
  group_modify(~ calc.ATE.Odds(.x, log.odds = log.odds)) %>% ungroup() 

plot_ATE_ITE_in_group(dev.data = data.dev.grouped.ATE, val.data = data.val.grouped.ATE, 
                      log.odds = log.odds, ylb = 0, yub = 4,
                      train.data.name = "Train", test.data.name = "Test")



# average treatment effects
mean(dgp_data$simulated_full_data$ITE_true)

mean(data.dev.rs$ITE_true)
mean(data.val.rs$ITE_true)


mean(data.dev.rs$ITE)
mean(data.val.rs$ITE)

# calcualte ATE
df <- dgp_data$simulated_full_data
mean(df$Y[df$Treatment == 1]) - mean(df$Y[df$Treatment == 0])





# save results for later
test.results.glm <- logis.ITE(dgp_data$test.compl.data , p=2)


data.dev.rs.glm = test.results[["data.dev.rs"]] %>%  as.data.frame()
data.val.rs.glm = test.results[["data.val.rs"]] %>%  as.data.frame()







