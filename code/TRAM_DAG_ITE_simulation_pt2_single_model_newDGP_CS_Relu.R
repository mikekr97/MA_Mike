##### When starting a new R Session ####
if (FALSE){
  reticulate::use_python("C:/ProgramData/Anaconda3/python.exe", required = TRUE)
}
# Get command-line arguments
args <- commandArgs(trailingOnly = TRUE)
if (length(args) == 0) {
  args <- c(3, 'cs')
  args <- c(1, 'ls')
}
F32 <- as.numeric(args[1])
M32 <- args[2]
print(paste("FS:", F32, "M32:", M32))


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
DIR = 'runs/TRAM_DAG_ITE_simulation_pt2_single_model_newDGP_CS_Relu/run'
# DIR = 'runs/TRAM_DAG_ITE_simulation_pt2_single_model_newDGP_CS_Relu_regularized/run'
if (!dir.exists(DIR)) {
  dir.create(DIR, recursive = TRUE)
}
# Copy this file to the directory DIR
file.copy('/code/TRAM_DAG_ITE_simulation_pt2_single_model_newDGP_CS_Relu.R', file.path(DIR, 'TRAM_DAG_ITE_simulation_pt2_single_model_newDGP_CS_Relu.R'), overwrite=TRUE)
# file.copy('/code/TRAM_DAG_ITE_simulation_pt2_single_model_newDGP_CS_Relu_regularized.R', file.path(DIR, 'TRAM_DAG_ITE_simulation_pt2_single_model_newDGP_CS_Relu_regularized.R'), overwrite=TRUE)


len_theta = 20 # Number of coefficients of the Bernstein polynomials
hidden_features_I = c(3,3,3) 
hidden_features_CS = c(2,5,5,2) # c(4,8,10,8,4)#c(2,5,5,2) #c(4,8,8,4)# c(2,5,5,2)

# if (F32 == 1){
#   FUN_NAME = 'DPGLinear'
#   f <- function(x) -0.3 * x
# } else if (F32 == 2){
#   f = function(x) 2 * x**3 + x
#   FUN_NAME = 'DPG2x3+x'
# } else if (F32 == 3){
#   f = function(x) 0.5*exp(x)
#   FUN_NAME = 'DPG0.5exp'
# }



MA =  matrix(c(
  0,   0,  0, 'cs',
  0,   0,  0, 'cs',
  0,   0,  0, 'ls',
  0,   0,  0,   0), nrow = 4, ncol = 4, byrow = TRUE)
MODEL_NAME = 'ModelCS'

MA

FUN_NAME = 'x1_ordinal_12_'
# FUN_NAME = 'x1_ordinal_12_CI_'

fn = file.path(DIR, paste0(FUN_NAME, '_', MODEL_NAME))
print(paste0("Starting experiment ", fn))






###############################################################################
# ITE in a simple RCT
###############################################################################



##### DGP ########
dgp <- function(n_obs=20000, doX=c(NA, NA, NA, NA), SEED=123) {
  #n_obs = 1e5 n_obs = 10
  set.seed(SEED)
  
  # Data simulation
  
  ## Case 1: continuous random variables
  
  # Define sample size
  n <- n_obs
  
  # Generate random binary treatment T
  Tr <- rbinom(n, size = 1, prob = 0.5)
  
  p <- 2  # number of variables 
  
  # Define the mean vector (all zeros for simplicity)
  mu <- rep(0, p)  # Mean vector of length p
  
  # Define the covariance matrix (compound symmetric for simplicity)
  rho <- 0.1  # Correlation coefficient
  Sigma <- matrix(rho, nrow = p, ncol = p)  # Start with all elements as rho
  diag(Sigma) <- 1  # Set diagonal elements to 1 (variances)
  
  # Generate n samples from the multivariate normal distribution
  data <- MASS::mvrnorm(n, mu = mu, Sigma = Sigma)
  colnames(data) <- paste0("X", 1:p)
  
  beta_0 <- 0.45
  beta_t <- -0.85
  beta_X <- c(-0.5, 0.1)
  beta_TX <- 0.7
  
  # Calculate the linear predictor (logit)
  logit_Y <- beta_0 + beta_t * Tr + data %*% beta_X + (data[,1] * beta_TX) * Tr
  
  
  # Convert logit to probability of outcome
  Y_prob <- plogis(logit_Y)
  
  # Generate binary outcome Y based on the probability
  Y <- rbinom(n, size = 1, prob = Y_prob)
  
  # Potential outcome for treated and untreated
  Y1 <- plogis(beta_0 + beta_t + data %*% beta_X + data[,1] * beta_TX)
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
  
  
  set.seed(12345)
  test.data <- split_data(simulated_data, 1/2)
  test.compl.data <- remove_NA_data(test.data)
  
  
  # for two-model structure we only need the 2 patient specific variables (no Tr)
  A <- matrix(c(0, 0, 0, 1, 
                0, 0, 0, 1,
                0, 0, 0, 1,
                0, 0, 0, 0), nrow = 4, ncol = 4, byrow = TRUE)
  
  # Full dataset
  dat.orig =  data.frame(x1 = simulated_full_data$Treatment, 
                         x2 = simulated_full_data$X1, 
                         x3 = simulated_full_data$X2, 
                         x4 = simulated_full_data$Y)
  dat_temp <- as.matrix(dat.orig)
  # dat_temp[,4] <- dat_temp[,4] + 1
  dat_temp[,c(1,4)] <- dat_temp[,c(1,4)] + 1
  dat.tf = tf$constant(as.matrix(dat_temp), dtype = 'float32')
  
  # train dataset
  dat.train <- data.frame(x1 = test.compl.data$data.dev$Tr, 
                          x2 = test.compl.data$data.dev$X1, 
                          x3 = test.compl.data$data.dev$X2, 
                          x4 = test.compl.data$data.dev$Y)
  dat_temp <- as.matrix(dat.train)
  # dat_temp[,4] <- dat_temp[,4] + 1
  dat_temp[,c(1,4)] <- dat_temp[,c(1,4)] + 1
  dat.train.tf = tf$constant(as.matrix(dat_temp), dtype = 'float32')
  
  dat.test <- data.frame(x1 = test.compl.data$data.val$Tr, 
                         x2 = test.compl.data$data.val$X1, 
                         x3 = test.compl.data$data.val$X2, 
                         x4 = test.compl.data$data.val$Y)
  dat_temp <- as.matrix(dat.test)
  # dat_temp[,4] <- dat_temp[,4] + 1
  dat_temp[,c(1,4)] <- dat_temp[,c(1,4)] + 1
  dat.test.tf = tf$constant(as.matrix(dat_temp), dtype = 'float32')
  
  
  q1 = c(1, 2)
  q2 = quantile(dat.orig[,2], probs = c(0.05, 0.95))
  q3 = quantile(dat.orig[,3], probs = c(0.05, 0.95))
  q4 = c(1, 2) #No Quantiles for ordinal data
  # q1 = quantile(dat.orig[,2], probs = c(0.05, 0.95)) 
  # q2 = quantile(dat.orig[,3], probs = c(0.05, 0.95))
  # q3 = c(0, 1) #No Quantiles for ordinal data
  
  
  return(list(
    df_orig=dat.tf, 
    df_R = dat.orig,
    min =  tf$reduce_min(dat.tf, axis=0L),
    max =  tf$reduce_max(dat.tf, axis=0L),
    min = tf$constant(c(q1[1], q2[1], q3[1], q4[1]), dtype = 'float32'),
    max = tf$constant(c(q1[2], q2[2], q3[2], q4[2]), dtype = 'float32'),
    
    # min = tf$constant(c(q1[1], q2[1], q3[1]), dtype = 'float32'),
    # max = tf$constant(c(q1[2], q2[2], q3[2]), dtype = 'float32'),
    type = c('o', 'c', 'c', 'o'),
    A=A,
    
    #train
    df_R_train = dat.train,
    df_orig_train = dat.train.tf,
    
    
    # df_orig_train_ct = dat.train.ct.tf,
    # df_R_train_ct = dat.train.ct,
    # 
    # df_orig_train_tx = dat.train.tx.tf,
    # df_R_train_tx = dat.train.tx,
    
    #test
    df_R_test = dat.test,
    df_orig_test = dat.test.tf,
    
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


n_obs <- 20000

dgp_data = dgp(n_obs)

# percentage of patients with Y=1
mean(dgp_data$simulated_full_data$Y)

# percentage of patients with Y=1 in Control (train)
mean(dgp_data$test.compl.data$data.dev.ct$Y)

# percentage of patients with Y=1 in Treatment (train)
mean(dgp_data$test.compl.data$data.dev.tx$Y)

dgp_data$df_orig_test

dgp_data$simulated_full_data

boxplot(Y_prob ~ Y, data = dgp_data$simulated_full_data)


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



#check predictive power of model (GLM with interaction term):

# train

df <- dgp_data$test.compl.data

fit_train <- glm(Y ~ Tr + X1 + X2 + Tr:X1, data = df$data.dev, family = binomial(link="logit")) # glm for binary (negative shift


pred_train <- predict(fit_train, newdata = df$data.dev, type = "response")
pred_test <- predict(fit_train, newdata = df$data.val, type = "response")

df$data.dev$Ypred <- pred_train
df$data.val$Ypred <- pred_test



# train
ggplot(df$data.dev, aes(x = Y_prob, y = Ypred, color = as.factor(Tr))) +
  geom_point() +
  geom_abline(slope = 1, intercept = 0, color = "red") +
  labs(x = "True Probabilities", y = "Estimated Probabilities", title = "Prob GLM (Train)") +
  theme_minimal() +
  theme(legend.position = "top")

# test
ggplot(df$data.val, aes(x = Y_prob, y = Ypred, color = as.factor(Tr))) +
  geom_point() +
  geom_abline(slope = 1, intercept = 0, color = "red") +
  labs(x = "True Probabilities", y = "Estimated Probabilities", title = "Prob GLM (Test)") +
  theme_minimal() +
  theme(legend.position = "top")






#################################################
# fit TRAM-DAG wit CS(T, X1) 
#################################################

(global_min = dgp_data$min)
(global_max = dgp_data$max)
data_type = dgp_data$type

# len_theta_max = len_theta
# for (i in 1:nrow(MA)){ #Maximum number of coefficients (BS and Levels - 1 for the ordinal)
#   if (dgp_data$type[i] == 'o'){
#     len_theta_max = max(len_theta_max, nlevels(dgp_data$df_R[,i]) - 1)
#   }
# }



##### Train on control group ####

param_model = create_param_model(MA, hidden_features_I=hidden_features_I, len_theta=len_theta, hidden_features_CS=hidden_features_CS)
optimizer = optimizer_adam(learning_rate = 0.001)
param_model$compile(optimizer, loss=struct_dag_loss)

h_params <- param_model(dgp_data$df_orig_train)

param_model$evaluate(x = dgp_data$df_orig_train, y=dgp_data$df_orig_train, batch_size = 7L)
summary(param_model)

# show activation function activation_68 --> Relu is used (before it was sigmoid)
param_model$get_layer("activation_48")$get_config()

##### Training ####

# num_epochs <- 1000
num_epochs <- 450   ### final model with c(2,5,5,2) and Relu
# fnh5 = paste0(fn, '_E', num_epochs, '.h5')
# fnRdata = paste0(fn, '_E', num_epochs, '.RData')

fnh5 = paste0(fn, '_E', num_epochs, 'CS.h5')   # 'CI.h5'
fnRdata = paste0(fn, '_E', num_epochs, 'CS.RData')   # 'CI.RData'
if (file.exists(fnh5)){
  param_model$load_weights(fnh5)
  load(fnRdata) #Loading of the workspace causes trouble e.g. param_model is zero
  # Quick Fix since loading global_min causes problem (no tensors as RDS)
  (global_min = dgp_data$min)
  (global_max = dgp_data$max)
} else {
  if (FALSE){ ### Full Training w/o diagnostics
    hist = param_model$fit(x = dgp_data$df_orig_train, y=dgp_data$df_orig_train, epochs = 200L,verbose = TRUE)
    param_model$save_weights(fn)
    plot(hist$epoch, hist$history$loss)
    plot(hist$epoch, hist$history$loss, ylim=c(1.07, 1.2))
  } else { ### Training with diagnostics
    # ws <- data.frame(w12 = numeric())
    ws <- data.frame(w34 = numeric())
    train_loss <- numeric()
    val_loss <- numeric()
    
    # Training loop
    for (e in 1:num_epochs) {
      print(paste("Epoch", e))
      hist <- param_model$fit(x = dgp_data$df_orig_train, y = dgp_data$df_orig_train, 
                              epochs = 1L, verbose = TRUE, 
                              validation_data = list(dgp_data$df_orig_test, dgp_data$df_orig_test))
      
      # Append losses to history
      train_loss <- c(train_loss, hist$history$loss)
      val_loss <- c(val_loss, hist$history$val_loss)
      
      # Extract specific weights
      w <- param_model$get_layer(name = "beta")$get_weights()[[1]]
      
      ws <- rbind(ws, data.frame(w34 = w[3,4]))
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

par(mfrow=c(1,1))
epochs = length(train_loss)
plot(1:length(train_loss), train_loss, type='l', main='Normal Training (green is valid)', ylim = c(2.7, 2.8))
lines(1:length(train_loss), val_loss, type = 'l', col = 'green')

# Last 50
diff = max(epochs - 100,0)
plot(diff:epochs, val_loss[diff:epochs], type = 'l', col = 'green', main='Last 50 epochs')
lines(diff:epochs, train_loss[diff:epochs], type='l')









# learned weights for linear Shift
param_model$get_layer(name = "beta")$get_weights()[[1]] * param_model$get_layer(name = "beta")$mask

# Weight estimates by glm()
# fit_321 <- glm(x3 ~ x1 + x2, data = dgp_data$df_R_train_ct, family = binomial(link="logit")) # glm for binary (negative shift)


p <- ggplot(ws, aes(x=1:nrow(ws))) + 
  geom_line(aes(y=w34, color='x2 --> Y')) + 
  # geom_line(aes(y=w23, color='x2 --> x3')) + 
  # geom_hline(aes(yintercept=-coef(fit_321)[2], color='glm'), linetype=2) +
  # geom_hline(aes(yintercept=-coef(fit_321)[3], color='glm'), linetype=2) +
  #scale_color_manual(values=c('x1 --> x2'='skyblue', 'x1 --> x3='red', 'x2 --> x3'='darkgreen')) +
  labs(x='Epoch', y='Coefficients') +
  theme_minimal() +
  theme(legend.title = element_blank())  # Removes the legend title

p










#################################################
# calculate ITE_i for train and test set
#################################################



do_probability = function (h_params){
  #t_i = intervention_0_tf # (40000, 3)    # original data x1, x2, x3 for each obs
  #h_params = h_params_ct                 # NN outputs (CS, LS, theta') for each obs
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
  #Thetas for intercept (bernstein polynomials?) -> to_theta3 to make them increasing
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
  
  # NLL = 0
  ### Continiuous dimensions
  #### At least one continuous dimension exits
  # if (length(cont_dims) != 0){
  #   
  #   # inputs in h_dag_extra:
  #   # data=(40000, 3), 
  #   # theta=(40000, 3, 20), k_min=(3), k_max=(3))
  #   
  #   # creates the value of the Bernstein at each observation
  #   # and current parameters: output shape=(40000, 3)
  #   h_I = h_dag_extra(t_i[,cont_dims, drop=FALSE], theta[,cont_dims,1:len_theta,drop=FALSE], k_min[cont_dims], k_max[cont_dims]) 
  #   
  #   # adding the intercepts and shifts: results in shape=(40000, 3)
  #   # basically the estimated value of the latent variable
  #   h = h_I + h_LS[,cont_dims, drop=FALSE] + h_CS[,cont_dims, drop=FALSE]
  #   
  #   #Compute terms for change of variable formula
  #   
  #   # log of standard logistic density at h
  #   log_latent_density = -h - 2 * tf$math$softplus(-h) #log of logistic density at h
  #   
  #   ## h' dh/dtarget is 0 for all shift terms
  #   log_hdash = tf$math$log(tf$math$abs(
  #     h_dag_dash_extra(t_i[,cont_dims, drop=FALSE], theta[,cont_dims,1:len_theta,drop=FALSE], k_min[cont_dims], k_max[cont_dims]))
  #   ) - 
  #     tf$math$log(k_max[cont_dims] - k_min[cont_dims])  #Chain rule! See Hathorn page 12 
  #   
  #   NLL = NLL - tf$reduce_mean(log_latent_density + log_hdash)
  # }
  
  ### Ordinal dimensions
  if (length(cont_ord) != 0){
    B = dim(h_params)[1]
    for (col in cont_ord){
      # col=3
      nol = tf$cast(k_max[col] - 1L, tf$int32) # Number of cut-points in respective dimension
      theta_ord = theta[,col,1:nol,drop=TRUE] # Intercept (2 values per observation if 2 cutpoints)
      
      
      h = theta_ord + h_LS[,col, drop=FALSE] + h_CS[,col, drop=FALSE]
      
      cdf_cut <- logistic_cdf(h)
      prob_Y1_X <- 1- cdf_cut
      # # putting -Inf and +Inf to the left and right of the cutpoints
      # neg_inf = tf$fill(c(B,1L), -Inf)
      # pos_inf = tf$fill(c(B,1L), +Inf)
      # h_with_inf = tf$concat(list(neg_inf, h, pos_inf), axis=-1L)
      # logistic_cdf_values = logistic_cdf(h_with_inf)
      # #cdf_diffs <- tf$subtract(logistic_cdf_values[, 2:ncol(logistic_cdf_values)], logistic_cdf_values[, 1:(ncol(logistic_cdf_values) - 1)])
      # cdf_diffs <- tf$subtract(logistic_cdf_values[, 2:tf$shape(logistic_cdf_values)[2]], logistic_cdf_values[, 1:(tf$shape(logistic_cdf_values)[2] - 1)])
      # 
      # # Picking the observed cdf_diff entry for column 4:
      # class_indices <- tf$cast(t_i[, col] - 1, tf$int32)  # Convert to zero-based index
      # # Create batch indices to pair with class indices
      # batch_indices <- tf$range(tf$shape(class_indices)[1])
      # # Combine batch_indices and class_indices into pairs of indices
      # gather_indices <- tf$stack(list(batch_indices, class_indices), axis=1)
      # cdf_diff_picked <- tf$gather_nd(cdf_diffs, gather_indices)
    }
  }
  
  ### DEBUG 
  #if (sum(is.infinite(log_lik$numpy())) > 0){
  #  print("Hall")
  #}
  return (prob_Y1_X)
}


### Training set

# Treatment = 0

# set the values of the first column of dgp_data$df_R_train to 0 and add 1 to the last column
train_df_T0 <- dgp_data$df_R_train %>%
  mutate(
    # x1 = 0,  # Set the first column (x1) to 0
    x1 = 1, # indicating level 0 control
    x4 = x4 + 1 # Add 1 to the last column (x4)
  )

# convert to tensor
train_tf_T0 <- tf$constant(as.matrix(train_df_T0), dtype = 'float32')

# outputs for T=0 on the train set
h_params_ct <- param_model(train_tf_T0)



# Treatment = 1
# set the values of the first column of dgp_data$df_R_train to 1 and add 1 to the last column
train_df_T1 <- dgp_data$df_R_train %>%
  mutate(
    # x1 = 1,  # Set the first column (x1) to 1
    x1 = 2,  # indicating level 1 treatment
    x4 = x4 + 1 # Add 1 to the last column (x4)
  )
# convert to tensor
train_tf_T1 <- tf$constant(as.matrix(train_df_T1), dtype = 'float32')
# outputs for T=1 on the train set
h_params_tx <- param_model(train_tf_T1)







Y0 <- do_probability(h_params_ct)

Y1 <- do_probability(h_params_tx)


ITE_i_train <- Y1 - Y0


ITE_true <- dgp_data$test.compl.data$data.dev$ITE_true

plot(ITE_true, ITE_i_train, xlab = "True ITE", ylab = "Estimated ITE TRAM-DAG", main = "ITE_i")
abline(0,1)



### Test set

# Treatment = 0

# set the values of the first column of dgp_data$df_R_test to 0 and add 1 to the last column
test_df_T0 <- dgp_data$df_R_test %>%
  mutate(
    # x1 = 0,  # Set the first column (x1) to 0
    x1 = 1, # indicating level 0 control
    x4 = x4 + 1 # Add 1 to the last column (x4)
  )
# convert to tensor
test_tf_T0 <- tf$constant(as.matrix(test_df_T0), dtype = 'float32')
# outputs for T=0 on the test set
h_params_ct <- param_model(test_tf_T0)

# Treatment = 1

# set the values of the first column of dgp_data$df_R_test to 1 and add 1 to the last column
test_df_T1 <- dgp_data$df_R_test %>%
  mutate(
    # x1 = 1,  # Set the first column (x1) to 1
    x1 = 2,  # indicating level 1 treatment
    x4 = x4 + 1 # Add 1 to the last column (x4)
  )
# convert to tensor
test_tf_T1 <- tf$constant(as.matrix(test_df_T1), dtype = 'float32')
# outputs for T=1 on the test set
h_params_tx <- param_model(test_tf_T1)


Y0 <- do_probability(h_params_ct)

Y1 <- do_probability(h_params_tx)


ITE_i_test <- Y1 - Y0


ITE_true <- dgp_data$test.compl.data$data.val$ITE_true

plot(ITE_true, ITE_i_test, xlab = "True ITE", ylab = "Estimated ITE TRAM-DAG", main = "ITE_i")
abline(0,1)



#################################################
# Analysis as Holly T-learner (TRAM-DAG)
#################################################

# combine results from TRAM-DAG to the Holly-GLM and the true data

#train
test.results$data.dev.rs$ITE <- as.numeric(ITE_i_train)
test.results$data.dev.rs$RS <- ifelse(test.results$data.dev.rs$ITE > 0, "harm", "benefit")

#test
test.results$data.val.rs$ITE <- as.numeric(ITE_i_test)
test.results$data.val.rs$RS <- ifelse(test.results$data.val.rs$ITE > 0, "harm", "benefit")

# 
# test.results$data.dev.rs
# # Calculate ITE with logistic T-learner
test.results.glm <- logis.ITE(dgp_data$test.compl.data , p=2)


# plot the estimated ITE of the glm and TRAM-DAG: They are almost exactly the same as expected:
par(mfrow=c(1,2))
plot(test.results.glm$data.dev.rs$ITE, ITE_i_train, xlab = "ITE_i glm", ylab = "ITE_i TRAM-DAG", main = "ITE_i Train")
abline(0,1)
plot(test.results.glm$data.val.rs$ITE, ITE_i_test, xlab = "ITE_i glm", ylab = "ITE_i TRAM-DAG", main = "ITE_i Test")
abline(0,1)


data.dev.rs = test.results[["data.dev.rs"]] %>%  as.data.frame()
data.val.rs = test.results[["data.val.rs"]] %>%  as.data.frame()

library(ggpubr)
plot_outcome_ITE(data.dev.rs = data.dev.rs, data.val.rs = data.val.rs, x_lim = c(-0.8,0.8))


plot_ITE_density(test.results = test.results,true.data = dgp_data$simulated_full_data)


plot_ITE_density_tx_ct(data = data.dev.rs)
plot_ITE_density_tx_ct(data = data.val.rs)

par(mfrow=c(1,2))
plot(ITE ~ ITE_true, data = data.dev.rs, col = "orange", pch = 19, cex = 0.5,
     main = "Training Data")
abline(0,1)
plot(ITE ~ ITE_true, data = data.val.rs, col = "#36648B", pch = 19, cex = 0.5,
     main = "Test Data")
abline(0,1)


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
                      log.odds = log.odds, ylb = 0, yub = 5,
                      train.data.name = "Train", test.data.name = "Test")


#True ITE vs TRAM-DAG estimate colored by Treatment


ggplot(test.results$data.dev.rs, aes(x=ITE_true, y=ITE, color=Treatment)) +
  geom_point() +
  geom_abline(slope = 1, intercept = 0, color = "red") +
  labs(title = "Training Data", x = "True ITE", y = "Estimated ITE") +
  theme_minimal() +
  theme(legend.position = "top")

ggplot(test.results$data.val.rs, aes(x=ITE_true, y=ITE, color=Treatment)) +
  geom_point() +
  geom_abline(slope = 1, intercept = 0, color = "red") +
  labs(title = "Test Data", x = "True ITE", y = "Estimated ITE") +
  theme_minimal() +
  theme(legend.position = "top")




# average treatment effects (TRAM-DAG)
mean(test.results$data.dev.rs$ITE)
mean(test.results$data.val.rs$ITE)

# average treatment effects (glm)
mean(test.results.glm$data.dev.rs$ITE)
mean(test.results.glm$data.val.rs$ITE)





################################################################################

##### Check predictive power of the model on train set

train_df <- dgp_data$test.compl.data$data.dev
h_params_orig <- param_model(dgp_data$df_orig_train)
Y_prob_tram_dag <- as.numeric(do_probability(h_params_orig))
train_df$Y_prob_tram <- Y_prob_tram_dag


ggplot(train_df, aes(x = Y_prob, y = Y_prob_tram, color = Treatment)) +
  geom_point() +
  geom_abline(slope = 1, intercept = 0, color = "red") +
  labs(x = "True Probabilities", y = "Estimated Probabilities", title = "Prob TRAM-DAG (Train)") +
  theme_minimal() +
  theme(legend.position = "top")









# interaction glm
glm_interaction <- glm(Y ~ Tr*X1 + X2 + Tr:X1, data = dgp_data$test.compl.data$data.dev, family = binomial(link="logit"))


Y_prob_glm_interaction <- predict(glm_interaction, newdata = dgp_data$test.compl.data$data.dev, type = "response")

train_df$Y_prob_glm <- Y_prob_glm_interaction

ggplot(train_df, aes(x = Y_prob, y = Y_prob_glm, color = as.factor(Tr))) +
  geom_point() +
  geom_abline(slope = 1, intercept = 0, color = "red") +
  labs(x = "True Probabilities", y = "Estimated Probabilities", title = "Prob GLM (Train)") +
  theme_minimal() +
  theme(legend.position = "top")








##### Check predictive power of the model on test set

test_df <- dgp_data$test.compl.data$data.val

# input test data into model
h_params_orig <- param_model(dgp_data$df_orig_test)

# probabilities for Y=1 on original test data
Y_prob_tram_dag <- as.numeric(do_probability(h_params_orig))

# true probabilities for y=1 on dgp
# Y_prob_dgp <- dgp_data$test.compl.data$data.val$Y_prob

test_df$Y_prob_tram <- Y_prob_tram_dag

# plot with ggplot the true probabilities Y_prob against the estimated Y_prob_tram_dag and color accoring to Tr

ggplot(test_df, aes(x = Y_prob, y = Y_prob_tram, color = Treatment)) +
  geom_point() +
  geom_abline(slope = 1, intercept = 0, color = "red") +
  labs(x = "True Probabilities", y = "Estimated Probabilities", title = "Prob TRAM-DAG (Test)") +
  theme_minimal() +
  theme(legend.position = "top")



# interaction glm
glm_interaction <- glm(Y ~ Tr*X1 + X2 + Tr:X1, data = dgp_data$test.compl.data$data.dev, family = binomial(link="logit"))


Y_prob_glm_interaction <- predict(glm_interaction, newdata = dgp_data$test.compl.data$data.val, type = "response")

test_df$Y_prob_glm <- Y_prob_glm_interaction

ggplot(test_df, aes(x = Y_prob, y = Y_prob_glm, color = as.factor(Tr))) +
  geom_point() +
  geom_abline(slope = 1, intercept = 0, color = "red") +
  labs(x = "True Probabilities", y = "Estimated Probabilities", title = "Prob GLM (Test)") +
  theme_minimal() +
  theme(legend.position = "top")

# 
# # Y_prob_glm_simple <- predict(glm_simple, newdata = dgp_data$test.compl.data$data.val, type = "response")
# Y_prob_glm_interaction <- predict(glm_interaction, newdata = dgp_data$test.compl.data$data.val, type = "response")
# plot(Y_prob_dgp, Y_prob_glm_simple, xlab = "True Probabilities", ylab = "Estimated Probabilities", main = "Prob glm_simple")
# abline(0,1, col = "red")
# plot(Y_prob_dgp, Y_prob_glm_interaction, xlab = "True Probabilities", ylab = "Estimated Probabilities", main = "Prob glm_interaction")
# abline(0,1, col = "red")





## seems like the TRAM-DAG has problems at the edges of the distribution



# ### Test
# 
# # input test data into model
# h_params_orig <- param_model(dgp_data$df_orig_train)
# 
# # probabilities for Y=1 on original test data
# Y_prob_tram_dag <- as.numeric(do_probability(h_params_orig))
# 
# # true probabilities for y=1 on dgp
# Y_prob_dgp <- dgp_data$test.compl.data$data.dev$Y_prob
# 
# par(mfrow= c(1,3))
# # plot true against estimated
# plot(Y_prob_dgp, Y_prob_tram_dag, xlab = "True Probabilities", ylab = "Estimated Probabilities", main = "Prob TRAM-DAG")
# abline(0,1, col = "red")





###### Check the estimated CS(T, X1) #######
cs_24_x1_1 <- cs_24_x1_2 <- xs <- seq(-3.2,3.2,length.out=111)

idx0 = which(xs == 0) #Index of 0 xs needs to be odd
for (i in 1:length(xs)){
  #i = 1
  x = xs[i]
  x1 <- 1   # first x1=1
  # Varying x2
  X = tf$constant(c(x1, x, 0, 2), shape=c(1L,4L)) 
  cs_24_x1_1[i] =   param_model(X)[1,4,1]$numpy() #2=CS Term X2->X4 when x1=1
  
  x1 <- 2   # first x1=2
  # Varying x2
  X = tf$constant(c(x1, x, 0, 2), shape=c(1L,4L)) 
  cs_24_x1_2[i] =   param_model(X)[1,4,1]$numpy() #2=CS Term X2->X4 when x1=2
}



q_5 <- quantile(dgp_data$simulated_full_data$X1, c(0.05, 0.95))[1]
q_95 <- quantile(dgp_data$simulated_full_data$X1, c(0.05, 0.95))[2]

# get real values for the shift from dgp
beta_x1 <-  dgp_data$dgp_params$beta_X[1]
beta_x1_tx <-  dgp_data$dgp_params$beta_X[1] + dgp_data$dgp_params$beta_TX

par(mfrow=c(1,2))

delta_0 = cs_24_x1_1[idx0] - 0
plot(xs, cs_24_x1_1 - delta_0, main='CS when X1 Control', 
     sub = 'Effect of x1 & x2 on x4', ylab = 'CS(x2, T=Control)',
     xlab='x2', col='red')
# abline(0, 2)
abline(v=q_5, col = "blue", lty = 2)
abline(v=q_95, col = "blue", lty = 2)
lines(xs, -beta_x1*xs, col = "black")
# if want to compare the treatment effec tin the control plot
# lines(xs, -beta_x1_tx*xs, col = "black")  
legend("topleft", legend=c("CS", "Quantiles", "DGP Effect"), 
       col=c("red", "blue", "black"), lty=c(1, 2, 1), bty="n", cex=0.6)

delta_0 = cs_24_x1_2[idx0] - 0
plot(xs, cs_24_x1_2 - delta_0, main='CS when X1 Treatment', 
     sub = 'Effect of x1 & x2 on x4', ylab = 'CS(x2, T=Treatment)',
     xlab='x2', col='red')
# abline(0, 2)
abline(v=q_5, col = "blue", lty = 2)
abline(v=q_95, col = "blue", lty = 2)
lines(xs, -beta_x1_tx*xs, col = "black")
legend("topright", legend=c("CS", "Quantiles", "DGP Effect"), 
       col=c("red", "blue", "black"), lty=c(1, 2, 1), bty="n", cex=0.6)



# # when Treatment=Control, the CS should only be the effect of x2 on x4
# cs_24_x1_1
# 
# # when x2=0, the CS should only be the intercept, no treatment effect
# cs_24_x1_1[idx0]
# 
# # similar for the CS for Treatment=2 when x2=0, this should only be the treatment effect
# cs_24_x1_2[idx0]
# 
# # the difference of the two curves at x2=0 should be the main effect of T on x4.
# cs_24_x1_2[idx0]- cs_24_x1_1[idx0]
# 
# # DGP main treatment effect is different!
# dgp_data$dgp_params$beta_t















#####################################3
# Calibration
##################################33333


# generate a calibration set (newly generated train set with new SEED)
calibration_dgp <- dgp(n_obs = 20000, SEED=1)
calibration_df <- calibration_dgp$test.compl.data$data.dev


# obtain train probabilities:
h_params_cal <- param_model(calibration_dgp$df_orig_train)
Y_prob_cal <- as.numeric(do_probability(h_params_cal))
calibration_df$Y_prob_cal <- Y_prob_cal



# 
# # plot true vs predicted probabilties
# ggplot(calibration_df, aes(x = Y_prob, y = Y_prob_cal, color = Treatment)) +
#   geom_point() +
#   geom_abline(slope = 1, intercept = 0, color = "red") +
#   labs(x = "True Probabilities", y = "Estimated Probabilities", title = "Prob TRAM-DAG (Calibration)") +
#   theme_minimal() +
#   theme(legend.position = "top")
# 
# # make calibration plot
# library(gbm)
# par(mfrow=c(1,1))
# calibrate.plot(y=calibration_df$Y,
#                p=calibration_df$Y_prob_cal,
#                distribution = "bernoulli")
# 
# # make calibration plot from scratch with 10 bins
# library(ggplot2)
# library(dplyr)
# 
# # Set number of bins
# bins <- 20
# 
# # Create equal-frequency bins based on predicted probabilities
# calibration_df <- calibration_df %>%
#   mutate(prob_bin = cut(
#     Y_prob_cal,
#     breaks = quantile(Y_prob_cal, probs = seq(0, 1, length.out = bins + 1), na.rm = TRUE),
#     include.lowest = TRUE
#   ))
# 
# # Compute average predicted probability and observed proportion in each bin
# calibration_summary <- calibration_df %>%
#   group_by(prob_bin) %>%
#   summarise(
#     mean_pred = mean(Y_prob_cal, na.rm = TRUE),
#     mean_obs = mean(Y, na.rm = TRUE),
#     n = n(),
#     .groups = "drop"
#   )
# 
# # Plot: Calibration curve
# library(ggplot2)
# library(dplyr)
# 
# # Set number of bins
# bins <- 50
# 
# # Create equal-frequency bins based on predicted probabilities
# calibration_df <- calibration_df %>%
#   mutate(prob_bin = cut(
#     Y_prob_cal,
#     breaks = quantile(Y_prob_cal, probs = seq(0, 1, length.out = bins + 1), na.rm = TRUE),
#     include.lowest = TRUE
#   ))
# 
# # Compute average predicted probability and observed proportion in each bin
# calibration_summary <- calibration_df %>%
#   group_by(prob_bin) %>%
#   summarise(
#     mean_pred = mean(Y_prob_cal, na.rm = TRUE),
#     mean_obs = mean(Y, na.rm = TRUE),
#     n = n(),
#     .groups = "drop"
#   )
# 
# # Plot: Calibration curve
# ggplot(calibration_summary, aes(x = mean_pred, y = mean_obs)) +
#   geom_point(size = 2, color = "blue") +
#   geom_line(color = "blue") +
#   geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "red") +
#   labs(
#     title = paste("Calibration Plot (", bins, " bins)", sep = ""),
#     x = "Mean Predicted Probability",
#     y = "Observed Proportion"
#   ) +
#   theme_minimal()
# 
# 
# 
# # Plot: pred vs true curve curve
# ggplot(calibration_df, aes(x = Y_prob_cal, y = Y_prob)) +
#   geom_point(size = 2, color = "blue") +
#   # geom_line(color = "blue") +
#   geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "red") +
#   labs(
#     title = paste("Prediction vs True Plot", sep = ""),
#     x = "Predicted Probability",
#     y = "True Proportion"
#   ) +
#   theme_minimal()
# 


# library(probably)
# 
# calibration_output <- cal_estimate_isotonic(calibration_df,
#                       truth = "Y",
#                       estimate = "Y_prob_cal")
# calibration_output$estimates


# Recalibrate with GAM

library(dplyr)
library(mgcv)
library(binom)
library(ggplot2)

# Set confidence level and bins
my_conf <- 0.95
bins <- 10

# Create equal-frequency bins
calibration_df <- calibration_df %>%
  mutate(prob_bin = cut(
    Y_prob_cal,
    breaks = quantile(Y_prob_cal, probs = seq(0, 1, length.out = bins + 1), na.rm = TRUE),
    # breaks = seq(min(Y_prob_cal), max(Y_prob_cal), length.out = bins + 1),
    include.lowest = TRUE
  ))

# Compute bin summaries
agg_bin <- calibration_df %>%
  group_by(prob_bin) %>%
  summarise(
    pred_probability = mean(Y_prob_cal),
    obs_proportion = mean(Y),
    n_pos = sum(Y == 1),
    n_total = n(),
    .groups = "drop"
  )

# Compute confidence intervals for observed proportions
bin_cis <- mapply(
  function(x, n) binom.confint(x, n, conf.level = my_conf, methods = "wilson")[, c("lower", "upper")],
  agg_bin$n_pos, agg_bin$n_total, SIMPLIFY = FALSE
)
cis_df <- do.call(rbind, bin_cis)
agg_bin$lo_CI_obs_prop <- cis_df[, 1]
agg_bin$up_CI_obs_prop <- cis_df[, 2]
agg_bin$width_CI <- abs(agg_bin$up_CI_obs_prop - agg_bin$lo_CI_obs_prop)

# Plot predicted against observed including CI
ggplot(agg_bin, aes(x = pred_probability, y = obs_proportion)) +
  geom_point(color = "blue", size = 2) +
  geom_errorbar(aes(ymin = lo_CI_obs_prop, ymax = up_CI_obs_prop), width = 0.03) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "red") +
  labs(
    title = paste("Calibration Plot (", bins, " bins)", sep = ""),
    x = "Predicted Probability",
    y = "Observed Proportion"
  ) +
  coord_equal() +
  theme_minimal()

# Fit weighted GAM model
raw_w <- 1 / agg_bin$width_CI
agg_bin$weights <- raw_w / sum(raw_w)

fit_gam <- gam(obs_proportion ~ s(pred_probability), weights = weights, data = agg_bin, gamma = 1)
# fit_gam <- gam(obs_proportion ~ s(pred_probability), data = agg_bin, gamma = 1) # without weights
plot(fit_gam)

# without weights almost same
# fit_gam <- gam(obs_proportion ~ s(pred_probability), data = agg_bin, gamma = 0)
# plot(fit_gam)

# Predict recalibrated probabilities on the calibration set
agg_bin$recal_pred <- predict(fit_gam, newdata = data.frame(pred_probability = agg_bin$pred_probability))
agg_bin$recal_se <- predict(fit_gam, newdata = data.frame(pred_probability = agg_bin$pred_probability), se.fit = TRUE)$se.fit

# Plot
ggplot(agg_bin, aes(x = recal_pred, y = obs_proportion)) +
  geom_point(color = "blue", size = 2) +
  geom_errorbar(aes(ymin = lo_CI_obs_prop, ymax = up_CI_obs_prop), width = 0.03) +
  geom_errorbarh(aes(xmin = recal_pred - recal_se, xmax = recal_pred + recal_se), height = 0.02) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "red") +
  labs(
    title = "GAM-Recalibrated Calibration Plot",
    x = "Recalibrated Predicted Probability",
    y = "Observed Proportion"
  ) +
  coord_equal() +
  theme_minimal()




# check predictions recalibrated on test set


# 1. Extract the test set
test_df <- dgp_data$test.compl.data$data.val

# 2. Compute predicted probabilities on the test set
# Assuming you use the same model function as for calibration:
h_params_test <- param_model(dgp_data$df_orig_test)  # still trained on training data
Y_prob_test <- as.numeric(do_probability(h_params_test))
test_df$Y_prob_cal <- Y_prob_test

# 3. Recalibrate the test probabilities using the GAM model


# Recalibrate using the GAM model from calibration step
test_df$Y_prob_recal <- predict(fit_gam, newdata = data.frame(pred_probability = test_df$Y_prob_cal), type = "response")
# Done! Now test_df has the recalibrated probabilities in Y_prob_recal



ggplot(test_df, aes(x = Y_prob_cal, y = Y_prob_recal, color = Treatment)) +
  geom_point(alpha = 0.3) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "red") +
  labs(title = "GAM Recalibration on Test Set",
       x = "Original Predicted Probability",
       y = "Recalibrated Probability") +
  theme_minimal()

#plot the true prob against recalibrated
ggplot(test_df, aes(x = Y_prob, y = Y_prob_recal, color = Treatment)) +
  geom_point(alpha = 0.3) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "red") +
  labs(title = "True vs Recalibrated Probability on Test Set",
       x = "True Probability",
       y = "Recalibrated Probability") +
  theme_minimal()

# plot the true prob against un-calibrated (test set)
ggplot(test_df, aes(x = Y_prob, y = Y_prob_cal, color = Treatment)) +
  geom_point(alpha = 0.3) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "red") +
  labs(title = "True vs Uncalibrated Probability on Test Set",
       x = "True Probability",
       y = "Uncalibrated Probability") +
  theme_minimal()


# plot the true prob against un-calibrated (train set)
ggplot(train_df, aes(x = Y_prob, y = Y_prob_tram)) +
  geom_point(alpha = 0.3, color = "blue") +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "red") +
  labs(title = "True vs Uncalibrated Probability on Test Set",
       x = "True Probability",
       y = "Uncalibrated Probability") +
  theme_minimal()

# plot the true prob against un-calibrated (calibration set)
ggplot(calibration_df, aes(x = Y_prob, y = Y_prob_cal)) +
  geom_point(alpha = 0.3, color = "blue") +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "red") +
  labs(title = "True vs Uncalibrated Probability on Calibration Set",
       x = "True Probability",
       y = "Uncalibrated Probability") +
  theme_minimal()



################### ITE with recalibrated

test_df



### Test set

# Treatment = 0

# set the values of the first column of dgp_data$df_R_test to 0 and add 1 to the last column
test_df_T0 <- dgp_data$df_R_test %>%
  mutate(
    # x1 = 0,  # Set the first column (x1) to 0
    x1 = 1, # indicating level 0 control
    x4 = x4 + 1 # Add 1 to the last column (x4)
  )
# convert to tensor
test_tf_T0 <- tf$constant(as.matrix(test_df_T0), dtype = 'float32')
# outputs for T=0 on the test set
h_params_ct <- param_model(test_tf_T0)

# Treatment = 1

# set the values of the first column of dgp_data$df_R_test to 1 and add 1 to the last column
test_df_T1 <- dgp_data$df_R_test %>%
  mutate(
    # x1 = 1,  # Set the first column (x1) to 1
    x1 = 2,  # indicating level 1 treatment
    x4 = x4 + 1 # Add 1 to the last column (x4)
  )
# convert to tensor
test_tf_T1 <- tf$constant(as.matrix(test_df_T1), dtype = 'float32')
# outputs for T=1 on the test set
h_params_tx <- param_model(test_tf_T1)


# un-calibrated ITE
Y0 <- do_probability(h_params_ct)
Y1 <- do_probability(h_params_tx)

ITE_i_test <- Y1 - Y0

ITE_true <- dgp_data$test.compl.data$data.val$ITE_true
par(mfrow=c(1,2))
plot(ITE_true, ITE_i_test, xlab = "True ITE", ylab = "Estimated ITE TRAM-DAG", main = "ITE (un-calibrated)")
abline(0,1, col="red")

df_test_uncalibrated <- data.frame(
  ITE_true = ITE_true,
  ITE_i_test = as.numeric(ITE_i_test),
  Treatment = dgp_data$test.compl.data$data.val$Treatment
)




# Recalibrated ITE
Y0_recal <- predict(fit_gam, newdata = data.frame(pred_probability = as.numeric(Y0)), type = "response")
Y1_recal <- predict(fit_gam, newdata = data.frame(pred_probability = as.numeric(Y1)), type = "response")


ITE_i_test_recal <- Y1_recal - Y0_recal

ITE_true <- dgp_data$test.compl.data$data.val$ITE_true

plot(ITE_true, ITE_i_test_recal, xlab = "True ITE", ylab = "Recalibrated ITE TRAM-DAG", main = "ITE (re-calibrated)")
abline(0,1, col="red")


df_test_recalibrated <- data.frame(
  ITE_true = ITE_true,
  ITE_i_test_recal = as.numeric(ITE_i_test_recal),
  Treatment = dgp_data$test.compl.data$data.val$Treatment
)



# ggplot of df_test_uncalibrated
ggplot(df_test_uncalibrated, aes(x = ITE_true, y = ITE_i_test, color = Treatment)) +
  geom_point() +
  geom_abline(slope = 1, intercept = 0, color = "red") +
  labs(title = "Uncalibrated ITE TRAM-DAG", x = "True ITE", y = "Estimated ITE") +
  theme_minimal() +
  theme(legend.position = "top")

# ggplot of df_test_recalibrated
ggplot(df_test_recalibrated, aes(x = ITE_true, y = ITE_i_test_recal, color = Treatment)) +
  geom_point() +
  geom_abline(slope = 1, intercept = 0, color = "red") +
  labs(title = "Recalibrated ITE TRAM-DAG", x = "True ITE", y = "Estimated ITE") +
  theme_minimal() +
  theme(legend.position = "top")





























#########################################
# Follwoing plots used for identification of problems In complex shift 
#########################################

# no more problems in last model with Relu and batchnormalization: 

# x1_ordinal_12__ModelCS_E450CS


### search for reason of bad border estimation

# problems appear outside of TRUE ITE of c(-0.45, 0.20)

# make a new variable in test.results$data.dev.rs for ITE below interval, between and above  c(-0.45, 0.20)

dat <- test.results$data.dev.rs %>%
  mutate(
    ITE_group = case_when(
      ITE_true < -0.45 ~ "below",
      ITE_true >= -0.45 & ITE_true <= 0.20 ~ "between",
      ITE_true > 0.20 ~ "above"
    )
  )

dat$Y_prob_tram <- Y_prob_tram_dag


# plot the true probabilities Y_prob against the estimated Y_prob_tram_dag and color according to ITE_group
ggplot(dat, aes(x = Y_prob, y = Y_prob_tram, color = ITE_group)) +
  # geom_point() +
  # make the points see through (like alpha?)
  geom_point(alpha = 0.3) +
  # select other color palette
  scale_color_manual(values = c("below" = "red", "between" = "green", "above" = "blue")) +
  geom_abline(slope = 1, intercept = 0, color = "red") +
  labs(x = "True Probabilities", y = "Estimated Probabilities", title = "Prob TRAM-DAG (Train Middle)") +
  theme_minimal() +
  theme(legend.position = "top")

# plot X1 against Y_prob_tram and color according to ITE_group
ggplot(dat, aes(x = X1, y = Y_prob_tram, color = ITE_group)) +
  geom_point() +
  # select other color palette
  scale_color_manual(values = c("below" = "red", "between" = "green", "above" = "blue")) +
  labs(x = "X1", y = "Estimated Probabilities", title = "Prob TRAM-DAG (Train Middle)") +
  theme_minimal() +
  theme(legend.position = "top")

# plot X1 against Tr and color according to ITE_group
ggplot(dat, aes(x = X1, y = Tr, color = ITE_group)) +
  geom_point() +
  # select other color palette
  scale_color_manual(values = c("below" = "red", "between" = "green", "above" = "blue")) +
  labs(x = "X1", y = "Treatment", title = "Prob TRAM-DAG (Train Middle)") +
  theme_minimal() +
  theme(legend.position = "top")

# plot X1 against Y and color according to ITE_group
ggplot(dat, aes(x = X1, y = Y, color = ITE_group)) +
  geom_point() +
  # select other color palette
  scale_color_manual(values = c("below" = "red", "between" = "green", "above" = "blue")) +
  labs(x = "X1", y = "Y", title = "Prob TRAM-DAG (Train Middle)") +
  theme_minimal() +
  theme(legend.position = "top")




# plot X1 against Y_prob_tram and color according to ITE_group, but do this separately grouped by Treatment
ggplot(dat, aes(x = X1, y = Y_prob_tram, color = ITE_group)) +
  geom_point() +
  # select other color palette
  scale_color_manual(values = c("below" = "red", "between" = "green", "above" = "blue")) +
  facet_wrap(~Tr) +
  labs(x = "X1", y = "Estimated Probabilities", title = "Prob TRAM-DAG (Train Middle)") +
  theme_minimal() +
  theme(legend.position = "top")



# plot X1 against Y and color according to ITE_group, but do this separately grouped by Treatment
ggplot(dat, aes(x = X1, y = Y, color = ITE_group)) +
  geom_point() +
  # select other color palette
  scale_color_manual(values = c("below" = "red", "between" = "green", "above" = "blue")) +
  facet_wrap(~Tr) +
  labs(x = "X1", y = "Y", title = "Prob TRAM-DAG (Train Middle)") +
  theme_minimal() +
  theme(legend.position = "top")


# plot X2 against Y and color according to ITE_group, but do this separately grouped by Treatment
ggplot(dat, aes(x = X2, y = Y, color = ITE_group)) +
  geom_point() +
  # select other color palette
  scale_color_manual(values = c("below" = "red", "between" = "green", "above" = "blue")) +
  facet_wrap(~Tr) +
  labs(x = "X2", y = "Y", title = "Prob TRAM-DAG (Train Middle)") +
  theme_minimal() +
  theme(legend.position = "top")


# make a summary of the proportion of Y=1 in each group
dat_summary <- dat %>%
  group_by(ITE_group) %>%
  summarise(
    mean_Y_prob = mean(Y_prob),
    mean_Y_prob_tram = mean(Y_prob_tram),
    mean_ITE_true = mean(ITE_true),
    mean_ITE = mean(ITE),
    mean_Y = mean(Y),
    mean_Tr = mean(Tr)
  )
dat_summary


# select only "below" group
dat_below_0.5 <- dat %>%
  filter(ITE_group == "below", 
         Y_prob < 0.5)

# summarize dat_below
dat_below_summary <- dat_below_0.5 %>%
  summarise(
    mean_Y_prob = mean(Y_prob),
    mean_Y_prob_tram = mean(Y_prob_tram),
    mean_ITE_true = mean(ITE_true),
    mean_ITE = mean(ITE),
    mean_Y = mean(Y),
    mean_Tr = mean(Tr)
  )

# summarize dat_below_0.5
dat_below_0.5_summary <- dat_below_0.5 %>%
  summarise(
    mean_Y_prob = mean(Y_prob),
    mean_Y_prob_tram = mean(Y_prob_tram),
    mean_ITE_true = mean(ITE_true),
    mean_ITE = mean(ITE),
    mean_Y = mean(Y),
    mean_Tr = mean(Tr)
  )


# plot dat_below_0.5_summary true prob vs estimate
ggplot(dat_below_0.5, aes(x = Y_prob, y = Y_prob_tram)) +
  geom_point() +
  geom_abline(slope = 1, intercept = 0, color = "red") +
  labs(x = "True Probabilities", y = "Estimated Probabilities", title = "Prob TRAM-DAG (Train Middle)") +
  theme_minimal() +
  theme(legend.position = "top")

# plot dat_below_0.5_summary true prob vs estimate, color by Y
ggplot(dat_below_0.5, aes(x = Y_prob, y = Y_prob_tram, color = X1)) +
  geom_point() +
  geom_abline(slope = 1, intercept = 0, color = "red") +
  labs(x = "True Probabilities", y = "Estimated Probabilities", title = "Prob TRAM-DAG (Train Middle)") +
  theme_minimal() +
  theme(legend.position = "top")

dat_below_0.5_X1_2 <- dat_below_0.5 %>%
  filter(X1 < -2)

# summarize dat_below_0.5_X1_2
dat_below_0.5_X1_2_summary <- dat_below_0.5_X1_2 %>%
  summarise(
    mean_Y_prob = mean(Y_prob),
    mean_Y_prob_tram = mean(Y_prob_tram),
    mean_ITE_true = mean(ITE_true),
    mean_ITE = mean(ITE),
    mean_Y = mean(Y),
    mean_Tr = mean(Tr)
  )
dat_below_0.5_X1_2_summary

#plot X1 against X2
ggplot(dat_below_0.5_X1_2, aes(x = X1, y = X2)) +
  geom_point() +
  # select other color palette
  labs(x = "X1", y = "X2", title = "Prob TRAM-DAG (Train Middle)") +
  theme_minimal() +
  theme(legend.position = "top")

# plot True against estimated probabilities
ggplot(dat_below_0.5_X1_2, aes(x = Y_prob, y = Y_prob_tram)) +
  geom_point() +
  geom_abline(slope = 1, intercept = 0, color = "red") +
  labs(x = "True Probabilities", y = "Estimated Probabilities", title = "Prob TRAM-DAG (Train Middle)") +
  theme_minimal() +
  theme(legend.position = "top")



# analyze for X1 c(-2, 1.5)

dat <- test.results$data.dev.rs %>%
  mutate(
    X1_group = case_when(
      X1 < -2 ~ "below",
      X1 >= -2 & X1 <= 1.5 ~ "between",
      X1 > 1.5 ~ "above"
    )
  )

dat$Y_prob_tram <- Y_prob_tram_dag


# plot the true probabilities Y_prob against the estimated Y_prob_tram_dag and color according to X1_group
ggplot(dat, aes(x = Y_prob, y = Y_prob_tram, color = X1_group)) +
  # geom_point() +
  # make the points see through (like alpha?)
  geom_point(alpha = 0.3) +
  # select other color palette
  scale_color_manual(values = c("below" = "red", "between" = "green", "above" = "blue")) +
  geom_abline(slope = 1, intercept = 0, color = "red") +
  labs(x = "True Probabilities", y = "Estimated Probabilities", title = "Prob TRAM-DAG (Train Middle)") +
  theme_minimal() +
  theme(legend.position = "top")


