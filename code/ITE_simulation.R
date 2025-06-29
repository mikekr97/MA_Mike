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

#### For TF (not needed here)
# source('code/utils/utils_tf.R')

#### For TFP (not needed here)
# library(tfprobability)
# source('code/utils/utils_tfp.R')

##### Flavor of experiment ######

#### Saving the current version of the script into runtime
DIR = 'runs/ITE_simulation/run'
if (!dir.exists(DIR)) {
  dir.create(DIR, recursive = TRUE)
}
# Copy this file to the directory DIR
file.copy('/code/ITE_simulation.R', file.path(DIR, 'ITE_simulation.R'), overwrite=TRUE)




###############################################################################
# ITE in an RCT 
###############################################################################

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
  
  return(list(
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






#### Scenarios: generate the data ####

# choose the scenario


# Scenario 1: fully observed, strong main & interaction effect
# Scenario 2: unobserved interaction, strong main & interaction effect
# Scenario 3: unobserved interaction, strong main & interaction effect & 4 unobserved variables
# Scenario 4: other setting, results look similar as in IST trial
# Scenario 5: fully observed, weak main & interaction effect (same as scenario 1 but small effects) 

scenario <- 6




# assign TRUE or FALSE to main_effect, interaction_effect according to selected scenario with if
if (scenario == 1) {
  data <-dgp_simulation(n_obs=20000, SEED=123,
                                      rho=0.1,
                                      beta_0 = 0.45,
                                      beta_t = -0.85,
                                      beta_X = c(-0.5, 0.8, 0.2, 0.6, -0.4),
                                      beta_TX = c(0.9, 0.1),
                                      p0 = 0, 
                                      confounder=FALSE,  # number indicates the index of confounder (only one possible)
                                      drop=FALSE)  
  
} else if (scenario == 2) {
  data <-dgp_simulation(n_obs=20000, SEED=123,
                                        rho=0.1,
                                        beta_0 = 0.45,
                                        beta_t = -0.85,
                                        beta_X = c(-0.5, 0.8, 0.2, 0.6, -0.4),
                                        beta_TX = c(0.9, 0.1),
                                        p0 = 0, 
                                        confounder=FALSE, 
                                        drop=c("X1"))      # X1 is not observed
} else if (scenario == 3) {
  data <-dgp_simulation(n_obs=20000, SEED=123,
                                        rho=0.1,
                                        beta_0 = 0.45,
                                        beta_t = -0.85,
                                        beta_X = c(-0.5, 0.8, 0.2, 0.6, -0.4),
                                        beta_TX = c(0.9, 0.1),
                                        p0 = 4,            # 4 unnecessary variables
                                        confounder=FALSE, 
                                        drop=c("X1")) 
} else if (scenario == 4) {
  data <- dgp_simulation(n_obs=20000, SEED=123,
                                       rho=0.1,
                                       beta_0 = 0.45,
                                       beta_t = 0.02,        # small direct effect
                                       beta_X = c(-0.5, 0.1, 0.2, -0.7, -0.15, 0.3, 0.05, -0.2), # other covariate effects
                                       beta_TX = c(-0.5, 0.04, 0.001),   # medium interaction effect
                                       p0 = 4, 
                                       confounder=FALSE, 
                                       drop=c("X1"))  # X1 is not observed
  } else if (scenario == 5) {
    data <- dgp_simulation(n_obs=20000, SEED=123,
                           rho=0.1,
                           beta_0 = 0.45,
                           beta_t = -0.05,  # small direct effect
                           beta_X = c(-0.5, 0.8, 0.2, 0.6, -0.4),
                           beta_TX = c(-0.01, 0.03),   # small interaction effect
                           p0 = 0, 
                           confounder=FALSE, 
                           drop=FALSE)
  } else if (scenario == 6) {
    data <-dgp_simulation(n_obs=20000, SEED=123,
                          rho=0,   # uncorrelated
                          beta_0 = 0.45,
                          beta_t = -0.85,
                          beta_X = c(-0.5, 0.8, 0.2, 0.6, -0.4),
                          beta_TX = c(0.9, -0.5),   # changed sign and value of present interaction 
                          p0 = 0, 
                          confounder=FALSE, 
                          drop=c("X1"))      # X1 is not observed
  } else {
  stop("Invalid scenario selected")
}





# set scenario_name for plot naming
if (scenario == 1) {
  scenario_name <- 'fully_observed'
} else if (scenario == 2) {
  scenario_name <- 'unobserved_interaction'
} else if (scenario == 3) {
  scenario_name <- 'unobserved_interaction_4unnecessary'
} else if (scenario == 4) {
  scenario_name <- 'similar_IST_trial'
} else if (scenario == 5) {
  scenario_name <- 'small_interaction'
} else if (scenario == 6) {
  scenario_name <- 'changed_sign_interaction'
} else {
  stop("Invalid scenario number selected")
}

DIR_szenario <- file.path(DIR, scenario_name)
# create a folder named MODEL_NAME in DIR and save the image p in this folder with name 'sampling_distributions.png'
if (!dir.exists(DIR_szenario)) {
  dir.create(DIR_szenario, recursive = TRUE)
}



#### Models: fit the model ####

# choose the model

# Model 1: fit.glm(df)               # GLM T-learner
# Model 2: fit.glmnet(df)            # GLMnet T-learner (lasso)
# Model 3: fit.glmnet.slearner(df)   # GLMnet S-learner (lasso)
# Model 5: fit.rf(df, ntrees = 100)  # randomForest T-learner (not tuned)
# Model 6: fit.tuned_rf(df)          # tuned Random Forest T-learner (comets)


model_nr <- 1

# set scenario_name for plot naming
if (model_nr == 1) {
  model_name <- 'glm_tlearner'
} else if (model_nr == 2) {
  model_name <- 'glmnet_tlearner'
} else if (model_nr == 3) {
  model_name <- 'glmnet_slearner'
} else if (model_nr == 4) {
  model_name <- 'rf_tlearner'
} else if (model_nr == 5) {
  model_name <- 'tuned_rf_tlearner'
} else {
  stop("Invalid model number selected")
}




#### fit model ####

set.seed(123) # for reproducibility

if (model_nr == 1) {
  model.results <- fit.glm(data$test.compl.data)
} else if (model_nr == 2) {
  model.results <- fit.glmnet(data$test.compl.data)
} else if (model_nr == 3) {
  model.results <- fit.glmnet.slearner(data$test.compl.data)
} else if (model_nr == 4) {
  model.results <- fit.rf(data$test.compl.data, ntrees = 100)
} else if (model_nr == 5) {
  model.results <- fit.tuned_rf(data$test.compl.data)
} else {
  stop("Invalid model number selected")
}



#### display results ####

print(paste0("Scenario: ", scenario_name, ", Model: ", model_name))

# show predicted ITE in train set (to see range of ITE)
par(mfrow=c(1,1))
hist(model.results$data.dev.rs$ITE)

# check ATE

# ATE_Estimated = mean(ITE_predicted)
# ATE_Observed = mean(Y_obs|T=1)-mean(Y_obs|T=0)
# ATE_True = mean(ITE_true)

check_ate(model.results)  

# plot results (adjust breaks according to ITE range)
# breaks <- round(seq(-0.8, 0.6, by = 0.2), 1)
breaks <- round(seq(-0.5, 0.2, by = 0.1), 1)
# breaks <- round(seq(-0.2, 0.2, by = 0.1), 1)
# breaks <- round(seq(-0.3, 0, by = 0.05), 2)
# breaks <- round(seq(-0.04, 0.04, by = 0.02), 2)


                
plot_for_slides(model.results = model.results, 
                breaks = breaks, 
                delta_horizontal = 0.002,
                ylim_delta = 0.1)


### check number of positive ITE_true and ITE in train and test set
sum(model.results$data.dev.rs$ITE_true > 0)
sum(model.results$data.dev.rs$ITE > 0)

sum(model.results$data.val.rs$ITE_true > 0)
sum(model.results$data.val.rs$ITE > 0)




# ATE in the high ITE group as P(Y=1|ITE_true>0.2 T=1) - P(Y=1|ITE_true>0.2 T=0)
dat_0.2 <- model.results$data.val.rs[model.results$data.val.rs$ITE_true > 0.2,]
mean(dat_0.2[dat_0.2$Tr == 1,]$Y) - mean(dat_0.2[dat_0.2$Tr == 0,]$Y)

# ATE in the low ITE group as P(Y=1|ITE_true<-0.5 T=1) - P(Y=1|ITE_true<-0.5 T=0)
dat_0.5 <- model.results$data.val.rs[model.results$data.val.rs$ITE_true < -0.5,]
mean(dat_0.5[dat_0.5$Tr == 1,]$Y) - mean(dat_0.5[dat_0.5$Tr == 0,]$Y)


# proportion of positive outcomes when the true ITE was positive (56%)
sum(model.results$data.val.rs[model.results$data.val.rs$ITE_true > 0,]$Y)/nrow(model.results$data.val.rs[model.results$data.val.rs$ITE_true > 0,])

# proportion of positive outcomes when the true ITE was negative (48%)
sum(model.results$data.val.rs[model.results$data.val.rs$ITE_true < 0,]$Y)/nrow(model.results$data.val.rs[model.results$data.val.rs$ITE_true < 0,])


# proportion of positive outcomes when the true ITE was > 0.2 (55%)
sum(model.results$data.val.rs[model.results$data.val.rs$ITE_true > 0.2,]$Y)/nrow(model.results$data.val.rs[model.results$data.val.rs$ITE_true > 0.2,])

# proportion of positive outcomes when the true ITE was <0.2 (48%)
sum(model.results$data.val.rs[model.results$data.val.rs$ITE_true < 0.2,]$Y)/nrow(model.results$data.val.rs[model.results$data.val.rs$ITE_true < 0.2,])


#### save results ####

file_name <- file.path(DIR_szenario, paste0(scenario_name, "_", model_name, '.png'))


png(filename = file_name, width = 2350, height = 1150, res = 300)

plot_for_slides(model.results = model.results, 
                breaks = breaks, 
                delta_horizontal = 0.002,
                ylim_delta = 0.1)
# Close PNG device
dev.off()

# save ATE check_ate(model.results)
ate_file_name <- file.path(DIR_szenario, paste0(scenario_name, "_", model_name, '_ate.txt'))
write.table(check_ate(model.results), file = ate_file_name, row.names = FALSE, col.names = TRUE, sep = "\t")











################# 
# other plots
#################


# only ITE-ATE plot (base R)
breaks <- round(seq(-0.6, 0.4, by=0.2), 1)
plot_CATE_vs_ITE_base_risk(model.results = model.results, breaks, delta_horizontal = 0.02)

# full results with ggplot including ITE-Outcome plot
plot_pred_ite(model.results, ate_ite = TRUE)
plot_pred_ite(model.results, ate_ite = FALSE)

# calibration plot
par(mfrow=c(1,1), pty="s")
library("CalibrationCurves")
# train
res <- val.prob.ci.2(model.results$data.dev.rs$Y_pred, model.results$data.dev.rs$Y)
# test
res <- val.prob.ci.2(model.results$data.val.rs$Y_pred, model.results$data.val.rs$Y)




file_name <- file.path(DIR_szenario, paste0(scenario_name, "_", model_name, 'train_calibration_plot.png'))
png(filename = file_name, width = 1600, height = 1600, res = 300)
# train
res <- val.prob.ci.2(model.results$data.dev.rs$Y_pred, model.results$data.dev.rs$Y)

# Close PNG device
dev.off()


file_name <- file.path(DIR_szenario, paste0(scenario_name, "_", model_name, 'test_calibration_plot.png'))
png(filename = file_name, width = 1600, height = 1600, res = 300)
# train
res <- val.prob.ci.2(model.results$data.val.rs$Y_pred, model.results$data.val.rs$Y)

# Close PNG device
dev.off()























#################################################
# Analysis as Holly T-learner GLM
#################################################

dgp_data <- data
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







