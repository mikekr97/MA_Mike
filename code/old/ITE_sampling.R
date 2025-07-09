




###############################################################################
# ITE in a simple RCT
###############################################################################

# draw following DAG with ggdag

# T = treatment, D = outcome, X = covariate
# R = randomization

library(dagitty)
library(ggdag)
library(ggplot2)
dag <- dagify(T ~ R,
              X1 ~ T,
              D ~ T + X1 + X2,
              exposure = "T",
              outcome = "D" )

tidy_dagitty(dag)
ggdag(dag)



###################################################################################################
source("code/utils/ITE_utils.R")

#########################################

# Data simulation

## Case 1: continuous random variables



# Define sample size
n <- 10000

set.seed(123)
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


# Define coefficients for the logistic model
beta_0 <- runif(1, min=-1, max=1)             # Intercept
beta_t <- runif(1, min=-0.5, max=0.5)             # Coefficient for treatment
beta_X <- runif(p, min=-1, max=1)          # Coefficients for covariates
beta_TX <- runif(1, min=-0.5, max=0.5)          # Coefficient for interaction terms
# beta_TX <- runif(p, min=-0.5, max=0.5)          # Coefficient for interaction terms

# Calculate the linear predictor (logit)
logit_Y <- beta_0 + beta_t * Tr + data %*% beta_X + (data[,1] * beta_TX) * Tr

# logit_Y <- beta_0 + beta_t * Tr + data %*% beta_X + (data %*% beta_TX) * Tr

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


# Combine all variables into a single data frame
simulated_full_data <- data.frame(ID = 1:n, Y=Y, Treatment=Tr, data, Y1, Y0, ITE_true)

library(dplyr)
# Data for testing ITE models
simulated_data <- data.frame(ID =1:n, Y=Y, Treatment=Tr, data, ITE_true = ITE_true) %>% 
  mutate(Treatment = ifelse(Treatment==1,"Y", "N")) %>% 
  mutate(Treatment = factor(Treatment, levels = c("N", "Y")))

# Check Simulated Data 

library(ggplot2)
# Visualize the distribution of the individual treatment effect
simulated_full_data %>% ggplot(aes(x=ITE_true)) +
  geom_histogram(color="gray8",bins = 100, alpha=.6) + 
  theme_minimal() + 
  xlab("True ITE") + 
  geom_vline(xintercept = 0, linetype="dashed") + coord_cartesian(xlim = c(-.5,.5))
simulated_full_data %>% ggplot(aes(x=ITE_true)) +
  geom_density(color="gray8",fill="skyblue",alpha=.6) + 
  theme_minimal() + 
  xlab("True ITE") + 
  geom_vline(xintercept = 0, linetype="dashed") + coord_cartesian(xlim = c(-.5,.5))



str(simulated_data)

# average treatment effects
mean(simulated_full_data$ITE_true)







set.seed(12345)
test.data <- split_data(simulated_data, 2/3)
test.compl.data <- remove_NA_data(test.data)
test.results <- logis.ITE(test.compl.data , p=2)



data.dev.rs = test.results[["data.dev.rs"]] %>%  as.data.frame()
data.val.rs = test.results[["data.val.rs"]] %>%  as.data.frame()

library(ggpubr)
plot_outcome_ITE(data.dev.rs = data.dev.rs, data.val.rs = data.val.rs, x_lim = c(-0.5,0.5))


plot_ITE_density(test.results = test.results,true.data = simulated_full_data)


plot_ITE_density_tx_ct(data = data.dev.rs)
plot_ITE_density_tx_ct(data = data.val.rs)

par(mfrow=c(1,2))
plot(ITE ~ ITE_true, data = data.dev.rs, col = "orange", pch = 19, cex = 0.5,
     xlab = "ITE", ylab = "True ITE", main = "Training Data")
plot(ITE ~ ITE_true, data = data.val.rs, col = "#36648B", pch = 19, cex = 0.5,
     xlab = "ITE", ylab = "True ITE", main = "Test Data")


par(mfrow=c(1,2))
plot(data.dev.rs$ITE_true, data.dev.rs$ITE)

# compare estimated ITE and true ITE
mean(abs(data.dev.rs$ITE_true - data.dev.rs$ITE))
mean(abs(data.val.rs$ITE_true - data.val.rs$ITE))

ggplot(data.dev.rs, aes(x=ITE_true, y=ITE)) + 
  geom_point() + 
  geom_abline(intercept = 0, slope = 1, linetype="dashed") + 
  labs(title="Training Data")+
  xlab("True ITE")+
  ylab("Estimated ITE")+
  theme_minimal()+
  theme(
    legend.position.inside = c(.9, .9),
    legend.justification = c("right", "top"),
    legend.box.just = "right",
    panel.grid.major = element_blank(),  # Removes major grid lines
    panel.grid.minor = element_blank(),  # Removes minor grid lines
    panel.background = element_blank(),  # Removes panel background
    plot.background = element_blank(),    # Removes plot background (outside the plot)
    text = element_text(size = 13),
    axis.line = element_line(color = "black"),
    axis.ticks = element_line(color = "black")
  )

ggplot(data.val.rs, aes(x=ITE_true, y=ITE)) +
  geom_point() + 
  geom_abline(intercept = 0, slope = 1, linetype="dashed") + 
  labs(title="Test Data")+
  xlab("True ITE")+
  ylab("Estimated ITE")+
  theme_minimal()+
  theme(
    legend.position.inside = c(.9, .9),
    legend.justification = c("right", "top"),
    legend.box.just = "right",
    panel.grid.major = element_blank(),  # Removes major grid lines
    panel.grid.minor = element_blank(),  # Removes minor grid lines
    panel.background = element_blank(),  # Removes panel background
    plot.background = element_blank(),    # Removes plot background (outside the plot)
    text = element_text(size = 13),
    axis.line = element_line(color = "black"),
    axis.ticks = element_line(color = "black")
  )




# breaks <-  c(-0.3, -0.15, -0.07, 0, 0.07, 0.15, 0.35)
#breaks <- c(-0.025, -0.01, -0.005, 0, 0.005, 0.01,  0.025)

breaks <- c(-0.15, -0.07, -0.02, 0.02, 0.06)
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
                      log.odds = log.odds, ylb = 0, yub = 2,
                      train.data.name = "Train", test.data.name = "Test")








###############################3

#  Holly code:

set.seed(1234)

# Define sample size
n <- 20000

# Generate random binary treatment T
Tr <- rbinom(n, size = 1, prob = 0.5)

p <- 20  # number of variables

# Define the mean vector (all zeros for simplicity)
mu <- rep(0, p)  # Mean vector of length p

# Define the covariance matrix (compound symmetric for simplicity)
rho <- 0.1  # Correlation coefficient
Sigma <- matrix(rho, nrow = p, ncol = p)  # Start with all elements as rho
diag(Sigma) <- 1  # Set diagonal elements to 1 (variances)

# Generate n samples from the multivariate normal distribution
data <- MASS::mvrnorm(n, mu = mu, Sigma = Sigma)
colnames(data) <- paste0("X", 1:p)

# Define coefficients for the logistic model
beta_0 <- runif(1, min=-1, max=1)             # Intercept
beta_t <- runif(1, min=-0.5, max=0.5)             # Coefficient for treatment
beta_X <- runif(p, min=-1, max=1)          # Coefficients for covariates
beta_TX <- runif(p, min=-0.5, max=0.5)          # Coefficients for interaction terms

# Calculate the linear predictor (logit)
logit_Y <- beta_0 + beta_t * Tr + data %*% beta_X + (data %*% beta_TX) * Tr

# Convert logit to probability of outcome
Y_prob <- plogis(logit_Y)

# Generate binary outcome Y based on the probability
Y <- rbinom(n, size = 1, prob = Y_prob)

# Potential outcome for treated and untreated
Y1 <- plogis(beta_0 + beta_t + data %*% beta_X + data %*% beta_TX)
Y0 <- plogis(beta_0 + data %*% beta_X)

# Calculate the individual treatment effect
ITE_true <- Y1 - Y0


# Combine all variables into a single data frame
simulated_full_data <- data.frame(ID = 1:n, Y=Y, Treatment=Tr, data, Y1, Y0, ITE_true)

# Data for testing ITE models
simulated_data <- data.frame(ID =1:n, Y=Y, Treatment=Tr, data, ITE_true = ITE_true) %>% 
  mutate(Treatment = ifelse(Treatment==1,"Y", "N")) %>% 
  mutate(Treatment = factor(Treatment, levels = c("N", "Y")))


# Visualize the distribution of the individual treatment effect
simulated_full_data %>% ggplot(aes(x=ITE_true)) +
  geom_histogram(color="gray8",bins = 100, alpha=.6) + 
  theme_minimal() + 
  xlab("True ITE") + 
  geom_vline(xintercept = 0, linetype="dashed") + coord_cartesian(xlim = c(-.5,.5))
simulated_full_data %>% ggplot(aes(x=ITE_true)) +
  geom_density(color="gray8",fill="skyblue",alpha=.6) + 
  theme_minimal() + 
  xlab("True ITE") + 
  geom_vline(xintercept = 0, linetype="dashed") + coord_cartesian(xlim = c(-.5,.5))


str(simulated_data)

# average treatment effects
mean(simulated_full_data$ITE_true)


simulated_data_truncated <- simulated_data %>% 
  dplyr::select(ID, Y, Treatment, X4, X8, X9, X12, X15, X18, ITE_true) %>%
  rename(X1 = X4, X2 = X8, X3 = X9, X4 = X12, X5 = X15, X6 = X18)

str(simulated_data_truncated)

filepath <- here("Results", "07_simulation", "20_continuous_covariates_20000_sample_insufficient_data")
model_name <- "base_logistic_regression"

set.seed(12345)
test.data <- split_data(simulated_data_truncated, 2/3)
test.compl.data <- remove_NA_data(test.data)
test.results <- logis.ITE(test.compl.data , p=6)

data.dev.rs = test.results[["data.dev.rs"]] %>%  as.data.frame()
data.val.rs = test.results[["data.val.rs"]] %>%  as.data.frame()

par(mfrow=c(1,2))
plot(ITE ~ ITE_true, data = data.dev.rs, col = "orange", pch = 19, cex = 0.5,
     xlab = "ITE", ylab = "True ITE", main = "Training Data")
plot(ITE ~ ITE_true, data = data.val.rs, col = "#36648B", pch = 19, cex = 0.5,
     xlab = "ITE", ylab = "True ITE", main = "Test Data")

plot_outcome_ITE(data.dev.rs = data.dev.rs, data.val.rs = data.val.rs, x_lim = c(-0.5,0.5))


plot_ITE_density(test.results = test.results,true.data = simulated_full_data)
ggsave(paste(model_name, "ITE_density.tiff", sep = "_"), width = 5, height = 5, units = "in", 
       dpi = 300, path = filepath)

plot_ITE_density_tx_ct(data = data.dev.rs)
plot_ITE_density_tx_ct(data = data.val.rs)

breaks <-  c(-0.3, -0.15, -0.07, 0, 0.07, 0.15, 0.35)
#breaks <- c(-0.025, -0.01, -0.005, 0, 0.005, 0.01,  0.025)
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

plot_ATE_ITE_in_group(dev.data = data.dev.grouped.ATE, val.data = data.val.grouped.ATE, log.odds = log.odds, ylb = 0, yub = 50)
ggsave(paste(model_name, "ATE_ITE.tiff", sep = "_"), width = 7, height = 4, units = "in", 
       dpi = 300, path = filepath)











###############################3

# recreate



# fit individual models one for T=0 and one for T=1
fit.dev.tx <- glm(Y ~X1 + X2 + X3 + X4 + X5 + X6, data = test.compl.data$data.dev.tx, family = binomial(link = "logit"))
fit.dev.ct <- glm(Y ~X1 + X2 + X3 + X4 + X5 + X6, data = test.compl.data$data.dev.ct,  family = binomial(link = "logit"))


## P(Y=1 | do(X1=0), X)
prob_T0 <- predict(fit.dev.ct, newdata = test.compl.data$data.dev, type = "response")

## P(Y=1 | do(X1=1), X)
prob_T1 <- predict(fit.dev.tx, newdata = test.compl.data$data.dev, type = "response")

# same as above
vars <- paste0("X", 1:6)
prob_T1 <- predict(fit.dev.tx, newdata = test.compl.data$data.dev[, vars], type = "response")
prob_T0 <- predict(fit.dev.ct, newdata = test.compl.data$data.dev[, vars], type = "response")

ITE_i <- prob_T1 - prob_T0

# validation
ITE_i_val <- predict(fit.dev.tx, newdata = test.compl.data$data.val, type = "response") - 
  predict(fit.dev.ct, newdata = test.compl.data$data.val, type = "response")


# histogram of ITE_i
par(mfrow=c(1,1))
hist(ITE_i, main='ITE_i')
hist(ITE_i_val, main='ITE_i_val')



par(mfrow=c(1,2))

# fit logistic regression, stratified by Treatment T
fit_ITE <- glm(Y ~ ITE_i:Treatment, data = test.compl.data$data.dev, family = binomial(link = "logit"))


# make predictions for T = 0 and T = 1
xs <- seq(min(ITE_i), max(ITE_i), length.out = 100)
pred_0 <- predict(fit_ITE, newdata = data.frame(ITE_i = xs, Treatment = as.factor("N")), type = "response")
pred_1 <- predict(fit_ITE, newdata = data.frame(ITE_i = xs, Treatment = as.factor("Y")), type = "response")

plot(ITE_i, test.compl.data$data.dev$Y, main='ITE_i vs Y_i', xlab='ITE_i', ylab='Y_i')
lines(xs, pred_0, col = "red", lwd = 2)
lines(xs, pred_1, col = "blue", lwd = 2)


# fit logistic regression, stratified by Treatment T
fit_ITE_val <- glm(Y ~ ITE_i_val:Treatment, data = test.compl.data$data.val, family = binomial(link = "logit"))


# make predictions for T = 0 and T = 1
pred_0 <- predict(fit_ITE_val, newdata = data.frame(ITE_i_val = xs, Treatment = as.factor("N")), type = "response")
pred_1 <- predict(fit_ITE_val, newdata = data.frame(ITE_i_val = xs, Treatment = as.factor("Y")), type = "response")

plot(ITE_i_val, test.compl.data$data.val$Y, main='ITE_i_val vs Y_i', xlab='ITE_i', ylab='Y_i')
lines(xs, pred_0, col = "red", lwd = 2)
lines(xs, pred_1, col = "blue", lwd = 2)

# --> same result





hist(data.dev.rs$ITE)

ITE_i == data.dev.rs$ITE
ITE_i_val == data.val.rs$ITE




###########################
# Observational Simulations
###########################


# Simulation 1


# draw following DAG with ggdag

# T = treatment {0,1}, Y = outcome {0,1}, X = covariate {Real-value}

# X1 -> T 
# X1 -> Y
# T -> Y
# T -> X2
# X2 -> Y

# plot dag
library(dagitty)
library(ggdag)
dag <- dagify(Y ~ Tr + X1 + X2,
              Tr ~ X1,
              X2 ~ Tr,
              exposure = "T",
              outcome = "Y" )
tidy_dagitty(dag)
ggdag(dag)



set.seed(123)



dgp <- function(n, SEED = 123, interaction = FALSE){
  
  set.seed(SEED)
  
  # Simulate X1 as a continuous covariate
  X1 <- rnorm(n, mean = 0, sd = 1)
  
  
  # Generate treatment T depending on X1
  logit_T <- 0.3 * X1
  prob_T <- plogis(logit_T)
  Tr <- rbinom(n, 1, prob_T)
  sum(Tr)
  
  # Generate mediator X2 based on treatment
  
  x_2_dash <- rlogis(n)
  
  # X2 <- 0.8 * Tr + rnorm(n)
  # x_2_dash <- 0.8*X2 + 0.5 * Tr
  X2 <- (x_2_dash - 0.4*Tr) / 1.5
  
  X2_T0 <- (x_2_dash - 0.4*0) / 1.5
  X2_T1 <- (x_2_dash - 0.4*1) / 1.5
  
  
  # Generate outcome Y depending on T, X1, and X2, with interaction
  beta_0 <- -0.3
  beta_t <- 0.7
  beta_X1 <- 0.4
  beta_X2 <- 0.5
  
  # interaction term Tr:X1 -> Y
  beta_TX1 <- ifelse(interaction, -0.3, 0)  # if with interaction, set to -0.3, else 0
  
  logit_Y <- beta_0 + beta_t * Tr + beta_X1 * X1 + beta_X2 * X2 + 
    beta_TX1 * X1 * Tr
  prob_Y <- plogis(logit_Y)
  Y <- rbinom(n, 1, prob_Y)
  
  
  # Potential outcome for treated and untreated
  Y1 <- plogis(beta_0 + beta_t + beta_X1 * X1 + beta_X2 * X2_T1 + 
                 beta_TX1 * X1)
  Y0 <- plogis(beta_0  + beta_X1 * X1 + beta_X2 * X2_T0)
  
  # Calculate the individual treatment effect
  ITE_true <- Y1 - Y0
  
  
  
  # Combine into data frame
  data <- data.frame(X1, Tr, X2, Y, Y1, Y0, ITE_true)
  return(data)
}


calc_ITE <- function(data, interaction = FALSE){
  
  # Fit model
  if(interaction){
    fit1 <- glm(Y ~ Tr + X1 + X2 + Tr:X1, data = data$train, family = binomial)
  } else {
    fit1 <- glm(Y ~ Tr + X1 + X2, data = data$train, family = binomial)
  }
  
  # potential outcome under treatment
  Y_T1 <- predict(fit1, newdata = data.frame(Tr = 1, X1 = data$train$X1, X2 = data$train$X2), type = "response")
  # potential outcome under control
  Y_T0 <- predict(fit1, newdata = data.frame(Tr = 0, X1 = data$train$X1, X2 = data$train$X2), type = "response")
  # Individualized treatment effect
  ITE_i <- Y_T1 - Y_T0
  data$train$ITE_i <- ITE_i
  # generate new column for treatment with factor levels "N" and "Y"
  data$train$Treatment <- ifelse(data$train$Tr == 1, "Y", "N")
  
  
  # ITE on the test set
  Y_T1_test <- predict(fit1, newdata = data.frame(Tr = 1, X1 = data$test$X1, X2 = data$test$X2), type = "response")
  Y_T0_test <- predict(fit1, newdata = data.frame(Tr = 0, X1 = data$test$X1, X2 = data$test$X2), type = "response")
  ITE_i_test <- Y_T1_test - Y_T0_test
  data$test$ITE_i <- ITE_i_test
  # generate new column for treatment with factor levels "N" and "Y"
  data$test$Treatment <- ifelse(data$test$Tr == 1, "Y", "N")
  
  return(list(data=data, model=fit1))
  
}


## Evaluate

rmse <- function(dat) {
  y_true <- dat$ITE_true
  y_pred <- dat$ITE_i
  sqrt(mean((y_true - y_pred)^2))  # same as (y_true - y_pred) %*% (y_true - y_pred) / nrow(dat)
}


train <- dgp(n = 1000, SEED = 123, interaction = FALSE)
test <- dgp(n = 1000, SEED = 1234, interaction = FALSE)

data <- list(train = train, test = test)

# treatment allocation
plot(train$X1, train$Tr)

# x1 vs x2 
plot(train$X1, train$X2, col=Tr+1, pch=19, cex=.5, xlab="X1", ylab="X2")


ite_estimation <- calc_ITE(data, interaction = FALSE)
train <- ite_estimation$data$train
test <- ite_estimation$data$test



# Check the distribution of ITE_i
ggplot(train, aes(x=ITE_i)) +
  geom_histogram(color="gray8",bins = 100, alpha=.6) + 
  theme_minimal() + 
  xlab("ITE_i") + 
  geom_vline(xintercept = 0, linetype="dashed") + coord_cartesian(xlim = c(-.5,.5))


# Check the distribution of ITE_true
ggplot(train, aes(x=ITE_true)) +
  geom_histogram(color="gray8",bins = 100, alpha=.6) + 
  theme_minimal() + 
  xlab("ITE_true") + 
  geom_vline(xintercept = 0, linetype="dashed") + coord_cartesian(xlim = c(-.5,.5))


# Plot ITE_true vs ITE_i
plot(ITE_i ~ ITE_true, data = train, col = "orange", pch = 19, cex = 0.5,
     xlab = "ITE_true", ylab = "ITE_i", main = "ITE_i vs ITE_true")


rmse(train)
rmse(test)



# Simulation 2 (with interaction)



# T = treatment {0,1}, Y = outcome {0,1}, X = covariate {Real-value}

# X1 -> T 
# X1 -> Y

# T -> X1:Y

# T -> Y
# T -> X2
# X2 -> Y




train <- dgp(n = 1000, SEED = 123, interaction = TRUE)
test <- dgp(n = 1000, SEED = 1234, interaction = TRUE)

data <- list(train = train, test = test)

# treatment allocation
plot(train$X1, train$Tr)

# x1 vs x2 
plot(train$X1, train$X2, col=Tr+1, pch=19, cex=.5, xlab="X1", ylab="X2")


ite_estimation <- calc_ITE(data, interaction = TRUE)
train <- ite_estimation$data$train
test <- ite_estimation$data$test



# Check the distribution of ITE_i
ggplot(train, aes(x=ITE_i)) +
  geom_histogram(color="gray8",bins = 100, alpha=.6) + 
  theme_minimal() + 
  xlab("ITE_i") + 
  geom_vline(xintercept = 0, linetype="dashed") + coord_cartesian(xlim = c(-.5,.5))


# Check the distribution of ITE_true
ggplot(train, aes(x=ITE_true)) +
  geom_histogram(color="gray8",bins = 100, alpha=.6) + 
  theme_minimal() + 
  xlab("ITE_true") + 
  geom_vline(xintercept = 0, linetype="dashed") + coord_cartesian(xlim = c(-.5,.5))


# Plot ITE_true vs ITE_i
plot(train$ITE_i ~ train$ITE_true, data = train, col = "orange", pch = 19, cex = 0.5,
     xlab = "ITE_true", ylab = "ITE_i", main = "ITE_i vs ITE_true")




# Check the distribution of ITE_i
ggplot(test, aes(x=ITE_i)) +
  geom_histogram(color="gray8",bins = 100, alpha=.6) + 
  theme_minimal() + 
  xlab("ITE_i") + 
  geom_vline(xintercept = 0, linetype="dashed") + coord_cartesian(xlim = c(-.5,.5))


# Check the distribution of ITE_true
ggplot(test, aes(x=ITE_true)) +
  geom_histogram(color="gray8",bins = 100, alpha=.6) + 
  theme_minimal() + 
  xlab("ITE_true") + 
  geom_vline(xintercept = 0, linetype="dashed") + coord_cartesian(xlim = c(-.5,.5))

# Plot ITE_true vs ITE_i
plot(ITE_i ~ ITE_true, data = test, col = "orange", pch = 19, cex = 0.5,
     xlab = "ITE_true", ylab = "ITE_i", main = "ITE_i vs ITE_true")


### The problem is that X2 is dependent on T, but we do not change X2 for the two 
# interventional queries.




p1 <- ggplot(data=train, aes(x=ITE_i, y=Y))+
  geom_point(aes(color=Treatment))+
  geom_smooth(aes(color=Treatment, fill=Treatment), method = "glm", method.args = list(family = "binomial"), alpha=0.5)+
  coord_cartesian(xlim=c(-0.5,0.5), ylim = c(0,1))+
  ylab("Outcome")+xlab("ITE")+
  scale_color_manual(values=c("N" = "orange", "Y" = "#36648B"), name="Treatment") +
  scale_fill_manual(values = c("N" = "orange", "Y" = "#36648B"), name="Treatment") +
  theme_minimal()+
  theme(
    panel.grid.major = element_blank(),  # Removes major grid lines
    panel.grid.minor = element_blank(),  # Removes minor grid lines
    panel.background = element_blank(),  # Removes panel background
    plot.background = element_blank(),    # Removes plot background (outside the plot)
    text = element_text(size = 14),
    axis.line = element_line(color = "black"),
    axis.ticks = element_line(color = "black")
  )
p1


p2 <- ggplot(data=test, aes(x=ITE_i, y=Y))+
  geom_point(aes(color=Treatment))+
  geom_smooth(aes(color=Treatment, fill=Treatment), method = "glm", method.args = list(family = "binomial"), alpha=0.5)+
  coord_cartesian(xlim=c(-0.5,0.5), ylim = c(0,1))+
  ylab("Outcome")+xlab("ITE")+
  scale_color_manual(values=c("N" = "orange", "Y" = "#36648B"), name="Treatment") +
  scale_fill_manual(values = c("N" = "orange", "Y" = "#36648B"), name="Treatment") +
  theme_minimal()+
  theme(
    panel.grid.major = element_blank(),  # Removes major grid lines
    panel.grid.minor = element_blank(),  # Removes minor grid lines
    panel.background = element_blank(),  # Removes panel background
    plot.background = element_blank(),    # Removes plot background (outside the plot)
    text = element_text(size = 14),
    axis.line = element_line(color = "black"),
    axis.ticks = element_line(color = "black")
  )
p2



rmse(train)
rmse(test)



#### Case 3: RCT

# T -> Y
# X1 -> Y
# X2 -> Y
# T:X1 -> Y  Optional interaction


dgp_RCT <- function(n, SEED = 123, interaction = FALSE, hidden_confounders = FALSE){
  
  set.seed(SEED)
  
  
  
  
  # Generate treatment T depending on X1
  Tr <- rbinom(n, 1, 0.5)
  # sum(Tr)
  
  # Generate X1 as bimodal continuout covariate
  X_1_A = rnorm(n, 0.25, 0.1)
  X_1_B = rnorm(n, 0.73, 0.05)
  X1 = ifelse(sample(1:2, replace = TRUE, size = n) == 1, X_1_A, X_1_B)
  
  # Simulate X2 as a continuous covariate
  X2 <- rnorm(n, mean = 2, sd = 0.7)
  
  if (hidden_confounders) {
    # Generate hidden confounder
    U <- rnorm(n, mean = 0.3, sd = 1.2)
    
    # generate Tr depending on U
    Tr <- rbinom(n, 1, plogis(0.5 * U))
  }
  
  
  
  # Generate outcome Y depending on T, X1, and X2, otional with interaction T:X1
  beta_0 <- -0.3
  beta_t <- 0.7
  beta_X1 <- 0.4
  beta_X2 <- 0.5
  
  # interaction term Tr:X1 -> Y
  beta_TX1 <- ifelse(interaction, -0.3, 0)  # if with interaction, set to -0.3, else 0
  
  logit_Y <- beta_0 + beta_t * Tr + beta_X1 * X1 + beta_X2 * X2 + 
    beta_TX1 * X1 * Tr

  prob_Y <- plogis(logit_Y)
  Y <- rbinom(n, 1, prob_Y)
  
  
  # Potential outcome for treated and untreated
  Y1 <- plogis(beta_0 + beta_t + beta_X1 * X1 + beta_X2 * X2 + 
                 beta_TX1 * X1)
  Y0 <- plogis(beta_0  + beta_X1 * X1 + beta_X2 * X2)
  
  
  
  if(hidden_confounders) {
    logit_Y <- beta_0 + beta_t * Tr + beta_X1 * X1 + beta_X2 * X2 + 
      beta_TX1 * X1 * Tr + 0.6 * U
    prob_Y <- plogis(logit_Y)
    Y <- rbinom(n, 1, prob_Y)
    # Potential outcome for treated and untreated
    Y1 <- plogis(beta_0 + beta_t + beta_X1 * X1 + beta_X2 * X2 + 
                   beta_TX1 * X1 + 0.6 * U)
    Y0 <- plogis(beta_0  + beta_X1 * X1 + beta_X2 * X2 + 0.6 * U)
  }
  
  # Calculate the individual treatment effect
  ITE_true <- Y1 - Y0
  
  
  
  # Combine into data frame
  data <- data.frame(X1, Tr, X2, Y, Y1, Y0, ITE_true)
  return(data)
}



train <- dgp_RCT(n = 1000, SEED = 123, interaction = FALSE)
test <- dgp_RCT(n = 1000, SEED = 1234, interaction = FALSE)

data <- list(train = train, test = test)


par(mfrow=c(1,3))
# plot the distribution of X1
hist(train$X1, main='X1', xlab='X1', ylab='Frequency')
# plot the distribution of X2
hist(train$X2, main='X2', xlab='X2', ylab='Frequency')
# plot the distribution of Y
hist(train$Y, main='Y', xlab='Y', ylab='Frequency')

par(mfrow=c(1,1))
# x1 vs x2 
plot(train$X1, train$X2, col=Tr+1, pch=19, cex=.5, xlab="X1", ylab="X2")



ite_estimation <- calc_ITE(data, interaction = FALSE)
train <- ite_estimation$data$train
test <- ite_estimation$data$test




# Check the distribution of ITE_i
ggplot(train, aes(x=ITE_i)) +
  geom_histogram(color="gray8",bins = 100, alpha=.6) + 
  theme_minimal() + 
  xlab("ITE_i") + 
  geom_vline(xintercept = 0, linetype="dashed") + coord_cartesian(xlim = c(-.5,.5))


# Check the distribution of ITE_true
ggplot(train, aes(x=ITE_true)) +
  geom_histogram(color="gray8",bins = 100, alpha=.6) + 
  theme_minimal() + 
  xlab("ITE_true") + 
  geom_vline(xintercept = 0, linetype="dashed") + coord_cartesian(xlim = c(-.5,.5))


# Plot ITE_true vs ITE_i
plot(train$ITE_i ~ train$ITE_true, data = train, col = "orange", pch = 19, cex = 0.5,
     xlab = "ITE_true", ylab = "ITE_i", main = "ITE_i vs ITE_true")




# Check the distribution of ITE_i
ggplot(test, aes(x=ITE_i)) +
  geom_histogram(color="gray8",bins = 100, alpha=.6) + 
  theme_minimal() + 
  xlab("ITE_i") + 
  geom_vline(xintercept = 0, linetype="dashed") + coord_cartesian(xlim = c(-.5,.5))


# Check the distribution of ITE_true
ggplot(test, aes(x=ITE_true)) +
  geom_histogram(color="gray8",bins = 100, alpha=.6) + 
  theme_minimal() + 
  xlab("ITE_true") + 
  geom_vline(xintercept = 0, linetype="dashed") + coord_cartesian(xlim = c(-.5,.5))

# Plot ITE_true vs ITE_i
plot(ITE_i ~ ITE_true, data = test, col = "orange", pch = 19, cex = 0.5,
     xlab = "ITE_true", ylab = "ITE_i", main = "ITE_i vs ITE_true")



p1 <- ggplot(data=train, aes(x=ITE_i, y=Y))+
  geom_point(aes(color=Treatment))+
  geom_smooth(aes(color=Treatment, fill=Treatment), method = "glm", method.args = list(family = "binomial"), alpha=0.5)+
  coord_cartesian(xlim=c(-0.5,0.5), ylim = c(0,1))+
  ylab("Outcome")+xlab("ITE")+
  scale_color_manual(values=c("N" = "orange", "Y" = "#36648B"), name="Treatment") +
  scale_fill_manual(values = c("N" = "orange", "Y" = "#36648B"), name="Treatment") +
  theme_minimal()+
  theme(
    panel.grid.major = element_blank(),  # Removes major grid lines
    panel.grid.minor = element_blank(),  # Removes minor grid lines
    panel.background = element_blank(),  # Removes panel background
    plot.background = element_blank(),    # Removes plot background (outside the plot)
    text = element_text(size = 14),
    axis.line = element_line(color = "black"),
    axis.ticks = element_line(color = "black")
  )
p1


p2 <- ggplot(data=test, aes(x=ITE_i, y=Y))+
  geom_point(aes(color=Treatment))+
  geom_smooth(aes(color=Treatment, fill=Treatment), method = "glm", method.args = list(family = "binomial"), alpha=0.5)+
  coord_cartesian(xlim=c(-0.5,0.5), ylim = c(0,1))+
  ylab("Outcome")+xlab("ITE")+
  scale_color_manual(values=c("N" = "orange", "Y" = "#36648B"), name="Treatment") +
  scale_fill_manual(values = c("N" = "orange", "Y" = "#36648B"), name="Treatment") +
  theme_minimal()+
  theme(
    panel.grid.major = element_blank(),  # Removes major grid lines
    panel.grid.minor = element_blank(),  # Removes minor grid lines
    panel.background = element_blank(),  # Removes panel background
    plot.background = element_blank(),    # Removes plot background (outside the plot)
    text = element_text(size = 14),
    axis.line = element_line(color = "black"),
    axis.ticks = element_line(color = "black")
  )
p2









### Case 4: RCT with interaction




train <- dgp_RCT(n = 1000, SEED = 123, interaction = TRUE)
test <- dgp_RCT(n = 1000, SEED = 1234, interaction = TRUE)

data <- list(train = train, test = test)


par(mfrow=c(1,3))
# plot the distribution of X1
hist(train$X1, main='X1', xlab='X1', ylab='Frequency')
# plot the distribution of X2
hist(train$X2, main='X2', xlab='X2', ylab='Frequency')
# plot the distribution of Y
hist(train$Y, main='Y', xlab='Y', ylab='Frequency')

par(mfrow=c(1,1))
# x1 vs x2 
plot(train$X1, train$X2, col=Tr+1, pch=19, cex=.5, xlab="X1", ylab="X2")



ite_estimation <- calc_ITE(data, interaction = TRUE)
train <- ite_estimation$data$train
test <- ite_estimation$data$test




# Check the distribution of ITE_i
ggplot(train, aes(x=ITE_i)) +
  geom_histogram(color="gray8",bins = 100, alpha=.6) + 
  theme_minimal() + 
  xlab("ITE_i") + 
  geom_vline(xintercept = 0, linetype="dashed") + coord_cartesian(xlim = c(-.5,.5))


# Check the distribution of ITE_true
ggplot(train, aes(x=ITE_true)) +
  geom_histogram(color="gray8",bins = 100, alpha=.6) + 
  theme_minimal() + 
  xlab("ITE_true") + 
  geom_vline(xintercept = 0, linetype="dashed") + coord_cartesian(xlim = c(-.5,.5))


# Plot ITE_true vs ITE_i
plot(train$ITE_i ~ train$ITE_true, data = train, col = "orange", pch = 19, cex = 0.5,
     xlab = "ITE_true", ylab = "ITE_i", main = "ITE_i vs ITE_true")




# Check the distribution of ITE_i
ggplot(test, aes(x=ITE_i)) +
  geom_histogram(color="gray8",bins = 100, alpha=.6) + 
  theme_minimal() + 
  xlab("ITE_i") + 
  geom_vline(xintercept = 0, linetype="dashed") + coord_cartesian(xlim = c(-.5,.5))


# Check the distribution of ITE_true
ggplot(test, aes(x=ITE_true)) +
  geom_histogram(color="gray8",bins = 100, alpha=.6) + 
  theme_minimal() + 
  xlab("ITE_true") + 
  geom_vline(xintercept = 0, linetype="dashed") + coord_cartesian(xlim = c(-.5,.5))

# Plot ITE_true vs ITE_i
plot(ITE_i ~ ITE_true, data = test, col = "orange", pch = 19, cex = 0.5,
     xlab = "ITE_true", ylab = "ITE_i", main = "ITE_i vs ITE_true")



p1 <- ggplot(data=train, aes(x=ITE_i, y=Y))+
  geom_point(aes(color=Treatment))+
  geom_smooth(aes(color=Treatment, fill=Treatment), method = "glm", method.args = list(family = "binomial"), alpha=0.5)+
  coord_cartesian(xlim=c(-0.5,0.5), ylim = c(0,1))+
  ylab("Outcome")+xlab("ITE")+
  scale_color_manual(values=c("N" = "orange", "Y" = "#36648B"), name="Treatment") +
  scale_fill_manual(values = c("N" = "orange", "Y" = "#36648B"), name="Treatment") +
  theme_minimal()+
  theme(
    panel.grid.major = element_blank(),  # Removes major grid lines
    panel.grid.minor = element_blank(),  # Removes minor grid lines
    panel.background = element_blank(),  # Removes panel background
    plot.background = element_blank(),    # Removes plot background (outside the plot)
    text = element_text(size = 14),
    axis.line = element_line(color = "black"),
    axis.ticks = element_line(color = "black")
  )
p1


p2 <- ggplot(data=test, aes(x=ITE_i, y=Y))+
  geom_point(aes(color=Treatment))+
  geom_smooth(aes(color=Treatment, fill=Treatment), method = "glm", method.args = list(family = "binomial"), alpha=0.5)+
  coord_cartesian(xlim=c(-0.5,0.5), ylim = c(0,1))+
  ylab("Outcome")+xlab("ITE")+
  scale_color_manual(values=c("N" = "orange", "Y" = "#36648B"), name="Treatment") +
  scale_fill_manual(values = c("N" = "orange", "Y" = "#36648B"), name="Treatment") +
  theme_minimal()+
  theme(
    panel.grid.major = element_blank(),  # Removes major grid lines
    panel.grid.minor = element_blank(),  # Removes minor grid lines
    panel.background = element_blank(),  # Removes panel background
    plot.background = element_blank(),    # Removes plot background (outside the plot)
    text = element_text(size = 14),
    axis.line = element_line(color = "black"),
    axis.ticks = element_line(color = "black")
  )
p2





rmse(train)
rmse(test)




### Case 5 : RCT with interaction and hidden confounders



train <- dgp_RCT(n = 1000, SEED = 123, interaction = FALSE,
                 hidden_confounders = TRUE)
test <- dgp_RCT(n = 1000, SEED = 1234, interaction = FALSE,
                hidden_confounders = TRUE)

data <- list(train = train, test = test)


par(mfrow=c(1,3))
# plot the distribution of X1
hist(train$X1, main='X1', xlab='X1', ylab='Frequency')
# plot the distribution of X2
hist(train$X2, main='X2', xlab='X2', ylab='Frequency')
# plot the distribution of Y
hist(train$Y, main='Y', xlab='Y', ylab='Frequency')

par(mfrow=c(1,1))
# x1 vs x2 
plot(train$X1, train$X2, col=Tr+1, pch=19, cex=.5, xlab="X1", ylab="X2")



ite_estimation <- calc_ITE(data, interaction = FALSE)
train <- ite_estimation$data$train
test <- ite_estimation$data$test




# Check the distribution of ITE_i
ggplot(train, aes(x=ITE_i)) +
  geom_histogram(color="gray8",bins = 100, alpha=.6) + 
  theme_minimal() + 
  xlab("ITE_i") + 
  geom_vline(xintercept = 0, linetype="dashed") + coord_cartesian(xlim = c(-.5,.5))


# Check the distribution of ITE_true
ggplot(train, aes(x=ITE_true)) +
  geom_histogram(color="gray8",bins = 100, alpha=.6) + 
  theme_minimal() + 
  xlab("ITE_true") + 
  geom_vline(xintercept = 0, linetype="dashed") + coord_cartesian(xlim = c(-.5,.5))


# Plot ITE_true vs ITE_i
plot(train$ITE_i ~ train$ITE_true, data = train, col = "orange", pch = 19, cex = 0.5,
     xlab = "ITE_true", ylab = "ITE_i", main = "ITE_i vs ITE_true")




# Check the distribution of ITE_i
ggplot(test, aes(x=ITE_i)) +
  geom_histogram(color="gray8",bins = 100, alpha=.6) + 
  theme_minimal() + 
  xlab("ITE_i") + 
  geom_vline(xintercept = 0, linetype="dashed") + coord_cartesian(xlim = c(-.5,.5))


# Check the distribution of ITE_true
ggplot(test, aes(x=ITE_true)) +
  geom_histogram(color="gray8",bins = 100, alpha=.6) + 
  theme_minimal() + 
  xlab("ITE_true") + 
  geom_vline(xintercept = 0, linetype="dashed") + coord_cartesian(xlim = c(-.5,.5))

# Plot ITE_true vs ITE_i
plot(ITE_i ~ ITE_true, data = test, col = "orange", pch = 19, cex = 0.5,
     xlab = "ITE_true", ylab = "ITE_i", main = "ITE_i vs ITE_true")



p1 <- ggplot(data=train, aes(x=ITE_i, y=Y))+
  geom_point(aes(color=Treatment))+
  geom_smooth(aes(color=Treatment, fill=Treatment), method = "glm", method.args = list(family = "binomial"), alpha=0.5)+
  coord_cartesian(xlim=c(-0.5,0.5), ylim = c(0,1))+
  ylab("Outcome")+xlab("ITE")+
  scale_color_manual(values=c("N" = "orange", "Y" = "#36648B"), name="Treatment") +
  scale_fill_manual(values = c("N" = "orange", "Y" = "#36648B"), name="Treatment") +
  theme_minimal()+
  theme(
    panel.grid.major = element_blank(),  # Removes major grid lines
    panel.grid.minor = element_blank(),  # Removes minor grid lines
    panel.background = element_blank(),  # Removes panel background
    plot.background = element_blank(),    # Removes plot background (outside the plot)
    text = element_text(size = 14),
    axis.line = element_line(color = "black"),
    axis.ticks = element_line(color = "black")
  )
p1


p2 <- ggplot(data=test, aes(x=ITE_i, y=Y))+
  geom_point(aes(color=Treatment))+
  geom_smooth(aes(color=Treatment, fill=Treatment), method = "glm", method.args = list(family = "binomial"), alpha=0.5)+
  coord_cartesian(xlim=c(-0.5,0.5), ylim = c(0,1))+
  ylab("Outcome")+xlab("ITE")+
  scale_color_manual(values=c("N" = "orange", "Y" = "#36648B"), name="Treatment") +
  scale_fill_manual(values = c("N" = "orange", "Y" = "#36648B"), name="Treatment") +
  theme_minimal()+
  theme(
    panel.grid.major = element_blank(),  # Removes major grid lines
    panel.grid.minor = element_blank(),  # Removes minor grid lines
    panel.background = element_blank(),  # Removes panel background
    plot.background = element_blank(),    # Removes plot background (outside the plot)
    text = element_text(size = 14),
    axis.line = element_line(color = "black"),
    axis.ticks = element_line(color = "black")
  )
p2



