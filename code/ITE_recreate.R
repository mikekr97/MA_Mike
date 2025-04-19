




###############################################################################
# ITE in a simple RCT
###############################################################################

# draw following DAG with ggdag

# R->T , T->D , T-> C , C->D
# T = treatment, D = outcome, C = covariate
# R = randomization

library(dagitty)
library(ggdag)
library(ggplot2)
dag <- dagify(T ~ R,
              C ~ T,
              D ~ T + C,
              exposure = "T",
              outcome = "D" )

tidy_dagitty(dag)
ggdag(dag, layout = "circle")


make_samples <- function(n){
  # simulate T {0,1}
  # n<-500
  T <- as.factor(rep(c(0,1), each=n))
  
  # simulate C ~ T
  C <- rnorm(1000, mean = as.numeric(T)*2, sd = 1)
  # sd(C)
  
  # simulate D ~ T + C
  Dc <- rnorm(1000, mean = as.numeric(T) + C*2, sd = 1)
  # hist(Dc)
  # mean(Dc)
  D <- as.factor(ifelse(Dc > 5.7, 1, 0))
  df <- data.frame(T=T, C=C, D=D)
  return(df) 
  
}

train <- make_samples(500)
test <- make_samples(500)


par(mfrow=c(1,3))
plot(train$T, main='T')
hist(train$C, main='C ~ T')
plot(train$D, main='D ~ T + C')


#####################################################
# fit a logistic regression model
#####################################################

fit1 <- glm(D ~ T + C, data = train, family = binomial(link = "logit"))
summary(fit1)


#####################################################
# calculate ITE_i for train set
#####################################################


#ITE_i = P[D=1 | do(T_i=1), X_i ] - P[D=1 | do(T_i=0), X_i ]

# set the do-intervention (X1=0)
intervention_0 <- train
intervention_0[,1] <- as.factor(0)

# set the outcome of interest (X3 = 1)
intervention_0[,3] <- as.factor(1)

# manipulated dataframe for do(x1=0)
head(intervention_0)



# set the do-intervention (X1=1)
intervention_1 <- train
intervention_1[,1] <- as.factor(1)

# set the outcome of interest (X3 = 1)
intervention_1[,3] <- as.factor(1)

# manipulated dataframe for do(x1=1)
head(intervention_1)


## P(Y=1 | do(X1=0), X2)
prob_T0 <- predict(fit1, newdata = intervention_0, type = "response")

## P(Y=1 | do(X1=1), X2)
prob_T1 <- predict(fit1, newdata = intervention_1, type = "response")



ITE_i <- prob_T1 - prob_T0

# histogram of ITE_i
par(mfrow=c(1,1))
hist(ITE_i, main='ITE_i')

plot(ITE_i, train$D, main='ITE_i vs Y_i', xlab='ITE_i', ylab='Y_i')


# fit logistic regression, stratified by Treatment T
fit_ITE <- glm(D ~ ITE_i + T, data = train, family = binomial(link = "logit"))


# make predictions for T = 0 and T = 1
pred_0 <- predict(fit_ITE, newdata = data.frame(ITE_i = ITE_i, T = as.factor(0)), type = "response")
pred_1 <- predict(fit_ITE, newdata = data.frame(ITE_i = ITE_i, T = as.factor(1)), type = "response")

plot(ITE_i, as.numeric(train$D)-1, main='ITE_i vs Y_i', xlab='ITE_i', ylab='Y_i')
lines(ITE_i, pred_0, col = "red", lwd = 2)
lines(ITE_i, pred_1, col = "blue", lwd = 2)











#####################################################
# calculate ITE_i for test set
#####################################################


#ITE_i = P[D=1 | do(T_i=1), X_i ] - P[D=1 | do(T_i=0), X_i ]

# set the do-intervention (X1=0)
intervention_0 <- test
intervention_0[,1] <- as.factor(0)

# set the outcome of interest (X3 = 1)
intervention_0[,3] <- as.factor(1)

# manipulated dataframe for do(x1=0)
head(intervention_0)



# set the do-intervention (X1=1)
intervention_1 <- test
intervention_1[,1] <- as.factor(1)

# set the outcome of interest (X3 = 1)
intervention_1[,3] <- as.factor(1)

# manipulated dataframe for do(x1=1)
head(intervention_1)


## P(Y=1 | do(X1=0), X2)
prob_T0 <- predict(fit1, newdata = intervention_0, type = "response")

## P(Y=1 | do(X1=1), X2)
prob_T1 <- predict(fit1, newdata = intervention_1, type = "response")



ITE_i <- prob_T1 - prob_T0

# histogram of ITE_i
par(mfrow=c(1,1))
hist(ITE_i, main='ITE_i')

plot(ITE_i, test$D, main='ITE_i vs Y_i', xlab='ITE_i', ylab='Y_i')


# fit logistic regression, stratified by Treatment T
fit_ITE <- glm(D ~ ITE_i + T, data = test, family = binomial(link = "logit"))


# make predictions for T = 0 and T = 1
pred_0 <- predict(fit_ITE, newdata = data.frame(ITE_i = ITE_i, T = as.factor(0)), type = "response")
pred_1 <- predict(fit_ITE, newdata = data.frame(ITE_i = ITE_i, T = as.factor(1)), type = "response")

plot(ITE_i, as.numeric(train$D)-1, main='ITE_i vs Y_i', xlab='ITE_i', ylab='Y_i')
lines(ITE_i, pred_0, col = "red", lwd = 2)
lines(ITE_i, pred_1, col = "blue", lwd = 2)



































##########################################################3
# simulation study from ITE paper
#############################################################


library("here")
library("readxl")
library("tidyverse")
library("summarytools")
library("tableone")
library("scales")
library("reshape2")
library("broom")
library("progress")
library("CalibrationCurves")
library("gtools")
library("rms")
library("splines")
library("randomForest")
library("xgboost")
library("e1071")
library("grf")
library("glmnet")
library("glinternet")
library("ggpubr")
library("model4you")
library("ggbeeswarm")
library("iteval")
library("BART")
library("bcf")
library("ranger")
library("xgboost")
library("MASS")

source(here("utils", "helpers_9.R"), local = knitr::knit_global())
source(here("utils", "helpers_10.R"), local = knitr::knit_global())

# Simulate Data
# Case 1: continuous random variables
set.seed(1234)

# Define sample size
n <- 20000

# Generate random binary treatment T
Tr <- rbinom(n, size = 1, prob = 0.5)
p <- 4  # number of variables

# Define the mean vector (all zeros for simplicity)
mu <- rep(0, p)  # Mean vector of length p

# Define the covariance matrix (compound symmetric for simplicity)
rho <- 0.1  # Correlation coefficient
Sigma <- matrix(rho, nrow = p, ncol = p)  # Start with all elements as rho
diag(Sigma) <- 1  # Set diagonal elements to 1 (variances)

# Generate n samples from the multivariate normal distribution
data <- mvrnorm(n, mu = mu, Sigma = Sigma)
colnames(data) <- paste0("X", 1:p)

# Define coefficients for the logistic model
beta_0 <- runif(1, min=-1, max=1)             # Intercept
beta_t <- runif(1, min=-0.5, max=0.5)             # Coefficient for treatment
beta_X <- runif(p, min=-1, max=1)          # Coefficients for covariates


beta_TX <- runif(p, min=-0.5, max=0.5)          # Coefficients for interaction terms

# Calculate the linear predictor (logit)
logit_Y <- beta_0 + beta_t * Tr + data %*% beta_X + (data %*% beta_TX) * Tr


# logit_Y <- beta_0 + beta_t * Tr + data %*% beta_X # without interaction

# Convert logit to probability of outcome
Y_prob <- plogis(logit_Y)

# Generate binary outcome Y based on the probability
Y <- rbinom(n, size = 1, prob = Y_prob)

# Potential outcome for treated and untreated
Y1 <- plogis(beta_0 + beta_t + data %*% beta_X + data %*% beta_TX)
# Y1 <- plogis(beta_0 + beta_t + data %*% beta_X) # without interaction
Y0 <- plogis(beta_0 + data %*% beta_X)

# Calculate the individual treatment effect
ITE_true <- Y1 - Y0





# Combine all variables into a single data frame
simulated_full_data <- data.frame(ID = 1:n, Y=Y, Treatment=Tr, data, Y1, Y0, ITE_true)

# Data for testing ITE models
simulated_data <- data.frame(ID =1:n, Y=Y, Treatment=Tr, data, ITE_true = ITE_true) %>% 
  mutate(Treatment = ifelse(Treatment==1,"Y", "N")) %>% 
  mutate(Treatment = factor(Treatment, levels = c("N", "Y")))





# Check Simulated Data

# Visualize the distribution of the individual treatment effect
simulated_data %>% ggplot(aes(x=ITE_true)) +
  geom_histogram(color="gray8",bins = 100, alpha=.6) + 
  theme_minimal() + 
  xlab("True ITE") + 
  geom_vline(xintercept = 0, linetype="dashed") + coord_cartesian(xlim = c(-.5,.5))
simulated_full_data %>% ggplot(aes(x=ITE_true)) +
  geom_density(color="gray8",fill="skyblue",alpha=.6) + 
  theme_minimal() + 
  xlab("True ITE") + 
  geom_vline(xintercept = 0, linetype="dashed") + coord_cartesian(xlim = c(-.05,.1))



str(simulated_data)

# average treatment effects
mean(simulated_full_data$ITE_true)


# split into training and test
set.seed(1234)
train_index <- sample(1:nrow(simulated_data), size = 0.6*nrow(simulated_data))
train <- simulated_data[train_index, ]
test <- simulated_data[-train_index, ]

# estimate ITE_i with glm

fit1 <- glm(Y ~ Treatment * (X1 + X2 + X3 + X4) , data = train, family = binomial(link = "logit"))
summary(fit1)

# calculate ITE_i for train set
#ITE_i = P[Y=1 | do(T_i=1), X_i ] - P[Y=1 | do(T_i=0), X_i ]
# set the do-intervention (Treatment=0)
intervention_0 <- train
intervention_0$Treatment<- as.factor("N")
# set the outcome of interest (Y = 1)
intervention_0$Y<- as.factor(1)

# manipulated dataframe for do(Treatment=0)
head(intervention_0)

# set the do-intervention (Treatment=1)
intervention_1 <- train
intervention_1$Treatment<- as.factor("Y")
# set the outcome of interest (Y = 1)
intervention_1$Y<- as.factor(1)
# manipulated dataframe for do(Treatment=1)
head(intervention_1)

ITE_i <- predict(fit1, newdata = intervention_1, type = "response") - 
  predict(fit1, newdata = intervention_0, type = "response")

# histogram of ITE_i
par(mfrow=c(1,1))
hist(ITE_i, main='ITE_i', xlab='ITE_i', ylab='Frequency', breaks=50)

plot(ITE_i, train$Y, main='ITE_i vs Y_i', xlab='ITE_i', ylab='Y_i')
# fit logistic regression, stratified by Treatment T
fit_ITE <- glm(Y ~ ITE_i:Treatment, data = train, family = binomial(link = "logit"))
# make predictions for T = 0 and T = 1
range_ITE <- seq(min(ITE_i), max(ITE_i), length.out = 100)
pred_0 <- predict(fit_ITE, newdata = data.frame(ITE_i = range_ITE, Treatment = as.factor("N")), type = "response")
pred_1 <- predict(fit_ITE, newdata = data.frame(ITE_i = range_ITE, Treatment = as.factor("Y")), type = "response")
plot(ITE_i, as.numeric(train$Y), main='ITE_i vs Y_i', xlab='ITE_i', ylab='Y_i')
lines(range_ITE, pred_0, col = "red", lwd = 2)
lines(range_ITE, pred_1, col = "blue", lwd = 2)
legend("topleft", legend=c("T=0", "T=1"), col=c("red", "blue"), lty=1, cex=0.8)






# calculate ITE_i for test set



# calculate ITE_i for train set
#ITE_i = P[Y=1 | do(T_i=1), X_i ] - P[Y=1 | do(T_i=0), X_i ]
# set the do-intervention (Treatment=0)
intervention_0 <- test
intervention_0$Treatment<- as.factor("N")
# set the outcome of interest (Y = 1)
intervention_0$Y<- as.factor(1)

# manipulated dataframe for do(Treatment=0)
head(intervention_0)

# set the do-intervention (Treatment=1)
intervention_1 <- test
intervention_1$Treatment<- as.factor("Y")
# set the outcome of interest (Y = 1)
intervention_1$Y<- as.factor(1)
# manipulated dataframe for do(Treatment=1)
head(intervention_1)

ITE_i <- predict(fit1, newdata = intervention_1, type = "response") - 
  predict(fit1, newdata = intervention_0, type = "response")

# histogram of ITE_i
par(mfrow=c(1,1))
hist(ITE_i, main='ITE_i', xlab='ITE_i', ylab='Frequency', breaks=50)

plot(ITE_i, test$Y, main='ITE_i vs Y_i', xlab='ITE_i', ylab='Y_i')
# fit logistic regression, stratified by Treatment T
fit_ITE <- glm(Y ~ ITE_i + Treatment, data = test, family = binomial(link = "logit"))
summary(fit_ITE)
fit_ITE <- glm(Y ~ ITE_i:Treatment, data = test, family = binomial(link = "logit"))
summary(fit_ITE)
# make predictions for T = 0 and T = 1
range_ITE <- seq(min(ITE_i), max(ITE_i), length.out = 100)
pred_0 <- predict(fit_ITE, newdata = data.frame(ITE_i = range_ITE, Treatment = as.factor("N")), type = "response")
pred_1 <- predict(fit_ITE, newdata = data.frame(ITE_i = range_ITE, Treatment = as.factor("Y")), type = "response")
plot(ITE_i, as.numeric(test$Y), main='ITE_i vs Y_i', xlab='ITE_i', ylab='Y_i')
lines(range_ITE, pred_0, col = "red", lwd = 2)
lines(range_ITE, pred_1, col = "blue", lwd = 2)
legend("topleft", legend=c("T=0", "T=1"), col=c("red", "blue"), lty=1, cex=0.8)




































###################################################################################################


## ---- Data Preparation ------

split_data <- function(data, split){
  data.dev <- sample_frac(data, split, replace = FALSE)
  data.val <- anti_join(data, data.dev) %>% suppressMessages()
  data.dev.tx <- data.dev %>% dplyr::filter(Treatment == "Y") %>% dplyr::select(-Treatment) 
  data.dev.ct <- data.dev %>% dplyr::filter(Treatment == "N") %>% dplyr::select(-Treatment) 
  data.val.tx <- data.val %>% dplyr::filter(Treatment == "Y") %>% dplyr::select(-Treatment) 
  data.val.ct <- data.val %>% dplyr::filter(Treatment == "N") %>% dplyr::select(-Treatment)
  return(list(data.dev.tx = data.dev.tx, data.dev.ct = data.dev.ct, 
              data.val.tx = data.val.tx, data.val.ct = data.val.ct, 
              data.dev = data.dev, data.val = data.val))
}

remove_NA_data <- function(data){
  data.dev.tx <- remove_missing(data$data.dev.tx)
  data.dev.ct <- remove_missing(data$data.dev.ct)
  data.val.tx <- remove_missing(data$data.val.tx)
  data.val.ct <- remove_missing(data$data.val.ct)
  data.dev <- remove_missing(data$data.dev)
  data.val <- remove_missing(data$data.val)
  return(list(data.dev.tx = data.dev.tx, data.dev.ct = data.dev.ct, 
              data.val.tx = data.val.tx, data.val.ct = data.val.ct, 
              data.dev = data.dev, data.val = data.val))
}

## ---- Models ------

logis.ITE <- function(data, p){
  variable_names <- paste0("X", 1:p)
  form <- as.formula(paste("Y ~", paste(variable_names, collapse = " + ")))
  
  fit.dev.tx <- glm(form, data = data$data.dev.tx, family = binomial(link = "logit"))
  fit.dev.ct <- glm(form, data = data$data.dev.ct, family = binomial(link = "logit"))
  
  # Predict ITE on derivation sample
  pred.data.dev <- data$data.dev %>% dplyr::select(variable_names)
  pred.dev <- predict(fit.dev.tx, newdata = pred.data.dev, type = "response") -
    predict(fit.dev.ct, newdata = pred.data.dev, type = "response")
  
  # Predict ITE on validation sample
  pred.data.val <- data$data.val %>% dplyr::select(variable_names)
  pred.val <- predict(fit.dev.tx, newdata = pred.data.val, type = "response") -
    predict(fit.dev.ct, newdata = pred.data.val, type = "response")
  
  # generate data
  data.dev.rs <- data$data.dev %>% 
    mutate(ITE = pred.dev, RS = ifelse(ITE < 0, "benefit", "harm")) %>%
    mutate(RS = as.factor(RS))
  
  data.val.rs <- data$data.val %>% 
    mutate(ITE = pred.val, RS = ifelse(ITE < 0, "benefit", "harm")) %>%
    mutate(RS = as.factor(RS))
  
  return(list(data.dev.rs = data.dev.rs, data.val.rs = data.val.rs, 
              model.dev.tx = fit.dev.tx, model.dev.ct = fit.dev.ct))
}

## ---- Outcome-ITE plot --------
plot_outcome_ITE <- function(data.dev.rs, data.val.rs , x_lim = c(-0.5,0.5)){
  p1 <- ggplot(data=data.dev.rs, aes(x=ITE, y=Y))+
    geom_point(aes(color=Treatment))+
    geom_smooth(aes(color=Treatment, fill=Treatment), method = "glm", method.args = list(family = "binomial"), alpha=0.5)+
    coord_cartesian(xlim=x_lim, ylim = c(0,1))+
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
  
  p2 <- ggplot(data=data.val.rs, aes(x=ITE, y=Y))+
    geom_point(aes(color=Treatment))+
    geom_smooth(aes(color=Treatment, fill=Treatment), method = "glm", method.args = list(family = "binomial"))+
    coord_cartesian(xlim=x_lim, ylim = c(0,1))+
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
  
  result <- ggarrange(p1, p2, 
                      labels = c("a) Training Data", "b) Test Data"),
                      label.x = 0, 
                      label.y = 1.04,
                      hjust = 0,
                      vjust = 1,
                      ncol = 1, nrow = 2, align = "v",
                      common.legend = T)
  
  return(result)
}

## ---- ITE density plot --------
plot_ITE_density <- function(test.results, true.data = simulated_full_data){
  result <- ggplot()+
    geom_density(aes(x = test.results$data.dev.rs$ITE, fill = "ITE.dev", color = "ITE.dev"), alpha = 0.5, linewidth=1) +
    geom_density(aes(x = test.results$data.val.rs$ITE, fill = "ITE.val", color = "ITE.val"), alpha = 0.5, linewidth=1) +
    geom_density(aes(x = true.data$ITE_true, fill = "ITE.true", color = "ITE.true"), alpha = 0.1, linewidth=1) +
    #  geom_vline(aes(xintercept = mean(test.results$data.dev.rs$ITE)), 
    #             color = "orange", linetype = "dashed") +
    #  geom_vline(aes(xintercept = mean(test.results$data.val.rs$ITE)), 
    #             color = "#36648B", linetype = "dashed") +
    geom_vline(aes(xintercept = 0), color = "black", linetype = "dashed", linewidth=1) +
    xlab("Individualized Treatment Effect") +
    ylab("Density") +
    scale_color_manual(name = "Group", 
                       labels = c("ITE.dev" = "Training Data", "ITE.val" = "Test Data", "ITE.true" = "True Data"), 
                       values = c("ITE.dev"= "orange", "ITE.val" = "#36648B", "ITE.true"= "lightgreen")) +
    scale_fill_manual(name = "Group", 
                      labels = c("ITE.dev" = "Training Data", "ITE.val" = "Test Data", "ITE.true" = "True Data"), 
                      values = c("ITE.dev"= "orange", "ITE.val" = "#36648B", "ITE.true"= "lightgreen")) +
    theme_minimal()+
    theme(
      legend.position.inside = c(1, 1),
      legend.justification = c("right", "top"),
      legend.box.just = "right",
      panel.grid.major = element_blank(),  # Removes major grid lines
      panel.grid.minor = element_blank(),  # Removes minor grid lines
      panel.background = element_blank(),  # Removes panel background
      plot.background = element_blank(),    # Removes plot background (outside the plot)
      text = element_text(size = 14),
      axis.line = element_line(color = "black"),
      axis.ticks = element_line(color = "black")
    )
  return(result)
}


## ---- ITE Density Plot by Treatment and Control Groups --------
plot_ITE_density_tx_ct <- function(data = test.results$data.dev.rs){
  result <- ggplot(data = data) +
    geom_density(aes(x = ITE, fill = Treatment, color = Treatment), alpha = 0.5, linewidth=1) +
    geom_vline(aes(xintercept = 0), color = "black", linetype = "dashed", linewidth=1) +
    xlab("Individualized Treatment Effect") +
    ylab("Density") +
    scale_color_manual(name = "Group", 
                       labels = c("Y" = "Treatment", "N" = "Control"), 
                       values = c("orange", "#36648B")) +
    scale_fill_manual(name = "Group",
                      labels = c("Y" = "Treatment", "N" = "Control"), 
                      values = c("orange", "#36648B")) +
    theme_minimal()+
    theme(
      legend.position.inside = c(1, 1),
      legend.justification = c("right", "top"),
      legend.box.just = "right",
      panel.grid.major = element_blank(),  # Removes major grid lines
      panel.grid.minor = element_blank(),  # Removes minor grid lines
      panel.background = element_blank(),  # Removes panel background
      plot.background = element_blank(),    # Removes plot background (outside the plot)
      text = element_text(size = 14),
      axis.line = element_line(color = "black"),
      axis.ticks = element_line(color = "black")
    )
  
  return(result)
}




#########################################

# Data simulation

## Case 1: continuous random variables

set.seed(123)

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


# (Optional) Truncate Data

simulated_data_truncated <- simulated_data %>% 
  dplyr::select(ID, Y, Treatment, X4, X8, X9, X12, X15, X18, ITE_true) %>%
  rename(X1 = X4, X2 = X8, X3 = X9, X4 = X12, X5 = X15, X6 = X18)

str(simulated_data_truncated)




set.seed(12345)
test.data <- split_data(simulated_data_truncated, 2/3)
test.compl.data <- remove_NA_data(test.data)
test.results <- logis.ITE(test.compl.data , p=6)



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



hist(data.dev.rs$ITE)


par(mfow = c(1,2))

plot(data.dev.rs$ITE_true, data.dev.rs$ITE)
plot(data.val.rs$ITE_true, data.val.rs$ITE)



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







