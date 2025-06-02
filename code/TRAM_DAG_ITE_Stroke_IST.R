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
DIR = 'runs/TRAM_DAG_ITE_Stroke_IST/run'
# DIR = 'runs/TRAM_DAG_ITE_simulation_pt2_single_model_newDGP_CS_Relu_CI/run'
if (!dir.exists(DIR)) {
  dir.create(DIR, recursive = TRUE)
}
# Copy this file to the directory DIR
file.copy('/code/TRAM_DAG_ITE_Stroke_IST.R', file.path(DIR, 'TRAM_DAG_ITE_Stroke_IST.R'), overwrite=TRUE)
# file.copy('/code/TRAM_DAG_ITE_simulation_pt2_single_model_newDGP_CS_Relu_CI.R', file.path(DIR, 'TRAM_DAG_ITE_simulation_pt2_single_model_newDGP_CS_Relu_CI.R'), overwrite=TRUE)


len_theta = 20 # Number of coefficients of the Bernstein polynomials
hidden_features_I = c(20, 10, 10, 2) # c(3,3,3,3) 
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



# MA =  matrix(c(
#   0,   0,  0, 'cs',
#   0,   0,  0, 'cs',
#   0,   0,  0, 'ls',
#   0,   0,  0,   0), nrow = 4, ncol = 4, byrow = TRUE)


# MODEL_NAME = 'ModelCI'
# MODEL_NAME = 'ModelCIDropout0.3'
MODEL_NAME = 'ModelCIDropout0.1_standardized'



fn = file.path(DIR, paste0(MODEL_NAME))
print(paste0("Starting experiment ", fn))






###############################################################################
# ITE in a simple RCT
###############################################################################


# load data from IST trial

# named IST.dataset.clean.6m
load('data/IST_model_data.RData')

str(IST.dataset.clean.6m)



library("tableone")


tableone <- CreateTableOne(strata = "RXASP", factorVars = "OUTCOME6M", includeNA = T, data = IST.dataset.clean.6m, test = FALSE, addOverall = FALSE)
print(tableone, showAllLevels = TRUE)
summary(tableone)


split_data <- function(data, split){
  data.dev <- sample_frac(data, split, replace = FALSE)
  data.val <- anti_join(data, data.dev) %>% suppressMessages()
  data.dev.tx <- data.dev %>% dplyr::filter(RXASP == "Y") %>% dplyr::select(-RXASP) 
  data.dev.ct <- data.dev %>% dplyr::filter(RXASP == "N") %>% dplyr::select(-RXASP) 
  data.val.tx <- data.val %>% dplyr::filter(RXASP == "Y") %>% dplyr::select(-RXASP) 
  data.val.ct <- data.val %>% dplyr::filter(RXASP == "N") %>% dplyr::select(-RXASP)
  return(list(data.dev.tx = data.dev.tx, data.dev.ct = data.dev.ct, 
              data.val.tx = data.val.tx, data.val.ct = data.val.ct, 
              data.dev = data.dev, data.val = data.val))
}

# original untransformed
set.seed(12345)
test.data <- split_data(IST.dataset.clean.6m, 2/3)
test.compl.data <- remove_NA_data(test.data)


# set.seed(12345)
# test.data <- split_data(IST.dataset.clean.6m, 2/3)
# test.compl.data <- remove_NA_data(test.data)

# check datatypes
str(IST.dataset.clean.6m)

IST.dataset.clean.6m$STYPE    

# make dataframe with encoded variables (with tidyverse)

library(tidyverse)
# 
# IST.dataset.transformed <- IST.dataset.clean.6m %>%
#   # Binary variables: Encode as 0/1
#   mutate(
#     RXASP = as.numeric(RXASP == "Y"),
#     RCT = as.numeric(RCT == "Y"),
#     RVISINF = as.numeric(RVISINF == "Y"),
#     RATRIAL = as.numeric(RATRIAL == "Y"),
#     RASP3 = as.numeric(RASP3 == "Y"),
#     SEX = as.numeric(SEX == "M")
#   ) %>%
#   
#   # Dummy encode RDEF1 to RDEF8 (levels: C, N, Y)
#   mutate(across(starts_with("RDEF"), ~
#                   case_when(
#                     . == "C" ~ list(c(0, 0)),
#                     . == "N" ~ list(c(1, 0)),
#                     . == "Y" ~ list(c(0, 1)),
#                     TRUE     ~ list(c(NA_real_, NA_real_))
#                   ),
#                 .names = "{.col}_dummy"
#   )) %>%
#   unnest_wider(RDEF1_dummy, names_sep = "_") %>%
#   rename(RDEF1_1 = RDEF1_dummy_1, RDEF1_2 = RDEF1_dummy_2) %>%
#   unnest_wider(RDEF2_dummy, names_sep = "_") %>%
#   rename(RDEF2_1 = RDEF2_dummy_1, RDEF2_2 = RDEF2_dummy_2) %>%
#   unnest_wider(RDEF3_dummy, names_sep = "_") %>%
#   rename(RDEF3_1 = RDEF3_dummy_1, RDEF3_2 = RDEF3_dummy_2) %>%
#   unnest_wider(RDEF4_dummy, names_sep = "_") %>%
#   rename(RDEF4_1 = RDEF4_dummy_1, RDEF4_2 = RDEF4_dummy_2) %>%
#   unnest_wider(RDEF5_dummy, names_sep = "_") %>%
#   rename(RDEF5_1 = RDEF5_dummy_1, RDEF5_2 = RDEF5_dummy_2) %>%
#   unnest_wider(RDEF6_dummy, names_sep = "_") %>%
#   rename(RDEF6_1 = RDEF6_dummy_1, RDEF6_2 = RDEF6_dummy_2) %>%
#   unnest_wider(RDEF7_dummy, names_sep = "_") %>%
#   rename(RDEF7_1 = RDEF7_dummy_1, RDEF7_2 = RDEF7_dummy_2) %>%
#   unnest_wider(RDEF8_dummy, names_sep = "_") %>%
#   rename(RDEF8_1 = RDEF8_dummy_1, RDEF8_2 = RDEF8_dummy_2) %>%
#   
#   # Dummy encode RCONSC (levels: D, F, U)
#   mutate(RCONSC_dummy = case_when(
#     RCONSC == "D" ~ list(c(0, 0)),
#     RCONSC == "F" ~ list(c(1, 0)),
#     RCONSC == "U" ~ list(c(0, 1)),
#     TRUE          ~ list(c(NA_real_, NA_real_))
#   )) %>%
#   unnest_wider(RCONSC_dummy, names_sep = "_") %>%
#   rename(RCONSC_1 = RCONSC_dummy_1, RCONSC_2 = RCONSC_dummy_2) %>%
#   
#   # Dummy encode STYPE (5 levels: LACS OTH PACS POCS TACS)
#   mutate(STYPE = factor(STYPE)) %>%
#   bind_cols(model.matrix(~ STYPE - 1, data = .)[, -1] %>% as_tibble()) %>%
#   
#   
#   # Dummy encode REGION with 7 dummies
#   mutate(REGION = factor(REGION)) %>%
#   bind_cols(model.matrix(~ REGION - 1, data = .)[, -1] %>% as_tibble()) %>%
#   
#   # Recode OUTCOME6M to 1/2
#   mutate(OUTCOME6M = OUTCOME6M + 1) %>%
# 
#   # Remove original categorical variables
#   select(-c(RDEF1, RDEF2, RDEF3, RDEF4, RDEF5, RDEF6, RDEF7, RDEF8,
#             RCONSC, REGION, STYPE    )) %>%
#  # Move OUTCOME6M to the last column
#   select(-OUTCOME6M, everything(), OUTCOME6M)
# 



head(IST.dataset.clean.6m)
# head(IST.dataset.transformed)

# Original
print(IST.dataset.clean.6m, n = 5, width = Inf)

# Transformed
# print(IST.dataset.transformed, n = 5, width = Inf)



# transform the dataset with mode.matrix. (-1 means no intercept)
# 
# IST_complete <- drop_na(IST.dataset.clean.6m)
# # dummy_set <- cbind(as.numeric(IST_complete[, 1]=="Y"), model.matrix(~ . , data = IST_complete[, -1]))
# dummy_set <- model.matrix(~ ., data = IST_complete)[, -1]
# colnames(dummy_set)[1] <- "RXASP"
# 
# dummy_set[, "OUTCOME6M"] <- dummy_set[, "OUTCOME6M"] + 1
# 
# ncol(dummy_set)
# str(dummy_set)
# dim(dummy_set)
# dim(IST.dataset.clean.6m)
# dim(IST.dataset.transformed)
# 
# transf_set <- drop_na(IST.dataset.transformed)
# 
# # is the same
# all(dummy_set == transf_set)


IST_complete <- drop_na(IST.dataset.clean.6m)
# dummy_set <- cbind(as.numeric(IST_complete[, 1]=="Y"), model.matrix(~ . , data = IST_complete[, -1]))
IST.dataset.transformed <- model.matrix(~ ., data = IST_complete)[, -1]
colnames(IST.dataset.transformed)[1] <- "RXASP"

IST.dataset.transformed[, "OUTCOME6M"] <- IST.dataset.transformed[, "OUTCOME6M"] + 1

IST.dataset.transformed <- as.data.frame(IST.dataset.transformed)

# scale numerical variables AGE RDELAY RSBP by -mean/sd
IST.dataset.transformed <- IST.dataset.transformed %>%
  mutate(AGE = (AGE - mean(AGE, na.rm = TRUE)) / sd(AGE, na.rm = TRUE),
         RDELAY = (RDELAY - mean(RDELAY, na.rm = TRUE)) / sd(RDELAY, na.rm = TRUE),
         RSBP = (RSBP - mean(RSBP, na.rm = TRUE)) / sd(RSBP, na.rm = TRUE))


# to split transfromed dataset
split_data_transformed <- function(data, split){
  data.dev <- sample_frac(data, split, replace = FALSE)
  data.val <- anti_join(data, data.dev) %>% suppressMessages()
  data.dev.tx <- data.dev %>% dplyr::filter(RXASP == 1) %>% dplyr::select(-RXASP) 
  data.dev.ct <- data.dev %>% dplyr::filter(RXASP == 0) %>% dplyr::select(-RXASP) 
  data.val.tx <- data.val %>% dplyr::filter(RXASP == 1) %>% dplyr::select(-RXASP) 
  data.val.ct <- data.val %>% dplyr::filter(RXASP == 0) %>% dplyr::select(-RXASP)
  return(list(data.dev.tx = data.dev.tx, data.dev.ct = data.dev.ct, 
              data.val.tx = data.val.tx, data.val.ct = data.val.ct, 
              data.dev = data.dev, data.val = data.val))
}

set.seed(12345)
test.data.transformed <- split_data_transformed(IST.dataset.transformed, 2/3)

test.compl.data.transformed <- remove_NA_data(test.data.transformed)

#number of variables:
N_var <- ncol(test.compl.data.transformed$data.dev)

# Create a zero matrix
A <- matrix(0, nrow = N_var, ncol = N_var)
# Set 1s in the last column, except the last row
A[1:(N_var - 1), N_var] <- 1


# Meta adjacency

MA <- ifelse(A==1, "ci", 0)


# train set as tensor
dat.train.tf = tf$constant(as.matrix(test.compl.data.transformed$data.dev), dtype = 'float32')

# test set as tensor
dat.test.tf = tf$constant(as.matrix(test.compl.data.transformed$data.val), dtype = 'float32')

q_Outcome = c(1, 2)

min = tf$constant(1, dtype = 'float32')
max = tf$constant(2, dtype = 'float32')

type <- data_type <-  c(rep('c', N_var-1), 'o')
# 
# ##### DGP ########
# dgp <- function(n_obs=20000, doX=c(NA, NA, NA, NA), SEED=123) {
#   #n_obs = 1e5 n_obs = 10
#   set.seed(SEED)
#   
#   
#   
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
#     min = tf$constant(c(q1[1], q2[1], q3[1], q4[1]), dtype = 'float32'),
#     max = tf$constant(c(q1[2], q2[2], q3[2], q4[2]), dtype = 'float32'),
#     
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
# n_obs <- 20000
# 
# dgp_data = dgp(n_obs)
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


#################################################
# Analysis as Holly T-learner GLM
#################################################

library(splines)
logis.ITE <- function(data){
  # Train model
  form <- OUTCOME6M ~ ns(AGE, df=3) + ns(RDELAY, df=3) + ns(RSBP, df=3) + SEX + RCT + RVISINF + RATRIAL + RASP3 + RDEF1 + RDEF2 + RDEF3 + RDEF4 + RDEF5 + RDEF6 + RDEF7 + RDEF8 + RCONSC + STYPE + REGION
  fit.dev.tx <- glm(form, data = data$data.dev.tx, family = binomial(link = "logit"))
  fit.dev.ct <- glm(form, data = data$data.dev.ct, family = binomial(link = "logit"))

  # Predict ITE on derivation sample
  pred.data.dev <- data$data.dev %>% dplyr::select(-c("RXASP","OUTCOME6M"))
  y.dev.tx <- predict(fit.dev.tx, newdata = pred.data.dev, type = "response")
  y.dev.ct <- predict(fit.dev.ct, newdata = pred.data.dev, type = "response")
  pred.dev <- y.dev.tx - y.dev.ct

  # Predict ITE on validation sample
  pred.data.val <- data$data.val %>% dplyr::select(-c("RXASP","OUTCOME6M"))
  y.val.tx <- predict(fit.dev.tx, newdata = pred.data.val, type = "response")
  y.val.ct <- predict(fit.dev.ct, newdata = pred.data.val, type = "response")
  pred.val <- y.val.tx - y.val.ct


  # homogeneous model
  # form <- OUTCOME6M ~ RXASP + ns(AGE, df=3) + ns(RDELAY, df=3) + ns(RSBP, df=3) + SEX + RCT + RVISINF + RATRIAL + RASP3 + RDEF1 + RDEF2 + RDEF3 + RDEF4 + RDEF5 + RDEF6 + RDEF7 + RDEF8 + RCONSC + STYPE + REGION
  # fit.dev <- glm(form, data = data$data.dev, family = binomial(link = "logit"))
  # 
  # #train pred
  # pred.data.dev.tx <- data$data.dev %>% 
  #   dplyr::select(-c("OUTCOME6M")) %>% 
  #   # set RXASP to 1 (treatment) as factor
  #   mutate(RXASP = "Y") %>%
  #   mutate(RXASP = as.factor(RXASP))
  # pred.data.dev.ct <- data$data.dev %>% 
  #   dplyr::select(-c("OUTCOME6M")) %>% 
  #   # set RXASP to 1 (treatment) as factor
  #   mutate(RXASP = "N") %>%
  #   mutate(RXASP = as.factor(RXASP))
  # 
  # y.dev.tx <- predict(fit.dev, newdata = pred.data.dev.tx, type = "response")
  # y.dev.ct <- predict(fit.dev, newdata = pred.data.dev.ct, type = "response")
  # pred.dev <- y.dev.tx - y.dev.ct
  # 
  # #test pred
  # pred.data.val.tx <- data$data.val %>% 
  #   dplyr::select(-c("OUTCOME6M")) %>% 
  #   # set RXASP to 1 (treatment) as factor
  #   mutate(RXASP = "Y") %>%
  #   mutate(RXASP = as.factor(RXASP))
  # pred.data.val.ct <- data$data.val %>%
  #   dplyr::select(-c("OUTCOME6M")) %>% 
  #   # set RXASP to 1 (treatment) as factor
  #   mutate(RXASP = "N") %>%
  #   mutate(RXASP = as.factor(RXASP))
  # y.val.tx <- predict(fit.dev, newdata = pred.data.val.tx, type = "response")
  # y.val.ct <- predict(fit.dev, newdata = pred.data.val.ct, type = "response")
  # pred.val <- y.val.tx - y.val.ct
  # fit.dev.tx <- NULL
  # fit.dev. <- NULL
  
  
  # generate data
  data.dev.rs <- data$data.dev %>% 
    mutate(ITE = pred.dev, 
           y.tx = y.dev.tx, 
           y.ct = y.dev.ct,
           RS = ifelse(ITE < 0, "benefit", "harm")) %>%
    mutate(RS = as.factor(RS))
  
  data.val.rs <- data$data.val %>% 
    mutate(ITE = pred.val, 
           y.tx = y.val.tx,
           y.ct = y.val.ct,
           RS = ifelse(ITE < 0, "benefit", "harm")) %>%
    mutate(RS = as.factor(RS))
  
  return(list(data.dev.rs = data.dev.rs, data.val.rs = data.val.rs, 
              model.dev.tx = fit.dev.tx, model.dev.ct = fit.dev.ct))
}

test.results.logistic <- logis.ITE(test.compl.data)


data.dev.rs = test.results.logistic[["data.dev.rs"]] %>%  as.data.frame()
data.val.rs = test.results.logistic[["data.val.rs"]] %>%  as.data.frame()


#compare benefit an harm groups in the train set

table.dev.rs <- CreateTableOne(strata = "RS", factorVars = "OUTCOME6M", includeNA = T, data = data.dev.rs[,c(1:23, 25)], test = FALSE, addOverall = FALSE)
print(table.dev.rs, showAllLevels = TRUE)


# compare benefit and harm groups in the test set
table.val.rs <- CreateTableOne(strata = "RS", factorVars = "OUTCOME6M", includeNA = T, data = data.val.rs[,c(1:23, 25)], test = FALSE, addOverall = FALSE)
print(table.val.rs, showAllLevels = TRUE)


plot_outcome_ITE <- function(data.dev.rs, data.val.rs , x_lim = c(-0.5,0.5)){
  p1 <- ggplot(data=data.dev.rs, aes(x=ITE, y=OUTCOME6M))+
    geom_point(aes(color=RXASP))+
    geom_smooth(aes(color=RXASP, fill=RXASP), method = "glm", method.args = list(family = "binomial"), alpha=0.5)+
    coord_cartesian(xlim=x_lim, ylim = c(0,1))+
    ylab("Outcome")+xlab("ITE")+
    scale_color_manual(values=c("N" = "orange", "Y" = "#36648B"),labels = c("N" = "No", "Y" = "Yes") , name="Treatment") +
    scale_fill_manual(values = c("N" = "orange", "Y" = "#36648B"),labels = c("N" = "No", "Y" = "Yes"), name="Treatment") +
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
  
  p2 <- ggplot(data=data.val.rs, aes(x=ITE, y=OUTCOME6M))+
    geom_point(aes(color=RXASP))+
    geom_smooth(aes(color=RXASP, fill=RXASP), method = "glm", method.args = list(family = "binomial"))+
    coord_cartesian(xlim=x_lim, ylim = c(0,1))+
    ylab("Outcome")+xlab("ITE")+
    scale_color_manual(values=c("N" = "orange", "Y" = "#36648B"),labels = c("N" = "No", "Y" = "Yes"), name="Treatment") +
    scale_fill_manual(values = c("N" = "orange", "Y" = "#36648B"),labels = c("N" = "No", "Y" = "Yes"), name="Treatment") +
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
                      labels = c("Training Data", "Test Data"),
                      label.x = 0, 
                      label.y = 1.04,
                      hjust = 0,
                      vjust = 1,
                      ncol = 1, nrow = 2, align = "v",
                      common.legend = T)
  
  return(result)
}


library(ggpubr)
plot_outcome_ITE(data.dev.rs = data.dev.rs, data.val.rs = data.val.rs, x_lim = c(-0.5,0.5))




calc.ATE.Odds <- function(data, log.odds = T){
  data <- as.data.frame(data)
  model <- glm(OUTCOME6M ~ RXASP, data = data, family = binomial(link = "logit"))
  if (!log.odds){
    ATE.odds <- exp(coef(model)[2])
    ATE.lb <- exp(suppressMessages(confint(model)[2,1]))
    ATE.ub <- exp(suppressMessages(confint(model)[2,2]))
  } else {
    ATE.odds <- coef(model)[2]
    ATE.lb <- suppressMessages(confint(model)[2,1])
    ATE.ub <- suppressMessages(confint(model)[2,2])
  }
  return(data.frame(ATE.odds = ATE.odds, ATE.lb = ATE.lb, ATE.ub = ATE.ub, 
                    n.total = nrow(data), 
                    n.tr = sum(data$RXASP == "Y"), n.ct = sum(data$RXASP == "N")))
}


plot_ATE_ITE_in_group <- function(dev.data = data.dev.rs, val.data = data.val.rs, log.odds = T, ylb=0, yub=2){
  data <- rbind(dev.data %>% mutate(sample = "derivation"), val.data %>%  mutate(sample = "validation"))
  result <- ggplot(data, aes(x = ITE.Group, y = ATE.odds)) +
    geom_line(aes(group = sample, color = sample), linewidth = 1, 
              position = position_dodge(width = 0.2)) +
    geom_point(aes(color = sample), size = 1.5, 
               position = position_dodge(width = 0.2)) +
    geom_errorbar(aes(ymin = ATE.lb, ymax = ATE.ub, color = sample), width = 0.2,
                  position = position_dodge(width = 0.2))+
    geom_hline(yintercept = ifelse(log.odds, 0, 1), linetype = "dashed", color = "black") +
    scale_color_manual(name = "Group",
                       labels = c("derivation" = "Training Data", "validation" = "Test Data"),
                       values = c("orange", "#36648B"))+
    scale_x_discrete(guide = guide_axis(n.dodge = 2))+
    ylim(ylb,yub)+
    xlab("ITE Group")+
    ylab(ifelse(log.odds, "ATE in Log Odds Ratio", "ATE in Odds Ratio"))+
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


breaks <- c(-0.3, -0.05, -0.025, 0.025, 0.05, 0.1, 0.3)
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

plot_ATE_ITE_in_group(dev.data = data.dev.grouped.ATE, val.data = data.val.grouped.ATE, log.odds = log.odds, ylb = 0, yub = 6)


# 
# #check predictive power of model (GLM with interaction term):
# 
# # train
# 
# df <- dgp_data$test.compl.data
# 
# fit_train <- glm(Y ~ Tr + X1 + X2 + Tr:X1, data = df$data.dev, family = binomial(link="logit")) # glm for binary (negative shift
# 
# 
# pred_train <- predict(fit_train, newdata = df$data.dev, type = "response")
# pred_test <- predict(fit_train, newdata = df$data.val, type = "response")
# 
# df$data.dev$Ypred <- pred_train
# df$data.val$Ypred <- pred_test
# 
# 
# 
# # train
# ggplot(df$data.dev, aes(x = Y_prob, y = Ypred, color = as.factor(Tr))) +
#   geom_point() +
#   geom_abline(slope = 1, intercept = 0, color = "red") +
#   labs(x = "True Probabilities", y = "Estimated Probabilities", title = "Prob GLM (Train)") +
#   theme_minimal() +
#   theme(legend.position = "top")
# 
# # test
# ggplot(df$data.val, aes(x = Y_prob, y = Ypred, color = as.factor(Tr))) +
#   geom_point() +
#   geom_abline(slope = 1, intercept = 0, color = "red") +
#   labs(x = "True Probabilities", y = "Estimated Probabilities", title = "Prob GLM (Test)") +
#   theme_minimal() +
#   theme(legend.position = "top")
# 
# 




#################################################
# fit TRAM-DAG wit CS(T, X1) 
#################################################

# (global_min = min)
# (global_max = min)
# data_type = dgp_data$type

# len_theta_max = len_theta
# for (i in 1:nrow(MA)){ #Maximum number of coefficients (BS and Levels - 1 for the ordinal)
#   if (dgp_data$type[i] == 'o'){
#     len_theta_max = max(len_theta_max, nlevels(dgp_data$df_R[,i]) - 1)
#   }
# }



#################################################
# Tuned Random Forest for ITE
#################################################


library(comets)
library(dplyr)

# extract the tuned_rf function
comets_tuned_rf <- comets:::tuned_rf

fit.tuned_rf_IST <- function(df, p = N_var-2) {
  # p <- sum(grepl("^X", colnames(df$data.dev)))
  # df <- test.compl.data.transformed
  variable_names <- colnames(df$data.dev)[2:(p+1)] # Exclude RXASP and OUTCOME6M from the variable names
  # form <- as.formula(paste("OUTCOME6M ~", paste(variable_names, collapse = " + ")))
  
  df$data.dev.tx$OUTCOME6M <- as.factor(df$data.dev.tx$OUTCOME6M -1)  # Ensure Y is a factor for classification
  df$data.dev.ct$OUTCOME6M <- as.factor(df$data.dev.ct$OUTCOME6M -1)  # Ensure Y is a factor for classification
  
  # Fit random forest for treatment and control groups
  fit.dev.tx <- comets_tuned_rf(y=as.matrix(df$data.dev.tx$OUTCOME6M), x=as.matrix(df$data.dev.tx %>% dplyr::select(variable_names)))
  fit.dev.ct <- comets_tuned_rf(y=as.matrix(df$data.dev.ct$OUTCOME6M), x=as.matrix(df$data.dev.ct %>% dplyr::select(variable_names)))
  
  # Feature set of derivation sample
  X_dev <- as.matrix(df$data.dev %>% dplyr::select(variable_names))
  
  # Predict probabilities on derivation sample
  pred_tx_dev <- predict(fit.dev.tx, data = X_dev)
  pred_ct_dev <- predict(fit.dev.ct, data = X_dev)
  
  # Predict outcome for observed T and X on derivation sample
  df$data.dev$Y_pred <- pred_tx_dev * df$data.dev$RXASP + pred_ct_dev * (1 - df$data.dev$RXASP)
  
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
  df$data.val$Y_pred <- pred_tx_val * df$data.val$RXASP + pred_ct_val * (1 - df$data.val$RXASP)
  
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
  
  return(list(data.dev.rs = data.dev.rs, data.val.rs = data.val.rs
              # , model.dev.tx = fit.dev.tx, model.dev.ct = fit.dev.ct
              ))
}







plot_outcome_ITE <- function(data.dev.rs, data.val.rs , x_lim = c(-0.5,0.5)){
  
  ### debugging
  # data.dev.rs <- test.results.tram$data.dev.rs
  # data.val.rs <- test.results.tram$data.val.rs
  
  
  # transform outcome back to 0, 1 coding
  data.dev.rs$OUTCOME6M <- data.dev.rs$OUTCOME6M -1
  data.val.rs$OUTCOME6M <- data.val.rs$OUTCOME6M -1
  
  # data.val.rs$ITE
  # data.val.rs$RXASP
  
  # encode RXASP as factor N, Y
  data.dev.rs$RXASP <- factor(data.dev.rs$RXASP, levels = c(0, 1), labels = c("N", "Y"))
  data.val.rs$RXASP <- factor(data.val.rs$RXASP, levels = c(0, 1), labels = c("N", "Y"))
  
  p1 <- ggplot(data=data.dev.rs, aes(x=ITE, y=OUTCOME6M))+
    geom_point(aes(color=RXASP))+
    geom_smooth(aes(color=RXASP, fill=RXASP), method = "glm", method.args = list(family = "binomial"), alpha=0.5)+
    coord_cartesian(xlim=x_lim, ylim = c(0,1))+
    ylab("Outcome")+xlab("ITE")+
    scale_color_manual(values=c("N" = "orange", "Y" = "#36648B"),labels = c("N" = "No", "Y" = "Yes") , name="Treatment") +
    scale_fill_manual(values = c("N" = "orange", "Y" = "#36648B"),labels = c("N" = "No", "Y" = "Yes"), name="Treatment") +
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
  
  p2 <- ggplot(data=data.val.rs, aes(x=ITE, y=OUTCOME6M))+
    geom_point(aes(color=RXASP))+
    geom_smooth(aes(color=RXASP, fill=RXASP), method = "glm", method.args = list(family = "binomial"))+
    coord_cartesian(xlim=x_lim, ylim = c(0,1))+
    ylab("Outcome")+xlab("ITE")+
    scale_color_manual(values=c("N" = "orange", "Y" = "#36648B"),labels = c("N" = "No", "Y" = "Yes"), name="Treatment") +
    scale_fill_manual(values = c("N" = "orange", "Y" = "#36648B"),labels = c("N" = "No", "Y" = "Yes"), name="Treatment") +
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
                      labels = c("Training Data", "Test Data"),
                      label.x = 0,
                      label.y = 1.04,
                      hjust = 0,
                      vjust = 1,
                      ncol = 1, nrow = 2, align = "v",
                      common.legend = T)
  
  return(result)
}

# plot the results
tuned_rf.results <- fit.tuned_rf_IST(test.compl.data.transformed)

tuned_rf.results$data.dev.rs$ITE
tuned_rf.results$data.dev.rs$OUTCOME6M
tuned_rf.results$data.dev.rs$Y_pred



plot_outcome_ITE(data.dev.rs = tuned_rf.results$data.dev.rs, data.val.rs = tuned_rf.results$data.val.rs, x_lim = c(-0.5,0.5))



plot_ITE_density <- function(test.results){
  result <- ggplot()+
    geom_density(aes(x = test.results$data.dev.rs$ITE, fill = "ITE.dev", color = "ITE.dev"), alpha = 0.5, linewidth=1) +
    geom_density(aes(x = test.results$data.val.rs$ITE, fill = "ITE.val", color = "ITE.val"), alpha = 0.5, linewidth=1) +
    #  geom_vline(aes(xintercept = mean(test.results$data.dev.rs$ITE)), 
    #             color = "orange", linetype = "dashed") +
    #  geom_vline(aes(xintercept = mean(test.results$data.val.rs$ITE)), 
    #             color = "#36648B", linetype = "dashed") +
    geom_vline(aes(xintercept = 0), color = "black", linetype = "dashed", linewidth=1) +
    xlab("Individualized Treatment Effect") +
    ylab("Density") +
    scale_color_manual(name = "Group", 
                       labels = c("ITE.dev" = "Training Data", "ITE.val" = "Test Data"), 
                       values = c("orange", "#36648B")) +
    scale_fill_manual(name = "Group", 
                      labels = c("ITE.dev" = "Training Data", "ITE.val" = "Test Data"), 
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



plot_ITE_density(tuned_rf.results)





calc.ATE.Odds.tram <- function(data, log.odds = T){
  data <- as.data.frame(data)
  data$OUTCOME6M <- data$OUTCOME6M - 1 #transform back to 0, 1 coding
  model <- glm(OUTCOME6M ~ RXASP, data = data, family = binomial(link = "logit"))
  if (!log.odds){
    ATE.odds <- exp(coef(model)[2])
    ATE.lb <- exp(suppressMessages(confint(model)[2,1]))
    ATE.ub <- exp(suppressMessages(confint(model)[2,2]))
  } else {
    ATE.odds <- coef(model)[2]
    ATE.lb <- suppressMessages(confint(model)[2,1])
    ATE.ub <- suppressMessages(confint(model)[2,2])
  }
  return(data.frame(ATE.odds = ATE.odds, ATE.lb = ATE.lb, ATE.ub = ATE.ub, 
                    n.total = nrow(data), 
                    n.tr = sum(data$RXASP == 1), n.ct = sum(data$RXASP == 0)))
}


plot_ATE_ITE_in_group <- function(dev.data = data.dev.rs, val.data = data.val.rs, log.odds = T, ylb=0, yub=2){
  data <- rbind(dev.data %>% mutate(sample = "derivation"), val.data %>%  mutate(sample = "validation"))
  result <- ggplot(data, aes(x = ITE.Group, y = ATE.odds)) +
    geom_line(aes(group = sample, color = sample), linewidth = 1, 
              position = position_dodge(width = 0.2)) +
    geom_point(aes(color = sample), size = 1.5, 
               position = position_dodge(width = 0.2)) +
    geom_errorbar(aes(ymin = ATE.lb, ymax = ATE.ub, color = sample), width = 0.2,
                  position = position_dodge(width = 0.2))+
    geom_hline(yintercept = ifelse(log.odds, 0, 1), linetype = "dashed", color = "black") +
    scale_color_manual(name = "Group",
                       labels = c("derivation" = "Training Data", "validation" = "Test Data"),
                       values = c("orange", "#36648B"))+
    scale_x_discrete(guide = guide_axis(n.dodge = 2))+
    ylim(ylb,yub)+
    xlab("ITE Group")+
    ylab(ifelse(log.odds, "ATE in Log Odds Ratio", "ATE in Odds Ratio"))+
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


# breaks <- c(-0.6, -0.2, -0.05, 0.025, 0.05, 0.1, 0.6)
breaks <- c(-0.3, -0.1, -0.05, 0.025, 0.05, 0.3)
log.odds <- F
data.dev.grouped.ATE <- tuned_rf.results$data.dev.rs %>% 
  mutate(ITE.Group = cut(ITE, breaks = breaks, include.lowest = T)) %>%
  dplyr::filter(!is.na(ITE.Group)) %>%
  group_by(ITE.Group) %>% 
  group_modify(~ calc.ATE.Odds.tram(.x, log.odds = log.odds)) %>% ungroup()
data.val.grouped.ATE <- tuned_rf.results$data.val.rs %>% 
  mutate(ITE.Group = cut(ITE, breaks = breaks, include.lowest = T)) %>%
  dplyr::filter(!is.na(ITE.Group)) %>%
  group_by(ITE.Group) %>%
  group_modify(~ calc.ATE.Odds.tram(.x, log.odds = log.odds)) %>% ungroup() 

plot_ATE_ITE_in_group(dev.data = data.dev.grouped.ATE, val.data = data.val.grouped.ATE, log.odds = log.odds, ylb = 0, yub = 7.5)





#same for risk difference

calc.ATE.Risks.tram <- function(data) {
  data <- as.data.frame(data)
  data$OUTCOME6M <- data$OUTCOME6M - 1  # transform to 0/1
  
  # Ensure RXASP is a factor
  # data$RXASP <- factor(data$RXASP, levels = c("N", "Y"))
  
  # Calculate proportions
  p1 <- mean(data$OUTCOME6M[data$RXASP ==  1])  # treated
  p0 <- mean(data$OUTCOME6M[data$RXASP == 0])  # control
  
  # Risk difference
  ATE.RiskDiff <- p1 - p0
  
  # Standard error for RD (Wald)
  n1 <- sum(data$RXASP == 1)
  n0 <- sum(data$RXASP == 0)
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
    ylim(ylb,yub)+
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


# breaks <- c(-0.6, -0.2, -0.05, 0.025, 0.05, 0.1, 0.6)
breaks <- c(-0.3, -0.1, -0.05, 0.025, 0.05, 0.3)
log.odds <- F
data.dev.grouped.ATE <- tuned_rf.results$data.dev.rs %>% 
  mutate(ITE.Group = cut(ITE, breaks = breaks, include.lowest = T)) %>%
  dplyr::filter(!is.na(ITE.Group)) %>%
  group_by(ITE.Group) %>% 
  group_modify(~ calc.ATE.Risks.tram(.x)) %>% ungroup()
data.val.grouped.ATE <- tuned_rf.results$data.val.rs %>% 
  mutate(ITE.Group = cut(ITE, breaks = breaks, include.lowest = T)) %>%
  dplyr::filter(!is.na(ITE.Group)) %>%
  group_by(ITE.Group) %>%
  group_modify(~ calc.ATE.Risks.tram(.x)) %>% ungroup() 

plot_ATE_ITE_in_group_risks(dev.data = data.dev.grouped.ATE, val.data = data.val.grouped.ATE, ylb = -0.5, yub = 0.5)





### Check calibration on the training set:



Y_prob_tuned_rf <- tuned_rf.results$data.dev.rs$Y_pred


# create train df with prediction and true prob

train_df <- data.frame(
  Y_prob_tuned_rf = Y_prob_tuned_rf,
  Y = as.numeric(tuned_rf.results$data.dev.rs$OUTCOME6M-1)
)

# Recalibrate with GAM

library(dplyr)
library(mgcv)
library(binom)
library(ggplot2)

# Set confidence level and bins
my_conf <- 0.95
bins <- 10

# Create equal-frequency bins
train_df <- train_df %>%
  mutate(prob_bin = cut(
    Y_prob_tuned_rf,
    # breaks = quantile(Y_prob_tram_dag, probs = seq(0, 1, length.out = bins + 1), na.rm = TRUE),
    breaks = seq(min(Y_prob_tuned_rf), max(Y_prob_tuned_rf), length.out = bins + 1),
    include.lowest = TRUE
  ))

# Compute bin summaries
agg_bin <- train_df %>%
  group_by(prob_bin) %>%
  summarise(
    pred_probability = mean(Y_prob_tuned_rf),
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
    title = paste("Calibration Plot (Train) (", bins, " bins)", sep = ""),
    x = "Predicted Probability",
    y = "Observed Proportion"
  ) +
  coord_equal() +
  theme_minimal()


# roc curve

library(pROC)

# Create ROC object
roc_obj <- roc(train_df$Y, train_df$Y_prob_tuned_rf)

# Plot ROC curve
plot(roc_obj, col = "#2c7fb8", lwd = 2, main = "ROC Curve - Train Set")

# Optional: Add AUC to the plot
auc_value <- auc(roc_obj)
legend("bottomright", legend = paste("AUC =", round(auc_value, 3)), col = "#2c7fb8", lwd = 2)





### Check calibration on the test set:


Y_prob_tuned_rf <- tuned_rf.results$data.val.rs$Y_pred


# create test df with prediction and true prob

test_df <- data.frame(
  Y_prob_tuned_rf = Y_prob_tuned_rf,
  Y = as.numeric(tuned_rf.results$data.val.rs$OUTCOME6M-1)
)

# Recalibrate with GAM

library(dplyr)
library(mgcv)
library(binom)
library(ggplot2)

# Set confidence level and bins
my_conf <- 0.95
bins <- 10

# Create equal-frequency bins
test_df <- test_df %>%
  mutate(prob_bin = cut(
    Y_prob_tuned_rf,
    # breaks = quantile(Y_prob_tram_dag, probs = seq(0, 1, length.out = bins + 1), na.rm = TRUE),
    breaks = seq(min(Y_prob_tuned_rf), max(Y_prob_tuned_rf), length.out = bins + 1),
    include.lowest = TRUE
  ))

# Compute bin summaries
agg_bin <- test_df %>%
  group_by(prob_bin) %>%
  summarise(
    pred_probability = mean(Y_prob_tuned_rf),
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
    title = paste("Calibration Plot (Test) (", bins, " bins)", sep = ""),
    x = "Predicted Probability",
    y = "Observed Proportion"
  ) +
  coord_equal() +
  theme_minimal()



# roc curve

library(pROC)

# Create ROC object
roc_obj <- roc(test_df$Y, test_df$Y_prob_tuned_rf)

# Plot ROC curve
plot(roc_obj, col = "#2c7fb8", lwd = 2, main = "ROC Curve - Test Set")

# Optional: Add AUC to the plot
auc_value <- auc(roc_obj)
legend("bottomright", legend = paste("AUC =", round(auc_value, 3)), col = "#2c7fb8", lwd = 2)
























##### Train on control group ####

param_model = create_param_model(MA, hidden_features_I=hidden_features_I, len_theta=len_theta, hidden_features_CS=hidden_features_CS)
optimizer = optimizer_adam(learning_rate = 0.001)
param_model$compile(optimizer, loss=struct_dag_loss_ITE_IST)

h_params <- param_model(dat.train.tf)

param_model$evaluate(x = dat.train.tf, y=dat.train.tf, batch_size = 7L)
summary(param_model)

# show activation function activation_68 --> Relu is used (before it was sigmoid)
param_model$get_layer("activation_48")$get_config()

##### Training ####

# num_epochs <- 1000
num_epochs <- 300   ### 200 is the file


# Split dat.train.tf into training and validation sets
set.seed(42)  # for reproducibility
n <- nrow(dat.train.tf)
val_idx <- sample(0:(n - 1), size = floor(0.2 * n))  
train_idx <- setdiff(0:(n - 1), val_idx)

# Convert to Tensor
val_idx_tf <- tf$constant(val_idx, dtype = tf$int32)
train_idx_tf <- tf$constant(train_idx, dtype = tf$int32)

# Use tf.gather
x_val   <- tf$gather(dat.train.tf, val_idx_tf)
dat.train.tf <- tf$gather(dat.train.tf, train_idx_tf)

fnh5 = paste0(fn, '_E', num_epochs, 'early_stopping_CI.h5')   # 'CI.h5'
fnRdata = paste0(fn, '_E', num_epochs, 'early_stopping_CI.RData')   # 'CI.RData'

if (file.exists(fnh5)) {
  param_model$load_weights(fnh5)
  load(fnRdata)
  (global_min = min)
  (global_max = min)
} else {
  if (FALSE) { ### Full Training w/o diagnostics
    hist = param_model$fit(x = x_train, y = x_train, epochs = 200L, verbose = TRUE,
                           validation_data = list(x_val, x_val))
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
        x = x_train, y = x_train,
        epochs = 1L, verbose = TRUE,
        validation_data = list(x_val, x_val)
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




# 
# fnh5 = paste0(fn, '_E', num_epochs, 'CI.h5')   # 'CI.h5'
# fnRdata = paste0(fn, '_E', num_epochs, 'CI.RData')   # 'CI.RData'
# if (file.exists(fnh5)){
#   param_model$load_weights(fnh5)
#   load(fnRdata) #Loading of the workspace causes trouble e.g. param_model is zero
#   # Quick Fix since loading global_min causes problem (no tensors as RDS)
#   (global_min = min)
#   (global_max = min)
# } else {
#   if (FALSE){ ### Full Training w/o diagnostics
#     hist = param_model$fit(x = dat.train.tf, y=dat.train.tf, epochs = 200L,verbose = TRUE)
#     param_model$save_weights(fn)
#     plot(hist$epoch, hist$history$loss)
#     plot(hist$epoch, hist$history$loss, ylim=c(1.07, 1.2))
#   } else { ### Training with diagnostics
#     # ws <- data.frame(w12 = numeric())
#     # ws <- data.frame(w34 = numeric())
#     train_loss <- numeric()
#     val_loss <- numeric()
#     
#     # Training loop
#     for (e in 1:num_epochs) {
#       print(paste("Epoch", e))
#       hist <- param_model$fit(x = dat.train.tf, y = dat.train.tf, 
#                               epochs = 1L, verbose = TRUE, 
#                               validation_data = list(dat.test.tf, dat.test.tf))
#       
#       # Append losses to history
#       train_loss <- c(train_loss, hist$history$loss)
#       val_loss <- c(val_loss, hist$history$val_loss)
#       
#       # Extract specific weights
#       # w <- param_model$get_layer(name = "beta")$get_weights()[[1]]
#       
#       # ws <- rbind(ws, data.frame(w34 = w[3,4]))
#     }
#     
#     
#     # Save the model
#     param_model$save_weights(fnh5)
#     save(train_loss, val_loss, train_loss, 
#          # f,
#          MA, len_theta,
#          hidden_features_I,
#          hidden_features_CS,
#          # ws,
#          #global_min, global_max,
#          file = fnRdata)
#   }
# }


par(mfrow=c(1,1))
epochs = length(train_loss)
plot(1:length(train_loss), train_loss, type='l', main='Normal Training (green is valid)', ylim = c(0.4, 0.8))
lines(1:length(train_loss), val_loss, type = 'l', col = 'green')


# Last XX epochs
last <- 300
diff = max(epochs - last,0)
plot(diff:epochs, val_loss[diff:epochs], type = 'l', col = 'green', main=paste0('Last ' ,last  ,' epochs'))
lines(diff:epochs, train_loss[diff:epochs], type='l')









# # learned weights for linear Shift
# param_model$get_layer(name = "beta")$get_weights()[[1]] * param_model$get_layer(name = "beta")$mask
# 
# # Weight estimates by glm()
# # fit_321 <- glm(x3 ~ x1 + x2, data = dgp_data$df_R_train_ct, family = binomial(link="logit")) # glm for binary (negative shift)
# 
# 
# p <- ggplot(ws, aes(x=1:nrow(ws))) + 
#   geom_line(aes(y=w34, color='x2 --> Y')) + 
#   # geom_line(aes(y=w23, color='x2 --> x3')) + 
#   # geom_hline(aes(yintercept=-coef(fit_321)[2], color='glm'), linetype=2) +
#   # geom_hline(aes(yintercept=-coef(fit_321)[3], color='glm'), linetype=2) +
#   #scale_color_manual(values=c('x1 --> x2'='skyblue', 'x1 --> x3='red', 'x2 --> x3'='darkgreen')) +
#   labs(x='Epoch', y='Coefficients') +
#   theme_minimal() +
#   theme(legend.title = element_blank())  # Removes the legend title
# 
# p










#################################################
# calculate ITE_i for train and test set
#################################################



do_probability = function (h_params){
  #t_i = intervention_0_tf # (40000, 3)    # original data x1, x2, x3 for each obs
  #h_params = h_params_orig                 # NN outputs (CS, LS, theta') for each obs
  # k_min <- k_constant(global_min)
  # k_max <- k_constant(global_max)
  
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
      nol = 1 # Number of cut-points in respective dimension
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


################################################################################

##### Check predictive power of the model on train set


h_params_orig <- param_model(dat.train.tf)
Y_prob_tram_dag <- as.numeric(do_probability(h_params_orig))

sqrt(mean((as.numeric(dat.train.tf[,39]-1) - Y_prob_tram_dag)^2))




##### Check predictive power of the model on test set

# input test data into model
h_params_orig <- param_model(dat.test.tf)

# probabilities for Y=1 on original test data
Y_prob_tram_dag <- as.numeric(do_probability(h_params_orig))

sqrt(mean((as.numeric(dat.test.tf[,39]-1) - Y_prob_tram_dag)^2))









### Check calibration on the training set:


h_params_orig <- param_model(dat.train.tf)
Y_prob_tram_dag <- as.numeric(do_probability(h_params_orig))


# create train df with prediction and true prob

train_df <- data.frame(
  Y_prob_tram_dag = Y_prob_tram_dag,
  Y = as.numeric(dat.train.tf[,39]-1)
)
  
# Recalibrate with GAM

library(dplyr)
library(mgcv)
library(binom)
library(ggplot2)

# Set confidence level and bins
my_conf <- 0.95
bins <- 10

# Create equal-frequency bins
train_df <- train_df %>%
  mutate(prob_bin = cut(
    Y_prob_tram_dag,
    # breaks = quantile(Y_prob_tram_dag, probs = seq(0, 1, length.out = bins + 1), na.rm = TRUE),
    breaks = seq(min(Y_prob_tram_dag), max(Y_prob_tram_dag), length.out = bins + 1),
    include.lowest = TRUE
  ))

# Compute bin summaries
agg_bin <- train_df %>%
  group_by(prob_bin) %>%
  summarise(
    pred_probability = mean(Y_prob_tram_dag),
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
    title = paste("Calibration Plot (Train) (", bins, " bins)", sep = ""),
    x = "Predicted Probability",
    y = "Observed Proportion"
  ) +
  coord_equal() +
  theme_minimal()


# roc curve

library(pROC)

# Create ROC object
roc_obj <- roc(train_df$Y, train_df$Y_prob_tram_dag)

# Plot ROC curve
plot(roc_obj, col = "#2c7fb8", lwd = 2, main = "ROC Curve - Train Set")

# Optional: Add AUC to the plot
auc_value <- auc(roc_obj)
legend("bottomright", legend = paste("AUC =", round(auc_value, 3)), col = "#2c7fb8", lwd = 2)




# do calibration (directly on the train set, which is probably problematic)

# weighted
# raw_w <- 1 / agg_bin$width_CI
# agg_bin$weights <- raw_w / sum(raw_w)
# 
# library(mgcv)
# fit_gam <- gam(obs_proportion ~ s(pred_probability), weights = weights, data = agg_bin, gamma = 0.3)
# # fit_gam <- gam(obs_proportion ~ s(pred_probability), data = agg_bin, gamma = 1) # without weights
# plot(fit_gam)
# 
# b0 <- fit_gam[[1]][1]
# abline(0-b0,1)








### Check calibration on the test set:


h_params_orig <- param_model(dat.test.tf)
Y_prob_tram_dag <- as.numeric(do_probability(h_params_orig))


# create test df with prediction and true prob

test_df <- data.frame(
  Y_prob_tram_dag = Y_prob_tram_dag,
  Y = as.numeric(dat.test.tf[,39]-1)
)

# Recalibrate with GAM

library(dplyr)
library(mgcv)
library(binom)
library(ggplot2)

# Set confidence level and bins
my_conf <- 0.95
bins <- 10

# Create equal-frequency bins
test_df <- test_df %>%
  mutate(prob_bin = cut(
    Y_prob_tram_dag,
    # breaks = quantile(Y_prob_tram_dag, probs = seq(0, 1, length.out = bins + 1), na.rm = TRUE),
    breaks = seq(min(Y_prob_tram_dag), max(Y_prob_tram_dag), length.out = bins + 1),
    include.lowest = TRUE
  ))

# Compute bin summaries
agg_bin <- test_df %>%
  group_by(prob_bin) %>%
  summarise(
    pred_probability = mean(Y_prob_tram_dag),
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
    title = paste("Calibration Plot (Test) (", bins, " bins)", sep = ""),
    x = "Predicted Probability",
    y = "Observed Proportion"
  ) +
  coord_equal() +
  theme_minimal()



# roc curve

library(pROC)

# Create ROC object
roc_obj <- roc(test_df$Y, test_df$Y_prob_tram_dag)

# Plot ROC curve
plot(roc_obj, col = "#2c7fb8", lwd = 2, main = "ROC Curve - Test Set")

# Optional: Add AUC to the plot
auc_value <- auc(roc_obj)
legend("bottomright", legend = paste("AUC =", round(auc_value, 3)), col = "#2c7fb8", lwd = 2)



















################################################

# Calculate ITE for train and test set


# param_model has to be trained first
tram.ITE <- function(data = test.compl.data.transformed, train = dat.train.tf, test = dat.test.tf){
  
  
  # Train set
  
  # Prepare Data
  # set the values of the first column of train to 0
  train_ct <- tf$concat(list(tf$zeros_like(train[, 1, drop = FALSE]), train[, 2:tf$shape(train)[2]]), axis = 1L)
  # set the values of the first column of train to 1
  train_tx <- tf$concat(list(tf$ones_like(train[, 1, drop = FALSE]), train[, 2:tf$shape(train)[2]]), axis = 1L)
  
  # Calculate potential outcomes
  h_params_ct <- param_model(train_ct)
  y_train_ct <- as.numeric(do_probability(h_params_ct))
  
  h_params_tx <- param_model(train_tx)
  y_train_tx <- as.numeric(do_probability(h_params_tx))
  
  # calculate ITE
  ITE_train <- y_train_tx - y_train_ct
  
  
  # Test set
  
  # Prepare Data
  # set the values of the first column of test to 0
  test_ct <- tf$concat(list(tf$zeros_like(test[, 1, drop = FALSE]), test[, 2:tf$shape(test)[2]]), axis = 1L)
  # set the values of the first column of test to 1
  test_tx <- tf$concat(list(tf$ones_like(test[, 1, drop = FALSE]), test[, 2:tf$shape(test)[2]]), axis = 1L)
  
  
  
  # Calculate potential outcomes
  h_params_ct <- param_model(test_ct)
  y_test_ct <- as.numeric(do_probability(h_params_ct))
  # y_test_ct <- predict(fit_gam, newdata = data.frame(pred_probability = y_test_ct), type = "response") # if recalibrate
  
  h_params_tx <- param_model(test_tx)
  y_test_tx <- as.numeric(do_probability(h_params_tx))
  # y_test_tx <- predict(fit_gam, newdata = data.frame(pred_probability = y_test_tx), type = "response") # if recalibrate
  
  # calculate ITE
  ITE_test <- y_test_tx - y_test_ct
  
  
  
  # generate data
  data.dev.rs <- data$data.dev %>% 
    mutate(ITE = ITE_train, 
           y.tx = y_train_tx, 
           y.ct = y_train_ct,
           RS = ifelse(ITE < 0, "benefit", "harm")) %>%
    mutate(RS = as.factor(RS))
  
  data.val.rs <- data$data.val %>% 
    mutate(ITE = ITE_test,, 
           y.tx = y_test_tx,
           y.ct = y_test_ct,
           RS = ifelse(ITE < 0, "benefit", "harm")) %>%
    mutate(RS = as.factor(RS))
  
  return(list(data.dev.rs = data.dev.rs, data.val.rs = data.val.rs))
}





# call function
test.results.tram <- tram.ITE(data = test.compl.data.transformed, train = dat.train.tf, test = dat.test.tf)



# ATE Estimated
mean(test.results.tram$data.dev.rs$ITE)
mean(test.results.tram$data.val.rs$ITE)

# ATE Observed
df <- test.results.tram   
mean(df$data.dev.rs$OUTCOME6M[df$data.dev.rs$RXASP == 1]-1) - mean(df$data.dev.rs$OUTCOME6M[df$data.dev.rs$RXASP == 0]-1)
mean(df$data.val.rs$OUTCOME6M[df$data.val.rs$RXASP == 1]-1) - mean(df$data.val.rs$OUTCOME6M[df$data.val.rs$RXASP == 0]-1)



plot_ITE_density <- function(test.results){
  result <- ggplot()+
    geom_density(aes(x = test.results$data.dev.rs$ITE, fill = "ITE.dev", color = "ITE.dev"), alpha = 0.5, linewidth=1) +
    geom_density(aes(x = test.results$data.val.rs$ITE, fill = "ITE.val", color = "ITE.val"), alpha = 0.5, linewidth=1) +
    #  geom_vline(aes(xintercept = mean(test.results$data.dev.rs$ITE)), 
    #             color = "orange", linetype = "dashed") +
    #  geom_vline(aes(xintercept = mean(test.results$data.val.rs$ITE)), 
    #             color = "#36648B", linetype = "dashed") +
    geom_vline(aes(xintercept = 0), color = "black", linetype = "dashed", linewidth=1) +
    xlab("Individualized Treatment Effect") +
    ylab("Density") +
    scale_color_manual(name = "Group", 
                       labels = c("ITE.dev" = "Training Data", "ITE.val" = "Test Data"), 
                       values = c("orange", "#36648B")) +
    scale_fill_manual(name = "Group", 
                      labels = c("ITE.dev" = "Training Data", "ITE.val" = "Test Data"), 
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



plot_ITE_density(test.results.tram)




plot_outcome_ITE <- function(data.dev.rs, data.val.rs , x_lim = c(-0.5,0.5)){
  
  ### debugging
  # data.dev.rs <- test.results.tram$data.dev.rs
  # data.val.rs <- test.results.tram$data.val.rs
  
  
  # transform outcome back to 0, 1 coding
  data.dev.rs$OUTCOME6M <- data.dev.rs$OUTCOME6M -1
  data.val.rs$OUTCOME6M <- data.val.rs$OUTCOME6M -1
  
  # data.val.rs$ITE
  # data.val.rs$RXASP
  
  # encode RXASP as factor N, Y
  data.dev.rs$RXASP <- factor(data.dev.rs$RXASP, levels = c(0, 1), labels = c("N", "Y"))
  data.val.rs$RXASP <- factor(data.val.rs$RXASP, levels = c(0, 1), labels = c("N", "Y"))
  
  p1 <- ggplot(data=data.dev.rs, aes(x=ITE, y=OUTCOME6M))+
    geom_point(aes(color=RXASP))+
    geom_smooth(aes(color=RXASP, fill=RXASP), method = "glm", method.args = list(family = "binomial"), alpha=0.5)+
    coord_cartesian(xlim=x_lim, ylim = c(0,1))+
    ylab("Outcome")+xlab("ITE")+
    scale_color_manual(values=c("N" = "orange", "Y" = "#36648B"),labels = c("N" = "No", "Y" = "Yes") , name="Treatment") +
    scale_fill_manual(values = c("N" = "orange", "Y" = "#36648B"),labels = c("N" = "No", "Y" = "Yes"), name="Treatment") +
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

  p2 <- ggplot(data=data.val.rs, aes(x=ITE, y=OUTCOME6M))+
    geom_point(aes(color=RXASP))+
    geom_smooth(aes(color=RXASP, fill=RXASP), method = "glm", method.args = list(family = "binomial"))+
    coord_cartesian(xlim=x_lim, ylim = c(0,1))+
    ylab("Outcome")+xlab("ITE")+
    scale_color_manual(values=c("N" = "orange", "Y" = "#36648B"),labels = c("N" = "No", "Y" = "Yes"), name="Treatment") +
    scale_fill_manual(values = c("N" = "orange", "Y" = "#36648B"),labels = c("N" = "No", "Y" = "Yes"), name="Treatment") +
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
                      labels = c("Training Data", "Test Data"),
                      label.x = 0,
                      label.y = 1.04,
                      hjust = 0,
                      vjust = 1,
                      ncol = 1, nrow = 2, align = "v",
                      common.legend = T)

  return(result)
}


library(ggpubr)
plot_outcome_ITE(data.dev.rs = test.results.tram$data.dev.rs, data.val.rs = test.results.tram$data.val.rs, x_lim = c(-0.6,0.6))




calc.ATE.Odds.tram <- function(data, log.odds = T){
  data <- as.data.frame(data)
  data$OUTCOME6M <- data$OUTCOME6M - 1 #transform back to 0, 1 coding
  model <- glm(OUTCOME6M ~ RXASP, data = data, family = binomial(link = "logit"))
  if (!log.odds){
    ATE.odds <- exp(coef(model)[2])
    ATE.lb <- exp(suppressMessages(confint(model)[2,1]))
    ATE.ub <- exp(suppressMessages(confint(model)[2,2]))
  } else {
    ATE.odds <- coef(model)[2]
    ATE.lb <- suppressMessages(confint(model)[2,1])
    ATE.ub <- suppressMessages(confint(model)[2,2])
  }
  return(data.frame(ATE.odds = ATE.odds, ATE.lb = ATE.lb, ATE.ub = ATE.ub, 
                    n.total = nrow(data), 
                    n.tr = sum(data$RXASP == 1), n.ct = sum(data$RXASP == 0)))
}


plot_ATE_ITE_in_group <- function(dev.data = data.dev.rs, val.data = data.val.rs, log.odds = T, ylb=0, yub=2){
  data <- rbind(dev.data %>% mutate(sample = "derivation"), val.data %>%  mutate(sample = "validation"))
  result <- ggplot(data, aes(x = ITE.Group, y = ATE.odds)) +
    geom_line(aes(group = sample, color = sample), linewidth = 1, 
              position = position_dodge(width = 0.2)) +
    geom_point(aes(color = sample), size = 1.5, 
               position = position_dodge(width = 0.2)) +
    geom_errorbar(aes(ymin = ATE.lb, ymax = ATE.ub, color = sample), width = 0.2,
                  position = position_dodge(width = 0.2))+
    geom_hline(yintercept = ifelse(log.odds, 0, 1), linetype = "dashed", color = "black") +
    scale_color_manual(name = "Group",
                       labels = c("derivation" = "Training Data", "validation" = "Test Data"),
                       values = c("orange", "#36648B"))+
    scale_x_discrete(guide = guide_axis(n.dodge = 2))+
    ylim(ylb,yub)+
    xlab("ITE Group")+
    ylab(ifelse(log.odds, "ATE in Log Odds Ratio", "ATE in Odds Ratio"))+
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


# breaks <- c(-0.6, -0.2, -0.05, 0.025, 0.05, 0.1, 0.6)
breaks <- c(-0.3, -0.1, -0.05, 0.025, 0.05, 0.3)
log.odds <- F
data.dev.grouped.ATE <- test.results.tram$data.dev.rs %>% 
  mutate(ITE.Group = cut(ITE, breaks = breaks, include.lowest = T)) %>%
  dplyr::filter(!is.na(ITE.Group)) %>%
  group_by(ITE.Group) %>% 
  group_modify(~ calc.ATE.Odds.tram(.x, log.odds = log.odds)) %>% ungroup()
data.val.grouped.ATE <- test.results.tram$data.val.rs %>% 
  mutate(ITE.Group = cut(ITE, breaks = breaks, include.lowest = T)) %>%
  dplyr::filter(!is.na(ITE.Group)) %>%
  group_by(ITE.Group) %>%
  group_modify(~ calc.ATE.Odds.tram(.x, log.odds = log.odds)) %>% ungroup() 

plot_ATE_ITE_in_group(dev.data = data.dev.grouped.ATE, val.data = data.val.grouped.ATE, log.odds = log.odds, ylb = 0, yub = 7.5)





#same for risk difference

calc.ATE.Risks.tram <- function(data) {
  data <- as.data.frame(data)
  data$OUTCOME6M <- data$OUTCOME6M - 1  # transform to 0/1
  
  # Ensure RXASP is a factor
  # data$RXASP <- factor(data$RXASP, levels = c("N", "Y"))
  
  # Calculate proportions
  p1 <- mean(data$OUTCOME6M[data$RXASP ==  1])  # treated
  p0 <- mean(data$OUTCOME6M[data$RXASP == 0])  # control
  
  # Risk difference
  ATE.RiskDiff <- p1 - p0
  
  # Standard error for RD (Wald)
  n1 <- sum(data$RXASP == 1)
  n0 <- sum(data$RXASP == 0)
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
    ylim(ylb,yub)+
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


# breaks <- c(-0.6, -0.2, -0.05, 0.025, 0.05, 0.1, 0.6)
breaks <- c(-0.3, -0.1, -0.05, 0.025, 0.05, 0.3)
log.odds <- F
data.dev.grouped.ATE <- test.results.tram$data.dev.rs %>% 
  mutate(ITE.Group = cut(ITE, breaks = breaks, include.lowest = T)) %>%
  dplyr::filter(!is.na(ITE.Group)) %>%
  group_by(ITE.Group) %>% 
  group_modify(~ calc.ATE.Risks.tram(.x)) %>% ungroup()
data.val.grouped.ATE <- test.results.tram$data.val.rs %>% 
  mutate(ITE.Group = cut(ITE, breaks = breaks, include.lowest = T)) %>%
  dplyr::filter(!is.na(ITE.Group)) %>%
  group_by(ITE.Group) %>%
  group_modify(~ calc.ATE.Risks.tram(.x)) %>% ungroup() 

plot_ATE_ITE_in_group_risks(dev.data = data.dev.grouped.ATE, val.data = data.val.grouped.ATE, ylb = -0.5, yub = 0.5)








# param_model has to be trained first
tram.ITE.earlystop <- function(data = test.compl.data.transformed, train_idx = train_idx, train = dat.train.tf, test = dat.test.tf){
  
  
  # Train set
  
  # Prepare Data
  # set the values of the first column of train to 0
  train_ct <- tf$concat(list(tf$zeros_like(train[, 1, drop = FALSE]), train[, 2:tf$shape(train)[2]]), axis = 1L)
  # set the values of the first column of train to 1
  train_tx <- tf$concat(list(tf$ones_like(train[, 1, drop = FALSE]), train[, 2:tf$shape(train)[2]]), axis = 1L)
  
  # Calculate potential outcomes
  h_params_ct <- param_model(train_ct)
  y_train_ct <- as.numeric(do_probability(h_params_ct))
  
  h_params_tx <- param_model(train_tx)
  y_train_tx <- as.numeric(do_probability(h_params_tx))
  
  # calculate ITE
  ITE_train <- y_train_tx - y_train_ct
  
  
  # Test set
  
  # Prepare Data
  # set the values of the first column of test to 0
  test_ct <- tf$concat(list(tf$zeros_like(test[, 1, drop = FALSE]), test[, 2:tf$shape(test)[2]]), axis = 1L)
  # set the values of the first column of test to 1
  test_tx <- tf$concat(list(tf$ones_like(test[, 1, drop = FALSE]), test[, 2:tf$shape(test)[2]]), axis = 1L)
  
  
  
  # Calculate potential outcomes
  h_params_ct <- param_model(test_ct)
  y_test_ct <- as.numeric(do_probability(h_params_ct))
  # y_test_ct <- predict(fit_gam, newdata = data.frame(pred_probability = y_test_ct), type = "response") # if recalibrate
  
  h_params_tx <- param_model(test_tx)
  y_test_tx <- as.numeric(do_probability(h_params_tx))
  # y_test_tx <- predict(fit_gam, newdata = data.frame(pred_probability = y_test_tx), type = "response") # if recalibrate
  
  # calculate ITE
  ITE_test <- y_test_tx - y_test_ct
  
  
  
  # generate data
  data.dev.rs <- data$data.dev %>% 
    # only where where train_idx (python format 0 index)
    filter(row_number() %in% (train_idx+1)) %>%
    mutate(ITE = ITE_train, 
           y.tx = y_train_tx, 
           y.ct = y_train_ct,
           RS = ifelse(ITE < 0, "benefit", "harm")) %>%
    mutate(RS = as.factor(RS))
  
  data.val.rs <- data$data.val %>% 
    mutate(ITE = ITE_test,, 
           y.tx = y_test_tx,
           y.ct = y_test_ct,
           RS = ifelse(ITE < 0, "benefit", "harm")) %>%
    mutate(RS = as.factor(RS))
  
  return(list(data.dev.rs = data.dev.rs, data.val.rs = data.val.rs))
}



# call function with early stopping
test.results.tram.earlystop <- tram.ITE.earlystop(data = test.compl.data.transformed, train_idx = train_idx, train = dat.train.tf, test = dat.test.tf)


plot_ITE_density(test.results.tram.earlystop)






plot_outcome_ITE(data.dev.rs = test.results.tram.earlystop$data.dev.rs, data.val.rs = test.results.tram.earlystop$data.val.rs, x_lim = c(-0.6,0.6))


# breaks <- c(-0.6, -0.2, -0.05, 0.025, 0.05, 0.1, 0.6)
breaks <- c(-1, -0.05, -0.02, 0, 0.025, 0.05)
log.odds <- F
data.dev.grouped.ATE <- test.results.tram.earlystop$data.dev.rs %>% 
  mutate(ITE.Group = cut(ITE, breaks = breaks, include.lowest = T)) %>%
  dplyr::filter(!is.na(ITE.Group)) %>%
  group_by(ITE.Group) %>% 
  group_modify(~ calc.ATE.Risks.tram(.x)) %>% ungroup()
data.val.grouped.ATE <- test.results.tram.earlystop$data.val.rs %>% 
  mutate(ITE.Group = cut(ITE, breaks = breaks, include.lowest = T)) %>%
  dplyr::filter(!is.na(ITE.Group)) %>%
  group_by(ITE.Group) %>%
  group_modify(~ calc.ATE.Risks.tram(.x)) %>% ungroup() 

plot_ATE_ITE_in_group_risks(dev.data = data.dev.grouped.ATE, val.data = data.val.grouped.ATE, ylb = -0.5, yub = 1.4)





# get NLL contributions




get_NLL_contributions = function (t_i, h_params){
  # t_i = dat.train.tf
  # h_params = h_params_orig                 # NN outputs (CS, LS, theta') for each obs
  # k_min <- k_constant(global_min)
  # k_max <- k_constant(global_max)
  
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

  ### Ordinal dimensions
  if (length(cont_ord) != 0){
    B = dim(h_params)[1]
    for (col in cont_ord){
      # col=39
      nol = 1 # Number of cut-points in respective dimension
      theta_ord = theta[,col,1:nol,drop=TRUE] # Intercept (2 values per observation if 2 cutpoints)
      
      
      h = theta_ord + h_LS[,col, drop=FALSE] + h_CS[,col, drop=FALSE]
      
      # cdf_cut <- logistic_cdf(h)
      # prob_Y1_X <- 1- cdf_cut
      
      h = theta_ord + h_LS[,col, drop=FALSE] + h_CS[,col, drop=FALSE]
      # putting -Inf and +Inf to the left and right of the cutpoints
      neg_inf = tf$fill(c(B,1L), -Inf)
      pos_inf = tf$fill(c(B,1L), +Inf)
      h_with_inf = tf$concat(list(neg_inf, h, pos_inf), axis=-1L)
      logistic_cdf_values = logistic_cdf(h_with_inf)
      #cdf_diffs <- tf$subtract(logistic_cdf_values[, 2:ncol(logistic_cdf_values)], logistic_cdf_values[, 1:(ncol(logistic_cdf_values) - 1)])
      cdf_diffs <- tf$subtract(logistic_cdf_values[, 2:tf$shape(logistic_cdf_values)[2]], logistic_cdf_values[, 1:(tf$shape(logistic_cdf_values)[2] - 1)])
      # Picking the observed cdf_diff entry
      class_indices <- tf$cast(t_i[, col] - 1, tf$int32)  # Convert to zero-based index
      # Create batch indices to pair with class indices
      batch_indices <- tf$range(tf$shape(class_indices)[1])
      # Combine batch_indices and class_indices into pairs of indices
      gather_indices <- tf$stack(list(batch_indices, class_indices), axis=1)
      cdf_diff_picked <- tf$gather_nd(cdf_diffs, gather_indices)
      # Gather the corresponding values from cdf_diffs
      NLL_contribution = -tf$math$log(cdf_diff_picked)
    }
  }
  
  ### DEBUG 
  #if (sum(is.infinite(log_lik$numpy())) > 0){
  #  print("Hall")
  #}
  return (as.numeric(NLL_contribution))
}


# NLL contributions on train set
h_params <- param_model(dat.train.tf)
NLL_contributions_train <- get_NLL_contributions(t_i = dat.train.tf, h_params = h_params)

# NLL contributions on test set
h_params <- param_model(dat.test.tf)
NLL_contributions_test <- get_NLL_contributions(t_i = dat.test.tf, h_params = h_params)


# plot NLL contributions (train and test set)
par(mfrow=c(1,2))
hist(NLL_contributions_train, breaks = 50, main = "NLL contributions train", xlab = "NLL contributions")
hist(NLL_contributions_test, breaks = 50, main = "NLL contributions test", xlab = "NLL contributions")


# get NLL contributions (train, under Do(T=1) and Do(T=0))

# Train set

# Prepare Data
# set the values of the first column of train to 0
train_ct <- tf$concat(list(tf$zeros_like(dat.train.tf[, 1, drop = FALSE]), dat.train.tf[, 2:tf$shape(dat.train.tf)[2]]), axis = 1L)
# set the values of the first column of train to 1
train_tx <- tf$concat(list(tf$ones_like(dat.train.tf[, 1, drop = FALSE]), dat.train.tf[, 2:tf$shape(dat.train.tf)[2]]), axis = 1L)

# calculate NLL contributions
h_params_dev_ct <- param_model(train_ct)
NLL_contributions_dev_ct <- get_NLL_contributions(t_i = train_ct, h_params = h_params_dev_ct)

h_params_dev_tx <- param_model(train_tx)
NLL_contributions_dev_tx <- get_NLL_contributions(t_i = train_tx, h_params = h_params_dev_tx)


cols_train <- as.numeric(dat.train.tf[, 1, drop = FALSE]) + 1
par(mfrow=c(1,2))
# scatterplot NLL contributions for T=0 vs T=1
plot(NLL_contributions_dev_ct, NLL_contributions_dev_tx, 
     xlab = "NLL contributions T=0", 
     ylab = "NLL contributions T=1", 
     main="Train set", col = cols_train)




# # Test set
# 
# Prepare Data
# set the values of the first column of test to 0
test_ct <- tf$concat(list(tf$zeros_like(dat.test.tf[, 1, drop = FALSE]), dat.test.tf[, 2:tf$shape(dat.test.tf)[2]]), axis = 1L)
# set the values of the first column of test to 1
test_tx <- tf$concat(list(tf$ones_like(dat.test.tf[, 1, drop = FALSE]), dat.test.tf[, 2:tf$shape(dat.test.tf)[2]]), axis = 1L)

# calculate NLL contributions
h_params_val_ct <- param_model(test_ct)
NLL_contributions_val_ct <- get_NLL_contributions(t_i = test_ct, h_params = h_params_val_ct)

h_params_val_tx <- param_model(test_tx)
NLL_contributions_val_tx <- get_NLL_contributions(t_i = test_tx, h_params = h_params_val_tx)

cols_test <- as.numeric(dat.test.tf[, 1, drop = FALSE]) + 1
# scatterplot NLL contributions for T=0 vs T=1
plot(NLL_contributions_val_ct, NLL_contributions_val_tx, 
     xlab = "NLL contributions T=0", ylab = "NLL contributions T=1", 
     main="Test set", col = cols_test)























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

# save recalibrated plot
# png("C:/Users/kraeh/OneDrive/Dokumente/Desktop/UZH_Biostatistik/Masterarbeit/MA_Mike/presentation_report/intermediate_presentation/img/ITE_recal.png",
#     width = 800, height = 600, res = 150)
# ggplot(df_test_recalibrated, aes(x = ITE_true, y = ITE_i_test_recal, color = Treatment)) +
#   geom_point() +
#   geom_abline(slope = 1, intercept = 0, color = "red") +
#   labs(title = "Recalibrated ITE (Test)", x = "True ITE", y = "Estimated ITE") +
#   theme_minimal() +
#   theme(legend.position = "top")
# dev.off()



# get recalibrated test set values
ITE_i_train_recal <- predict(fit_gam, newdata = data.frame(pred_probability = data.dev.rs$ITE), type = "response")





# make ate plot

breaks <- c(-0.75, -0.4, -0.2, 0.1, 0.5)
log.odds <- F
data.dev.grouped.ATE.recal <- data.dev.rs %>% 
  # add as.numeric(ITE_i_test_recal) to variable ITE
  mutate(ITE = as.numeric(ITE_i_train_recal)) %>%
  mutate(ITE.Group = cut(ITE, breaks = breaks, include.lowest = T)) %>%
  dplyr::filter(!is.na(ITE.Group)) %>%
  group_by(ITE.Group) %>% 
  group_modify(~ calc.ATE.Odds(.x, log.odds = log.odds)) %>% ungroup()
data.val.grouped.ATE.recal <- data.val.rs %>% 
  # add as.numeric(ITE_i_test_recal) to variable ITE
  mutate(ITE = as.numeric(ITE_i_test_recal)) %>%
  mutate(ITE.Group = cut(ITE, breaks = breaks, include.lowest = T)) %>%
  dplyr::filter(!is.na(ITE.Group)) %>%
  group_by(ITE.Group) %>%
  group_modify(~ calc.ATE.Odds(.x, log.odds = log.odds)) %>% ungroup() 

png("C:/Users/kraeh/OneDrive/Dokumente/Desktop/UZH_Biostatistik/Masterarbeit/MA_Mike/presentation_report/intermediate_presentation/img/ATE_ITE_recal.png",
    width = 800, height = 600, res = 150)
plot_ATE_ITE_in_group(dev.data = data.dev.grouped.ATE.recal, val.data = data.val.grouped.ATE.recal, 
                      log.odds = log.odds, ylb = 0, yub = 3.7,
                      train.data.name = "Train", test.data.name = "Test")
dev.off()



























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






# check distribution of Y against X1 on treat and control (maybe reason for weird shift for treatment group)


#train set

treat <- dgp_data$test.compl.data$data.dev %>%
  filter(Tr == 1) 
contr <- dgp_data$test.compl.data$data.dev %>%
  filter(Tr == 0)


# fit gam for binary response Y with continuous predictor X1 on treat, 

gam_model_treat <- gam(Y ~ s(X1)+ s(X2), data = treat, family = binomial(link="logit"), gamma=0.4)
# select only plot for X1
plot(gam_model_treat, main= "Y~s(X1) treated (train set)", 
     xlab = "X1", cex.axis=0.8, cex.lab=0.8, cex.main=0.8, select=1)


# fit gam for binary response Y with continuous predictor X1 on contr,
gam_model_contr <- gam(Y ~ s(X1) + s(X2), data = contr, family = binomial(link="logit"), gamma=0.4)
plot(gam_model_contr, main= "Y~s(X1) control (train set)", 
     xlab = "X1", cex.axis=0.8, cex.lab=0.8, cex.main=0.8, select=1)


# same on test set

treat_test <- dgp_data$test.compl.data$data.val %>%
  filter(Tr == 1)
contr_test <- dgp_data$test.compl.data$data.val %>%
  filter(Tr == 0)

# fit gam for binary response Y with continuous predictor X1 on treat,
gam_model_treat_test <- gam(Y ~ s(X1) + s(X2), data = treat_test, family = binomial(link="logit"), gamma=0.4)
plot(gam_model_treat_test, main= "Y~s(X1) treated (test set)", 
     xlab = "X1", cex.axis=0.8, cex.lab=0.8, cex.main=0.8, select=1)

# fit gam for binary response Y with continuous predictor X1 on contr,
gam_model_contr_test <- gam(Y ~ s(X1)+ s(X2), data = contr_test, family = binomial(link="logit"), gamma=0.4)
plot(gam_model_contr_test)


