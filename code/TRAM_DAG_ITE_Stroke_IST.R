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

# for ITE
library(ggpubr)
library(splines)
library(comets)


#### For ITE (change to utils_ITE_IST_stroke_trial.R)
source('code/utils/utils_ITE_IST_stroke_trial.R')
# source('code/utils/ITE_utils.R')

#### For TF
source('code/utils/utils_tf.R')

#### For TFP
library(tfprobability)
source('code/utils/utils_tfp.R')

##### Flavor of experiment ######

#### Saving the current version of the script into runtime
DIR = 'runs/TRAM_DAG_ITE_Stroke_IST/run'

if (!dir.exists(DIR)) {
  dir.create(DIR, recursive = TRUE)
}
# Copy this file to the directory DIR
file.copy('/code/TRAM_DAG_ITE_Stroke_IST.R', file.path(DIR, 'TRAM_DAG_ITE_Stroke_IST.R'), overwrite=TRUE)


len_theta = 20 # Number of coefficients of the Bernstein polynomials
hidden_features_I = c(20, 10, 10, 2) # architecture for complex intercept (CI)
hidden_features_CS = c(2,5,5,2) # not used in this experiment



# MODEL_NAME = 'ModelCI'
# MODEL_NAME = 'ModelCIDropout0.3'
MODEL_NAME = 'ModelCIDropout0.1_standardized'



fn = file.path(DIR, paste0(MODEL_NAME))
print(paste0("Starting experiment ", fn))






###############################################################################
# IST Stroke Trial: load and prepare data
###############################################################################


# raw data downloaded from: (https://trialsjournal.biomedcentral.com/articles/10.1186/1745-6215-12-101#Sec8)
# and saved as .csv in folder \data

# same code for preprossessing used as by (https://gitlab.uzh.ch/hongruyu.chen/causal_ml-for-ite)
# name of file \code\IST_data_processing.Rmd

# cleaned dataset saved as IST_model_data.RData in \data (file in RData is named IST.dataset.clean.6m)


################################################################################
# load data of the IST trial
################################################################################


# named IST.dataset.clean.6m
load('data/IST_model_data.RData')

# remove IST.dataset.clean.14d, IST.dataset.clean.14d.dr as not needed
remove(list = c("IST.dataset.clean.14d", "IST.dataset.clean.14d.dr"))

# check structure of the dataset (19'435 individuals)
str(IST.dataset.clean.6m)


sum(!complete.cases(IST.dataset.clean.6m))/nrow(IST.dataset.clean.6m)

# duplicates
duplicate_rows <- IST.dataset.clean.6m[duplicated(IST.dataset.clean.6m) | duplicated(IST.dataset.clean.6m, fromLast = TRUE), ]
duplicate_rows

################################################################################
# Table one
################################################################################

library("tableone")
tableone <- CreateTableOne(strata = "RXASP", factorVars = "OUTCOME6M", includeNA = T, data = IST.dataset.clean.6m, test = FALSE, addOverall = FALSE)
print(tableone, showAllLevels = TRUE)
summary(tableone)




################################################################################
# split the data into train (2/3) and test (1/3), as well as in treated (.tx) and untreated (.ct) groups
################################################################################



# original untransformed (factor encoded)
set.seed(12345)
test.data <- split_data(IST.dataset.clean.6m, 2/3)

# remove missing values
test.compl.data <- remove_NA_data(test.data)

# number of complete patients (18271)
(n_complete <- nrow(test.compl.data$data.dev) + nrow(test.compl.data$data.val))

# percentage of incomplete patients (5.98%)
(perc_incomplete <- (nrow(IST.dataset.clean.6m) - n_complete)/ 
  nrow(IST.dataset.clean.6m)) 



################################################################################
# same with dummy encoding for TRAM-DAG
################################################################################



# check datatypes
str(IST.dataset.clean.6m)    

# original dataset (factor encoded)
print(IST.dataset.clean.6m, n = 5, width = Inf)




# Apply to each dataset in the list to the function transform_dataset (dummy encodes)
test.compl.data.transformed <- lapply(test.compl.data, transform_dataset)

str(test.compl.data.transformed$data.dev) # check structure of the transformed dataset


### just two quick checks for correctness of the transformation

# compare first 5 rows of original and transformed datasets (dev)
head(test.compl.data$data.dev, n = 5, width = Inf)
head(test.compl.data.transformed$data.dev, n = 5, width = Inf)
# --> yes, same individuals!

# check if the same outcomes are in the .ct datasets (val)
identical(test.compl.data$data.dev.ct$OUTCOME6M, test.compl.data.transformed$data.dev.ct$OUTCOME6M -1) 
# --> yes, same individuals!




################################################################################
# TRAM-DAG convert to tensor + define meta adjacency matrix
################################################################################



## convert to tensor

# train set as tensor
dat.train.tf = tf$constant(as.matrix(test.compl.data.transformed$data.dev), dtype = 'float32')

# test set as tensor
dat.test.tf = tf$constant(as.matrix(test.compl.data.transformed$data.val), dtype = 'float32')

# define min, max for Outcome
q_Outcome = c(1, 2) # outcome levels
min = tf$constant(1, dtype = 'float32')
max = tf$constant(2, dtype = 'float32')

# define datatypes: all variables are treated as continuous (c) except outcome as Ordinal (o)

N_var <- ncol(test.compl.data.transformed$data.dev) #number of variables:
type <- data_type <-  c(rep('c', N_var-1), 'o')



## Meta Adjacency Matrix (MA)

# RCT: all variables have an influence on Outcome6M by a Complex Intercept


# Create a zero matrix NxN
A <- matrix(0, nrow = N_var, ncol = N_var)

# Set 1s in the last column, except the last row
A[1:(N_var - 1), N_var] <- 1


# Meta adjacency
MA <- ifelse(A==1, "ci", 0)

################################################################################
# ITE estimation with GLM, tuned Random Forest and TRAM-DAG
################################################################################


#################################################
# Analysis T-learner GLM (same as Chen et al. (2025))
#################################################



# create a sub-folder named "IST_glm_tlearner"
if (!dir.exists(file.path(DIR, "IST_glm_tlearner"))) {
  dir.create(file.path(DIR, "IST_glm_tlearner"))
}

model  <- "glm_tlearner_"





# apply logistic t-learner
test.results.logistic <- logis.ITE(test.compl.data)

data.dev.rs = test.results.logistic[["data.dev.rs"]] %>%  as.data.frame()
data.val.rs = test.results.logistic[["data.val.rs"]] %>%  as.data.frame()


# check ITE distribution
plot(density(data.dev.rs$ITE))
lines(density(data.val.rs$ITE), col = "red")
legend("topright", legend = c("Train", "Test"), col = c("black", "red"), lty = 1)








# recreate ITE-Outcome plot from Chen et. al (2025) - p.18
plot_outcome_ITE(data.dev.rs = data.dev.rs, data.val.rs = data.val.rs, x_lim = c(-0.3,0.3))



# recreate ATE-ITE (Risk Ratio) plot from Chen et. al (2025) - p.18
breaks <- c(-0.3, -0.1, -0.05, -0.025, 0, 0.025, 0.05, 0.1, 0.3)
log.odds <- F
data.dev.grouped.ATE <- data.dev.rs %>% 
  mutate(ITE.Group = cut(ITE, breaks = breaks, include.lowest = T)) %>%
  dplyr::filter(!is.na(ITE.Group)) %>%
  group_by(ITE.Group) %>% 
  group_modify(~ calc.ATE.RR(.x)) %>% ungroup()
data.val.grouped.ATE <- data.val.rs %>% 
  mutate(ITE.Group = cut(ITE, breaks = breaks, include.lowest = T)) %>%
  dplyr::filter(!is.na(ITE.Group)) %>%
  group_by(ITE.Group) %>%
  group_modify(~ calc.ATE.RR(.x)) %>% ungroup() 

plot_ATE_ITE_in_group(dev.data = data.dev.grouped.ATE, val.data = data.val.grouped.ATE, log.odds = log.odds, ylb = 0.2, yub = 1.6)




# ITE-ATE plot in terms of Risk Difference

breaks <- round(seq(-0.2, 0.2, by = 0.1), 1)

plot_ATE_vs_ITE_base_RiskDiff(model.results = test.results.logistic, 
                              breaks = breaks, 
                              delta_horizontal = 0.02, 
                              ylim_delta = 0.1)



# calibration plot
par(mfrow=c(1,1), pty="s")
library("CalibrationCurves")
# train
res <- val.prob.ci.2(test.results.logistic$data.dev.rs$Y_pred, test.results.logistic$data.dev.rs$OUTCOME6M)
# test
res <- val.prob.ci.2(test.results.logistic$data.val.rs$Y_pred, test.results.logistic$data.val.rs$OUTCOME6M)



# save calibration plots

file_name <- file.path(DIR, "IST_glm_tlearner", paste0(model, 'train_calibration_plot.png'))
png(filename = file_name, width = 1600, height = 1600, res = 300)

res <- val.prob.ci.2(test.results.logistic$data.dev.rs$Y_pred, test.results.logistic$data.dev.rs$OUTCOME6M)
# Close PNG device
dev.off()


# test
file_name <- file.path(DIR, "IST_glm_tlearner", paste0(model, 'test_calibration_plot.png'))
png(filename = file_name, width = 1600, height = 1600, res = 300)

res <- val.prob.ci.2(test.results.logistic$data.val.rs$Y_pred, test.results.logistic$data.val.rs$OUTCOME6M)
# Close PNG device
dev.off()



# plot for slides

breaks <- round(seq(-0.2, 0.2, by = 0.1), 1)
plot_for_slides_IST(model.results = test.results.logistic, 
              breaks = breaks, 
              delta_horizontal = 0.008, 
              ylim_delta = 0.1, 
              xlim_delta = 0.1)


# save plot for slides

file_name <- file.path(DIR, "IST_glm_tlearner", paste0(model, 'density_ITE_ATE.png'))
png(filename = file_name, width = 2350, height = 1150, res = 300)

plot_for_slides_IST(model.results = test.results.logistic, 
                breaks = breaks, 
                delta_horizontal = 0.005,
                ylim_delta = 0.1,
                xlim_delta = 0.1)
# Close PNG device
dev.off()








# ATE Observed
df <- test.results.logistic   
mean(df$data.dev.rs$OUTCOME6M[df$data.dev.rs$RXASP == "Y"]) - mean(df$data.dev.rs$OUTCOME6M[df$data.dev.rs$RXASP == "N"])
mean(df$data.val.rs$OUTCOME6M[df$data.val.rs$RXASP == "Y"]) - mean(df$data.val.rs$OUTCOME6M[df$data.val.rs$RXASP == "N"])

lm_dev <- lm(OUTCOME6M ~ RXASP, data = df$data.dev.rs)
lm_val <- lm(OUTCOME6M ~ RXASP, data = df$data.val.rs)




# ATE Estimated
mean(test.results.logistic$data.dev.rs$ITE)
mean(test.results.logistic$data.val.rs$ITE)

boot_result_dev <- ITE_CI_bootstrap(ITE = test.results.logistic$data.dev.rs$ITE, n_boot = 10000)
boot_result_val <- ITE_CI_bootstrap(ITE = test.results.logistic$data.val.rs$ITE, n_boot = 10000)





# make the table as Row (mean, lower, upper) and Column (ATE_obs_train, ATE_obs_test, ATE_est_train, ATE_est_test)

# Construct the matrix with rows: mean, lower, upper
ATE_summary_table <- data.frame(
  row.names = c("mean", "lower", "upper"),
  ATE_obs_train = c(coef(lm_dev)[2], confint(lm_dev)[2, 1], confint(lm_dev)[2, 2]),
  ATE_obs_test = c(coef(lm_val)[2], confint(lm_val)[2, 1], confint(lm_val)[2, 2]),
  ATE_est_train = c(boot_result_dev$mean, boot_result_dev$ci_lower, boot_result_dev$ci_upper),
  ATE_est_test = c(boot_result_val$mean, boot_result_val$ci_lower, boot_result_val$ci_upper)
)

# View the table
print(ATE_summary_table)


# save table ATE_summary_table, test.results.logistic in the created folder
save(ATE_summary_table, test.results.logistic, file = file.path(DIR, "IST_glm_tlearner", "glm_tlearner_results.RData"))


#### note that the CI for the ATE = mean(ITE) is much smaller as for ATE_obs
#### -> not directly comparable, not full uncertainty, see plot below

# Prepare data
mean_vals <- c(
  ATE_obs_train = coef(lm_dev)[2],
  ATE_obs_test = coef(lm_val)[2],
  ATE_est_train = boot_result_dev$mean,
  ATE_est_test = boot_result_val$mean
)

lower_vals <- c(
  confint(lm_dev)[2, 1],
  confint(lm_val)[2, 1],
  boot_result_dev$ci_lower,
  boot_result_val$ci_lower
)

upper_vals <- c(
  confint(lm_dev)[2, 2],
  confint(lm_val)[2, 2],
  boot_result_dev$ci_upper,
  boot_result_val$ci_upper
)

labels <- c("Obs Train", "Obs Test", "Est Train", "Est Test")
colors <- c("orange", "orange", "#36648B", "#36648B")
pch_vals <- c(16, 17, 16, 17)  # 16 = filled circle, 17 = filled triangle
x_pos <- 1:4

# Set up plot
ylim <- range(c(lower_vals, upper_vals))
plot(x_pos, mean_vals, ylim = ylim, xaxt = "n", xlab = "", ylab = "ATE (Risk Difference)",
     pch = pch_vals, col = colors, cex = 1.5,
     main = "ATE Estimates with 95% CI")

# Add CI error bars
arrows(x_pos, lower_vals, x_pos, upper_vals, angle = 90, code = 3, length = 0.05, col = colors, lwd = 2)

# Add horizontal reference line
abline(h = 0, lty = "dotted", col = "gray")

# Add custom axis labels
axis(1, at = x_pos, labels = labels)

# Add legend
legend("topright",
       legend = c("Observed", "Estimated", "Train", "Test"),
       col = c("orange", "#36648B", "black", "black"),
       pch = c(16, 16, 16, 17),
       bty = "n", cex = 0.9)




#################################################
# Tuned Random Forest for ITE
#################################################



# create a sub-folder named "IST_tuned_rf_tlearner"
if (!dir.exists(file.path(DIR, "IST_tuned_rf_tlearner"))) {
  dir.create(file.path(DIR, "IST_tuned_rf_tlearner"))
}

model  <- "IST_tuned_rf_tlearner_"



# Fit the tuned random forest with the transformed data
set.seed(12345)
tuned_rf.results <- fit.tuned_rf_IST(test.compl.data.transformed)

data.dev.rs = tuned_rf.results[["data.dev.rs"]] %>%  as.data.frame()
data.val.rs = tuned_rf.results[["data.val.rs"]] %>%  as.data.frame()


# check ITE distribution
plot(density(data.dev.rs$ITE))
lines(density(data.val.rs$ITE), col = "red")
legend("topright", legend = c("Train", "Test"), col = c("black", "red"), lty = 1)


# recreate ITE-Outcome
plot_outcome_ITE_tuned_rf(data.dev.rs = tuned_rf.results$data.dev.rs, data.val.rs = tuned_rf.results$data.val.rs, x_lim = c(-0.5,0.5))





# check treatment and outcome
str(tuned_rf.results$data.dev.rs$RXASP)      # 0/1 encoded -> change to N/Y
str(tuned_rf.results$data.dev.rs$OUTCOME6M)  # 1/2 encoded -> change to 0/1

tuned_rf.results_plot <- tuned_rf.results
tuned_rf.results_plot$data.dev.rs$RXASP <- ifelse(tuned_rf.results$data.dev.rs$RXASP == 0, "N", "Y")
tuned_rf.results_plot$data.val.rs$RXASP <- ifelse(tuned_rf.results$data.val.rs$RXASP == 0, "N", "Y")

tuned_rf.results_plot$data.dev.rs$OUTCOME6M <- ifelse(tuned_rf.results$data.dev.rs$OUTCOME6M == 1, 0, 1)
tuned_rf.results_plot$data.val.rs$OUTCOME6M <- ifelse(tuned_rf.results$data.val.rs$OUTCOME6M == 1, 0, 1)

# check treatment and outcome
str(tuned_rf.results_plot$data.dev.rs$RXASP)      # N/Y encoded
str(tuned_rf.results_plot$data.dev.rs$OUTCOME6M)  # 0/1 encoded


# ITE-ATE plot in terms of Risk Difference

breaks <- round(seq(-0.2, 0.2, by = 0.1), 1)
plot_ATE_vs_ITE_base_RiskDiff(model.results = tuned_rf.results_plot, 
                              breaks = breaks, 
                              delta_horizontal = 0.004, 
                              ylim_delta = 0.1)




# same with ggplot

# breaks <- c(-0.6, -0.2, -0.05, 0.025, 0.05, 0.1, 0.6)
breaks <- c(-0.3, -0.1, -0.05, 0.025, 0.05, 0.3)
breaks <- round(seq(-0.2, 0.2, by = 0.1), 1)

log.odds <- F
data.dev.grouped.ATE <- tuned_rf.results$data.dev.rs %>% 
  mutate(ITE.Group = cut(ITE, breaks = breaks, include.lowest = T)) %>%
  dplyr::filter(!is.na(ITE.Group)) %>%
  group_by(ITE.Group) %>% 
  group_modify(~ calc.ATE.RiskDiff_transformed(.x)) %>% ungroup()
data.val.grouped.ATE <- tuned_rf.results$data.val.rs %>% 
  mutate(ITE.Group = cut(ITE, breaks = breaks, include.lowest = T)) %>%
  dplyr::filter(!is.na(ITE.Group)) %>%
  group_by(ITE.Group) %>%
  group_modify(~ calc.ATE.RiskDiff_transformed(.x)) %>% ungroup() 

plot_ATE_ITE_in_group_RiskDiff_ggplot(dev.data = data.dev.grouped.ATE, val.data = data.val.grouped.ATE, ylb = -0.5, yub = 0.5)




# calibration plot (CalibrationCurves)
par(mfrow=c(1,1), pty="s")
library("CalibrationCurves")
# train
res <- val.prob.ci.2(tuned_rf.results_plot$data.dev.rs$Y_pred, tuned_rf.results_plot$data.dev.rs$OUTCOME6M)
# test
res <- val.prob.ci.2(tuned_rf.results_plot$data.val.rs$Y_pred, tuned_rf.results_plot$data.val.rs$OUTCOME6M)



# save calibration plots

file_name <- file.path(DIR, "IST_tuned_rf_tlearner", paste0(model, 'train_calibration_plot.png'))
png(filename = file_name, width = 1600, height = 1600, res = 300)

res <- val.prob.ci.2(tuned_rf.results_plot$data.dev.rs$Y_pred, tuned_rf.results_plot$data.dev.rs$OUTCOME6M)
# Close PNG device
dev.off()


# test
file_name <- file.path(DIR, "IST_tuned_rf_tlearner", paste0(model, 'test_calibration_plot.png'))
png(filename = file_name, width = 1600, height = 1600, res = 300)

res <- val.prob.ci.2(tuned_rf.results_plot$data.val.rs$Y_pred, tuned_rf.results_plot$data.val.rs$OUTCOME6M)
# Close PNG device
dev.off()



# calibration plot (manually)

# Train set
plot_calibration(y_obs = tuned_rf.results$data.dev.rs$OUTCOME6M - 1, 
                 y_pred = tuned_rf.results$data.dev.rs$Y_pred, bins = 10, title_suffix = "Train")

plot_roc(
  y_obs = tuned_rf.results$data.dev.rs$OUTCOME6M - 1, 
  y_pred = tuned_rf.results$data.dev.rs$Y_pred, 
  title_suffix = "Train"
)

# Test set
plot_calibration(y_obs = tuned_rf.results$data.val.rs$OUTCOME6M - 1, 
                 y_pred = tuned_rf.results$data.val.rs$Y_pred, bins = 10, title_suffix = "Test")

auc_test <- plot_roc(
  y_obs = tuned_rf.results$data.val.rs$OUTCOME6M - 1, 
  y_pred = tuned_rf.results$data.val.rs$Y_pred, 
  title_suffix = "Test"
)





# plot for slides

breaks <- round(seq(-0.2, 0.2, by = 0.1), 1)
plot_for_slides_IST(model.results = tuned_rf.results_plot, 
              breaks = breaks, 
              delta_horizontal = 0.004, 
              ylim_delta = 0.1, 
              xlim_delta = 0.1)

# save plot for slides
file_name <- file.path(DIR, "IST_tuned_rf_tlearner", paste0(model, 'density_ITE_ATE.png'))
png(filename = file_name, width = 2350, height = 1150, res = 300)
plot_for_slides_IST(model.results = tuned_rf.results_plot, 
                breaks = breaks, 
                delta_horizontal = 0.004,
                ylim_delta = 0.1,
                xlim_delta = 0.1)
# Close PNG device
dev.off()





# ATE Observed
df <- tuned_rf.results_plot
mean(df$data.dev.rs$OUTCOME6M[df$data.dev.rs$RXASP == "Y"]) - mean(df$data.dev.rs$OUTCOME6M[df$data.dev.rs$RXASP == "N"])
mean(df$data.val.rs$OUTCOME6M[df$data.val.rs$RXASP == "Y"]) - mean(df$data.val.rs$OUTCOME6M[df$data.val.rs$RXASP == "N"])


lm_dev <- lm(OUTCOME6M ~ RXASP, data = df$data.dev.rs)
lm_val <- lm(OUTCOME6M ~ RXASP, data = df$data.val.rs)



# ATE Estimated
mean(tuned_rf.results_plot$data.dev.rs$ITE)
mean(tuned_rf.results_plot$data.val.rs$ITE)

boot_result_dev <- ITE_CI_bootstrap(ITE = tuned_rf.results_plot$data.dev.rs$ITE, n_boot = 10000)
boot_result_val <- ITE_CI_bootstrap(ITE = tuned_rf.results_plot$data.val.rs$ITE, n_boot = 10000)

# make the table as Row (mean, lower, upper) and Column (ATE_obs_train, ATE_obs_test, ATE_est_train, ATE_est_test)

ATE_summary_table <- data.frame(
  row.names = c("mean", "lower", "upper"),
  ATE_obs_train = c(coef(lm_dev)[2], confint(lm_dev)[2, 1], confint(lm_dev)[2, 2]),
  ATE_obs_test = c(coef(lm_val)[2], confint(lm_val)[2, 1], confint(lm_val)[2, 2]),
  ATE_est_train = c(boot_result_dev$mean, boot_result_dev$ci_lower, boot_result_dev$ci_upper),
  ATE_est_test = c(boot_result_val$mean, boot_result_val$ci_lower, boot_result_val$ci_upper)
)



ATE_summary_table


# save ATE_summary_table, tuned_rf.results_plot in the created folder

save(ATE_summary_table, tuned_rf.results_plot, file = file.path(DIR, "IST_tuned_rf_tlearner", "tuned_rf_tlearner_results.RData"))















#################################################
# TRAM-DAG for ITE (S-learner)
#################################################



dat.train.tf # check structure of the train data tensor

##### Train on control group ####

# create the S-learner TRAM-DAG with Complex Intercept (CI) of all predictors
param_model = create_param_model(MA, hidden_features_I=hidden_features_I, 
                                 len_theta=len_theta, 
                                 hidden_features_CS=hidden_features_CS,
                                 dropout=TRUE, batchnorm=TRUE, activation = "relu")


optimizer = optimizer_adam(learning_rate = 0.001)

# custom loss function "struct_dag_loss_ITE_IST" for optimizing only OUTCOME6M
param_model$compile(optimizer, loss=struct_dag_loss_ITE_IST)

h_params <- param_model(dat.train.tf)

param_model$evaluate(x = dat.train.tf, y=dat.train.tf, batch_size = 7L)
summary(param_model)



##### Training ####

num_epochs <- 250
# num_epochs <- 300   ### 200 is the file


# Split dat.train.tf into training and validation sets
set.seed(42)  # for reproducibility
n <- nrow(dat.train.tf)
val_idx <- sample(0:(n - 1), size = floor(0.2 * n))  
train_idx <- setdiff(0:(n - 1), val_idx)

# Convert to Tensor
val_idx_tf <- tf$constant(val_idx, dtype = tf$int32)
train_idx_tf <- tf$constant(train_idx, dtype = tf$int32)

# Validation set (used for early stopping)
x_val   <- tf$gather(dat.train.tf, val_idx_tf)

# Training set (used for training)
x_train <- tf$gather(dat.train.tf, train_idx_tf)

# fnh5 = paste0(fn, '_E', num_epochs, 'early_stopping_CI.h5')   # 'CI.h5'
# fnRdata = paste0(fn, '_E', num_epochs, 'early_stopping_CI.RData')   # 'CI.RData'


# num_epochs <- 400
# fnh5 = paste0(fn, '_E', num_epochs, 'ModelCIDropout0.1.h5')   # 'CI.h5'
# fnRdata = paste0(fn, '_E', num_epochs, 'ModelCIDropout0.1.RData')   # 'CI.RData'


fnh5 = paste0(fn, '_E', num_epochs, 'CI_batchnorm_dropout0.1.h5')   # 'CI.h5'
fnRdata = paste0(fn, '_E', num_epochs, 'CI_batchnorm_dropout0.1.RData')   # 'CI.RData'



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
    patience <- 30
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




# plot the loss curve
par(mfrow=c(1,1))
epochs = length(train_loss)
plot(1:length(train_loss), train_loss, type='l', main='Normal Training (green is valid)', ylim = c(0.4, 0.8))
lines(1:length(train_loss), val_loss, type = 'l', col = 'green')


# Last XX epochs
last <- 30
diff = max(epochs - last,0)
plot(diff:epochs, val_loss[diff:epochs], type = 'l', col = 'green', main=paste0('Last ' ,last  ,' epochs'))
lines(diff:epochs, train_loss[diff:epochs], type='l')







### Check calibration on the training set:


h_params_orig <- param_model(dat.train.tf)
Y_prob_tram_dag <- as.numeric(do_probability_IST(h_params_orig))

# brier score 
mean((as.numeric(dat.train.tf[,39]-1) - Y_prob_tram_dag)^2)



plot_calibration(y_obs = as.numeric(dat.train.tf[,39]-1), 
                 y_pred = Y_prob_tram_dag, bins = 10, title_suffix = "Train")


# ROC on training
plot_roc(
  y_obs = as.numeric(dat.train.tf[,39]-1), 
  y_pred = Y_prob_tram_dag, 
  title_suffix = "Train"
)





### Check calibration on the test set:

##### Check predictive power of the model on test set

# input test data into model
h_params_orig <- param_model(dat.test.tf)

# probabilities for Y=1 on original test data
Y_prob_tram_dag_test <- as.numeric(do_probability_IST(h_params_orig))

# brier score
mean((as.numeric(dat.test.tf[,39]-1) - Y_prob_tram_dag_test)^2)



plot_calibration(y_obs = as.numeric(dat.test.tf[,39]-1), 
                 y_pred = Y_prob_tram_dag_test, bins = 10, title_suffix = "Test")


# ROC on test
plot_roc(
  y_obs = as.numeric(dat.test.tf[,39]-1), 
  y_pred = Y_prob_tram_dag_test, 
  title_suffix = "Test"
)




# calibration plot (CalibrationCurves)
par(mfrow=c(1,1), pty="s")
library("CalibrationCurves")
# train
res <- val.prob.ci.2(Y_prob_tram_dag, as.numeric(dat.train.tf[,39]-1))
# test
res <- val.prob.ci.2(Y_prob_tram_dag_test, as.numeric(dat.test.tf[,39]-1))


# does not need recalibration




#################################################
# calculate ITE on train and test set (TRAM-DAG S-learner)
#################################################



# create a sub-folder named "IST_TRAM_DAG_slearner"
if (!dir.exists(file.path(DIR, "IST_TRAM_DAG_slearner"))) {
  dir.create(file.path(DIR, "IST_TRAM_DAG_slearner"))
}

model  <- "IST_TRAM_DAG_slearner_"



# calculate ITE
test.results.tram <- tram.ITE(data = test.compl.data.transformed, train = dat.train.tf, test = dat.test.tf)


# extract train and test sets
data.dev.rs = test.results.tram[["data.dev.rs"]] %>%  as.data.frame()
data.val.rs = test.results.tram[["data.val.rs"]] %>%  as.data.frame()




# save calibration plots

file_name <- file.path(DIR, "IST_TRAM_DAG_slearner", paste0(model, 'train_calibration_plot.png'))
png(filename = file_name, width = 1600, height = 1600, res = 300)

res <- val.prob.ci.2(Y_prob_tram_dag, as.numeric(dat.train.tf[,39]-1))
# Close PNG device
dev.off()


# test
file_name <- file.path(DIR, "IST_TRAM_DAG_slearner", paste0(model, 'test_calibration_plot.png'))
png(filename = file_name, width = 1600, height = 1600, res = 300)

res <- val.prob.ci.2(Y_prob_tram_dag_test, as.numeric(dat.test.tf[,39]-1))
# Close PNG device
dev.off()




# check ITE distribution
plot(density(data.dev.rs$ITE))
lines(density(data.val.rs$ITE), col = "red")
legend("topright", legend = c("Train", "Test"), col = c("black", "red"), lty = 1)


# recreate ITE-Outcome
plot_outcome_ITE_tuned_rf(data.dev.rs = test.results.tram$data.dev.rs, data.val.rs = test.results.tram$data.val.rs, x_lim = c(-0.1,0.05))




# check treatment and outcome
str(test.results.tram$data.dev.rs$RXASP)      # 0/1 encoded -> change to N/Y
str(test.results.tram$data.dev.rs$OUTCOME6M)  # 1/2 encoded -> change to 0/1

test.results.tram_plot <- test.results.tram
test.results.tram_plot$data.dev.rs$RXASP <- ifelse(test.results.tram$data.dev.rs$RXASP == 0, "N", "Y")
test.results.tram_plot$data.val.rs$RXASP <- ifelse(test.results.tram$data.val.rs$RXASP == 0, "N", "Y")

test.results.tram_plot$data.dev.rs$OUTCOME6M <- ifelse(test.results.tram$data.dev.rs$OUTCOME6M == 1, 0, 1)
test.results.tram_plot$data.val.rs$OUTCOME6M <- ifelse(test.results.tram$data.val.rs$OUTCOME6M == 1, 0, 1)

# check treatment and outcome
str(test.results.tram_plot$data.dev.rs$RXASP)      # N/Y encoded
str(test.results.tram_plot$data.dev.rs$OUTCOME6M)  # 0/1 encoded


# ITE-ATE plot in terms of Risk Difference

breaks <- round(seq(-0.06, 0.0, by = 0.01), 2)
plot_ATE_vs_ITE_base_RiskDiff(model.results = test.results.tram_plot, 
                              breaks = breaks, 
                              delta_horizontal = 0.0008, 
                              ylim_delta = 0.2,
                              xlim_delta = 0.01)







# plot for slides

breaks <- round(seq(-0.06, 0.0, by = 0.01), 2)
plot_for_slides_IST(model.results = test.results.tram_plot, 
                    breaks = breaks, 
                    delta_horizontal = 0.0008, 
                    ylim_delta = 0.2, 
                    xlim_delta = 0.01)

# save plot for slides
file_name <- file.path(DIR, "IST_TRAM_DAG_slearner", paste0(model, 'density_ITE_ATE.png'))
png(filename = file_name, width = 2350, height = 1150, res = 300)
plot_for_slides_IST(model.results = test.results.tram_plot, 
                    breaks = breaks, 
                    delta_horizontal = 0.0008, 
                    ylim_delta = 0.2, 
                    xlim_delta = 0.01)
# Close PNG device
dev.off()




# ATE Estimated
mean(test.results.tram$data.dev.rs$ITE)
mean(test.results.tram$data.val.rs$ITE)

# ATE Observed
df <- test.results.tram  
mean(df$data.dev.rs$OUTCOME6M[df$data.dev.rs$RXASP == 1]-1) - mean(df$data.dev.rs$OUTCOME6M[df$data.dev.rs$RXASP == 0]-1)
mean(df$data.val.rs$OUTCOME6M[df$data.val.rs$RXASP == 1]-1) - mean(df$data.val.rs$OUTCOME6M[df$data.val.rs$RXASP == 0]-1)





# ATE Observed
df <- test.results.tram_plot
mean(df$data.dev.rs$OUTCOME6M[df$data.dev.rs$RXASP == "Y"]) - mean(df$data.dev.rs$OUTCOME6M[df$data.dev.rs$RXASP == "N"])
mean(df$data.val.rs$OUTCOME6M[df$data.val.rs$RXASP == "Y"]) - mean(df$data.val.rs$OUTCOME6M[df$data.val.rs$RXASP == "N"])


lm_dev <- lm(OUTCOME6M ~ RXASP, data = df$data.dev.rs)
lm_val <- lm(OUTCOME6M ~ RXASP, data = df$data.val.rs)



# ATE Estimated
mean(test.results.tram$data.dev.rs$ITE)
mean(test.results.tram$data.val.rs$ITE)

boot_result_dev <- ITE_CI_bootstrap(ITE = test.results.tram_plot$data.dev.rs$ITE, n_boot = 10000)
boot_result_val <- ITE_CI_bootstrap(ITE = test.results.tram_plot$data.val.rs$ITE, n_boot = 10000)

# make the table as Row (mean, lower, upper) and Column (ATE_obs_train, ATE_obs_test, ATE_est_train, ATE_est_test)

ATE_summary_table <- data.frame(
  row.names = c("mean", "lower", "upper"),
  ATE_obs_train = c(coef(lm_dev)[2], confint(lm_dev)[2, 1], confint(lm_dev)[2, 2]),
  ATE_obs_test = c(coef(lm_val)[2], confint(lm_val)[2, 1], confint(lm_val)[2, 2]),
  ATE_est_train = c(boot_result_dev$mean, boot_result_dev$ci_lower, boot_result_dev$ci_upper),
  ATE_est_test = c(boot_result_val$mean, boot_result_val$ci_lower, boot_result_val$ci_upper)
)



ATE_summary_table


# save ATE_summary_table, test.results.tram_plot in the created folder

save(ATE_summary_table, test.results.tram_plot, file = file.path(DIR, "IST_TRAM_DAG_slearner", "TRAM_DAG_slearner_results.RData"))










#################################################
# Following code was used for debugging, problem finding and re-calibration (with a GAM)
#################################################






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


