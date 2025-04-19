##### Mr Browns MAC ####
if (FALSE){
  reticulate::use_python("C:/ProgramData/Anaconda3/python.exe", required = TRUE)
}
# Get command-line arguments
args <- commandArgs(trailingOnly = TRUE)
if (length(args) == 0) {
  args <- c(3, 'cs')
  # args <- c(1, 'ls')
}

# args <- c(4, 'cs')
F32 <- as.numeric(args[1])
M32 <- args[2]
print(paste("FS:", F32, "M32:", M32))


#### A mixture of discrete and continuous variables ####
library(tensorflow)
library(keras)
library(mlt)
library(tram)
library(MASS)
library(tensorflow)
library(keras)
library(tidyverse)
source('utils_tf_ordinal_dummy.R')    # new tf file for ordinal data (predictors)

#### For TFP
library(tfprobability)
source('utils_tfp_ordinal_dummy.R')

##### Flavor of experiment ######

#### Saving the current version of the script into runtime
DIR = 'runs/experiment_5_all_ordinal_dummy_encoded_real_data_complex/run'
if (!dir.exists(DIR)) {
  dir.create(DIR, recursive = TRUE)
}
# Copy this file to the directory DIR
file.copy('/experiment_5_all_ordinal_dummy_encoded_real_data_complex.R', file.path(DIR, 'experiment_5_all_ordinal_dummy_encoded_real_data_complex.R'), overwrite=TRUE)


len_theta = 20 # Number of coefficients of the Bernstein polynomials
hidden_features_I = c(2,2,2,2) 
hidden_features_CS = c(2,5, 5,2)

if (F32 == 1){
  FUN_NAME = 'DPGLinear'
  f <- function(x) -0.3 * x
} else if (F32 == 2){
  f = function(x) 2 * x**3 + x
  FUN_NAME = 'DPG2x3+x'
} else if (F32 == 3){
  f = function(x) 0.5*exp(x)
  FUN_NAME = 'DPG0.5exp'
}

if (M32 == 'ls') {
  MA =  matrix(c(
    0, 'ls', 'ls',
    0,    0, 'ls',
    0,    0,   0), nrow = 3, ncol = 3, byrow = TRUE)
  MODEL_NAME = 'ModelLS'
} else{
  MA =  matrix(c(
    0, 'cs', 'ls',
    0,    0, 'ls',
    0,    0,   0), nrow = 3, ncol = 3, byrow = TRUE)
  MODEL_NAME = 'ModelCS'
}




FUN_NAME = 'RealData'

MA

# fn = 'triangle_mixed_DGPLinear_ModelLinear.h5'
# fn = 'triangle_mixed_DGPSin_ModelCS.h5'
fn = file.path(DIR, paste0('triangle_mixed_', FUN_NAME, '_', MODEL_NAME))
print(paste0("Starting experiment ", fn))

xs = seq(-1,1,0.1)

plot(xs, f(xs), sub=fn, xlab='x2', ylab='f(x2)', main='DGP influence of x2 on x3')




# =============================================================================
# Load and prepare data
# =============================================================================



library(ncdf4)

# read the file "enso_son.nc"  without raster stored in the same directory

enso <- nc_open("data/enso_son.nc")
enso_y <- ncvar_get(enso, "Year")
enso_v <- ncvar_get(enso, "enso")

# print the timeseries of enso
# plot(enso_y, enso_v, type = "l", xlab = "Year", ylab = "ENSO")


# read the file "iod_son.nc" stored in the same directory

iod <- nc_open("data/iod_son.nc")
print(iod)
iod_y <- ncvar_get(iod, "Year")
iod_v <- ncvar_get(iod, "iod")

# print the timeseries of iod
# plot(iod_y, iod_v, type = "l", xlab = "Year", ylab = "IOD")


# read the file "precip_au_son.nc" stored in the same directory

precip <- nc_open("data/precip_au_son.nc")
print(precip)
precip_y <- ncvar_get(precip, "year")
precip_v <- ncvar_get(precip, "precip")


# plot all together
par(mfrow=c(3,1))
plot(enso_y, enso_v, type = "l", xlab = "Year", ylab = "ENSO")
plot(iod_y, iod_v, type = "l", xlab = "Year", ylab = "IOD")
plot(precip_y, precip_v, type = "l", xlab = "Year", ylab = "AU Precipitation")



#===============================================================================
# Data preprocessing
#===============================================================================

# standardize (zero to mean, unit variance)
ENSO <- (enso_v - mean(enso_v))/sd(enso_v)
IOD <- (iod_v - mean(iod_v))/sd(iod_v)
AU <- (precip_v - mean(precip_v))/sd(precip_v)

# 3 histograms of the standardized data
par(mfrow=c(3,1))
hist(ENSO, main = "ENSO", xlab = "ENSO")
hist(IOD, main = "IOD", xlab = "IOD")
hist(AU, main = "Precipitation", xlab = "Precipitation")

# detrend

ENSO_detrended <- residuals(lm(ENSO ~ enso_y))
IOD_detrended <- residuals(lm(IOD ~ iod_y))
AU_detrended <- residuals(lm(AU ~ precip_y))

# plot the 3 detrended variables (Title "standardized-detrended")
par(mfrow=c(3,1))

plot(enso_y, ENSO_detrended, type = "l", xlab = "Year", ylab = "ENSO", 
     main = "standardized-detrended")
plot(iod_y, IOD_detrended, type = "l", xlab = "Year", ylab = "IOD",
     main = "standardized-detrended")
plot(precip_y, AU_detrended, type = "l", xlab = "Year", ylab = "AU Precipitation",
     main = "standardized-detrended")

# 3 histograms of the detrended data
par(mfrow=c(3,1))
hist(ENSO_detrended, main = "ENSO", xlab = "ENSO")
hist(IOD_detrended, main = "IOD", xlab = "IOD")
hist(AU_detrended, main = "Precipitation", xlab = "Precipitation")



raw_df <- data.frame(
  ENSO = ENSO_detrended,
  IOD = IOD_detrended,
  AU = AU_detrended
)


# Pairsplot of variables (ENSO, IOD, AU)
pairs(raw_df)


# # Step 2: Convert Numeric Variables to ordinal
raw_df$ENSO_ordinal <- cut(raw_df$ENSO, breaks = quantile(raw_df$ENSO, probs = c(0, 1/3, 2/3, 1)), labels = c(1, 2, 3), include.lowest = TRUE)
raw_df$IOD_ordinal <- cut(raw_df$IOD, breaks = quantile(raw_df$IOD, probs = c(0, 1/3, 2/3, 1)), labels = c(1, 2, 3), include.lowest = TRUE)
raw_df$AU_binary <- cut(raw_df$AU, breaks = quantile(raw_df$AU, probs = c(0, 0.5, 1)), labels = c(1, 2), include.lowest = TRUE)

# 
raw_df



#===============================================================================
# TRAM-DAG
#===============================================================================




set.seed(123)

# train test split (90% train, 10% test)
train_indizes <- sample(1:nrow(raw_df), 0.9*nrow(raw_df))
train_df <- raw_df[train_indizes,]
test_df <- raw_df[-train_indizes,]


# create the data object

dgp <- function(n_obs, doX=c(NA, NA, NA), seed=123, data = NULL) {
  
  # n_obs <- 1000
  # doX <- c(NA, NA, NA)
  # seed=-1
  # data = train_df
  
  set.seed(seed)
  
  
  if (!is.null(data)){
    
    if (is.na(doX[1])){
      X_1 = data$ENSO_ordinal
      X_1 = ordered(X_1, levels=c(1,2,3))
    } else{
      X_1 = rep(doX[1], n_obs)
    }
    #hist(X_1)
    
    if (is.na(doX[2])){
      
      X_2 = data$IOD_ordinal
      X_2 = ordered(X_2, levels=c(1,2,3))
      
    } else{
      X_2 = rep(doX[2], n_obs)
    }
    #hist(X_2)
    
    if (is.na(doX[3])){
      
      X_3 = data$AU_binary
      
      X_3 = ordered(X_3, levels=c(1,2))
      
    } else{
      X_3 = rep(doX[3], n_obs)
    }
  } 
  
  
  
  # par(mfrow=c(1,3))
  # plot(X_1)
  # plot(X_2)
  # plot(X_3)
  
  
  # define dummy encoded variables
  
  
  if (is.na(doX[1])){
    X_1_t1 = as.numeric(data$ENSO_ordinal == 2)  # 1 if level 2 else 0
    X_1_t2 = as.numeric(data$ENSO_ordinal == 3)  # 1 if level 3 else 0
  } else{
    X_1_t1 = rep(as.numeric(doX[1] == 2), n_obs)
    X_1_t2 = rep(as.numeric(doX[1] == 3), n_obs)
  }
  
  if (is.na(doX[2])){
    X_2_t1 = as.numeric(data$IOD_ordinal == 2)  
    X_2_t2 = as.numeric(data$IOD_ordinal == 3)  
  } else{
    X_2_t1 = rep(as.numeric(doX[2] == 2), n_obs)
    X_2_t2 = rep(as.numeric(doX[2] == 3), n_obs)
  }
  
  if (is.na(doX[3])){
    X_3_t1 = as.numeric(data$AU_binary == 2)  # Only one threshold for binary variable
  } else{
    X_3_t1 = rep(as.numeric(doX[3] == 2), n_obs)
  }
  
  
  
  # Define adjacency matrix
  A <- matrix(c(0, 1, 1, 0, 0, 1, 0, 0, 0), nrow = 3, ncol = 3, byrow = TRUE)
  # Put smaples in a dataframe
  dat.orig =  data.frame(x1 = X_1, x2 = X_2, x3 = X_3)
  # Put samples in a tensor
  dat.tf = tf$constant(as.matrix(dat.orig), dtype = 'float32')
  
  # calculate 5% quantiles of the sampled variables
  q1 = c(1,3)
  q2 = c(1,3)
  q3 = c(1,2) #no quantiles for ordinal outcomes
  
  
  # same for encoded variables
  A_e <- matrix(c(0, 1, 1,0, 1, 1, 0, 0, 1, 0, 0, 1,0, 0, 0), nrow = 5, ncol = 3, byrow = TRUE)
  dat.encoded = data.frame(x1_t1 = X_1_t1, x1_t2 = X_1_t2, 
                           x2_t1 = X_2_t1, x2_t2 = X_2_t2, 
                           x3_t1 = X_3_t1)
  dat.encoded.tf = tf$constant(as.matrix(dat.encoded), dtype = 'float32')
  
  # quantiles of variables (only 0 and 1)
  q1_e = c(0,1)
  q2_e = c(0,1)
  q3_e = c(0,1)
  q4_e = c(0,1)
  q5_e = c(0,1)
  
  
  
  
  
  
  # return samples in a tensor, the original dataframe, the min and max values 
  # of the variables (the quantiles), and the adjacency matrix
  return(list(
    df_orig=dat.tf, 
    df_R = dat.orig,
    #min =  tf$reduce_min(dat.tf, axis=0L),
    #max =  tf$reduce_max(dat.tf, axis=0L),
    min = tf$constant(c(q1[1], q2[1], q3[1]), dtype = 'float32'),
    max = tf$constant(c(q1[2], q2[2], q3[2]), dtype = 'float32'),
    type = c('o', 'o', 'o'),
    A=A,
    
    df_encoded = dat.encoded.tf,
    df_R_encoded = dat.encoded,
    
    min_encoded = tf$constant(c(q1_e[1], q2_e[1], q3_e[1], q4_e[1], q5_e[1]), dtype = 'float32'),
    max_encoded = tf$constant(c(q1_e[2], q2_e[2], q3_e[2], q4_e[2], q5_e[2]), dtype = 'float32'),
    
    type_encoded = c('o', 'o', 'o', 'o', 'o'),
    A_e = A_e
    
  ))
} 

# generate data
train <- dgp(nrow(train_df), seed=123, data = train_df)
test <- dgp(nrow(test_df), seed=123, data = test_df)

# example, X1 on ordinal scale
train$df_R$x1

# can be reconstructed as


# Reconstruct x1, x2, and x3
x1_reconstructed <- as.matrix(train$df_R_encoded[, c("x1_t1", "x1_t2")]) %*% c(1, 2) + 1
x1_reconstructed
# x2_reconstructed <- as.matrix(train$df_R_encoded[, c("x2_t1", "x2_t2")]) %*% c(1, 2) + 1
# x3_reconstructed <- ifelse(train$df_R_encoded[, "x3_t1"] == 1, 1, 2)

# Combine into a dataframe
# df_reconstructed <- data.frame(x1 = x1_reconstructed, x2 = x2_reconstructed, x3 = x3_reconstructed)


# ordinal scale
global_min = train$min
global_max = train$max
data_type = train$type


# endoded scale
global_min_encoded = train$min_encoded
global_max_encoded = train$max_encoded
data_type_encoded = train$type_encoded


len_theta_max = len_theta
for (i in 1:nrow(MA)){ #Maximum number of coefficients (BS and Levels - 1 for the ordinal)
  if (train$type[i] == 'o'){
    len_theta_max = max(len_theta_max, nlevels(train$df_R[,i]) - 1)
  }
}

MA_encoded = ifelse(train$A_e, "ls", 0)
MA_encoded[1:2, 2] <- c('cs', 'cs')
MA_encoded

param_model = create_param_model(MA, MA_encoded = MA_encoded,  hidden_features_I=hidden_features_I, len_theta=len_theta, hidden_features_CS=hidden_features_CS, train = train)

# input the samples into the model
# h_params = param_model(train$df_orig)
h_params = param_model(train$df_encoded)

# Attention: in loss, the levels in ordinal varaibles are hard-coded !!!
struct_dag_loss(t_i=train$df_encoded, h_params=h_params)

optimizer = optimizer_adam(learning_rate = 0.005)
param_model$compile(optimizer, loss=struct_dag_loss)
param_model$evaluate(x = train$df_encoded, y=train$df_encoded, batch_size =  7L) # 6.29
summary(param_model)

# show the beta layer
param_model$get_layer(name = "beta")$get_weights()

##### Training ####
num_epochs <- 10000
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
    ws <- data.frame(w12 = numeric())
    train_loss <- numeric()
    val_loss <- numeric()
    
    # Training loop
    for (e in 1:num_epochs) {
      # e <- 1
      print(paste("Epoch", e))
      hist <- param_model$fit(x = train$df_encoded, y = train$df_encoded, 
                              epochs = 1L, verbose = TRUE, 
                              validation_data = list(test$df_encoded, test$df_encoded))
      
      # Append losses to history
      train_loss <- c(train_loss, hist$history$loss)
      val_loss <- c(val_loss, hist$history$val_loss)
      
      # Extract specific weights
      w <- param_model$get_layer(name = "beta")$get_weights()[[1]]
      
      ws <- rbind(ws, data.frame(w12 = w[1, 2], w13 = w[1, 3], w22 = w[2, 2],
                                 w23 = w[2, 3], w33 = w[3, 3], w43 = w[4, 3]))
    }
    # Save the model
    f <- "realData"
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
#pdf(paste0('loss_',fn,'.pdf'))
epochs = length(train_loss)
plot(1:length(train_loss), train_loss, type='l', main='Normal Training (green is valid)', ylim = c(2.45, 13))
lines(1:length(train_loss), val_loss, type = 'l', col = 'green')

# Last 50
diff = max(epochs - 100,0)
plot(diff:epochs, val_loss[diff:epochs], type = 'l', col = 'green', main='Last 50 epochs')
lines(diff:epochs, train_loss[diff:epochs], type='l')

# plot(1:epochs, ws[,1], type='l', main='Coef', ylim=c(-0.5, 3))#, ylim=c(0, 6))
# abline(h=2, col='green')
# lines(1:epochs, ws[,2], type='l', ylim=c(0, 3))
# abline(h=0.2, col='green')
# lines(1:epochs, ws[,3], type='l', ylim=c(0, 3))
# abline(h=-0.3, col='green')



# fit Polr form tram package (negative shift, so different as in Colr)
# fit_21 <- Colr(x2 ~ x1, data = train$df_R)  # Colr for continuous outcome (positive shift)

# logistic regression of the form x3 ~ x1 + x2 (x3 is binary)
fit_321 <- glm(x3 ~ x1 + x2, data = train$df_R, family = binomial(link = "logit"))
summary(fit_321)

# fit_21 <- glm(x2 ~ x1 , data = train$df_R, family = binomial(link = "logit"))
# summary(fit_21)


wes <- round(ws[nrow(ws),],3)


contrasts(train$df_R$x1) <- contr.treatment(nlevels(train$df_R$x1))
contrasts(train$df_R$x2) <- contr.treatment(nlevels(train$df_R$x2))
contrasts(train$df_R$x3) <- contr.treatment(nlevels(train$df_R$x3))
polr(x2 ~ x1, data = train$df_R, Hess=TRUE)

fit_321 <- glm(x3 ~ x1 + x2, data = train$df_R, family = binomial(link = "logit"))


fit_21 <- polr(x2 ~ x1, data = train$df_R, Hess=TRUE)


comp <- round(rbind(-c(coef(fit_21)[1], coef(fit_321)[2], 
                       coef(fit_21)[2], coef(fit_321)[3], 
                       coef(fit_321)[4], coef(fit_321)[5]),
                    wes), 3)
rownames(comp) <- c("glm/polr", "TRAM-DAG")
comp




# do the same plot for all 5 weights previously saved
p <- ggplot(ws, aes(x=1:nrow(ws))) + 
  # geom_line(aes(y=w12, color='x1.1 --> x2')) + 
  # geom_line(aes(y=w22, color='x1.2 --> x2')) + 
  geom_line(aes(y=w13, color='x1.1 --> x3')) + 
  geom_line(aes(y=w23, color='x1.2 --> x3')) + 
  geom_line(aes(y=w33, color='x2.1 --> x3')) + 
  geom_line(aes(y=w43, color='x2.2 --> x3')) + 
  # geom_hline(aes(yintercept=coef(fit_21)[2], color='Colr/glm'), linetype=2) +
  # geom_hline(aes(yintercept=coef(fit_21)[3], color='Colr/glm'), linetype=2) +
  # geom_hline(aes(yintercept=-(coef(fit_21)[1]), color='x1.1 --> x2'), linetype=2) +
  # geom_hline(aes(yintercept=-(coef(fit_21)[2]), color='x1.2 --> x2'), linetype=2) +
  geom_hline(aes(yintercept=-(coef(fit_321)[2]), color='x1.1 --> x3'), linetype=2) +
  geom_hline(aes(yintercept=-(coef(fit_321)[3]), color='x1.2 --> x3'), linetype=2) +
  geom_hline(aes(yintercept=-(coef(fit_321)[4]), color='x2.1 --> x3'), linetype=2) +
  geom_hline(aes(yintercept=-(coef(fit_321)[5]), color='x2.2 --> x3'), linetype=2) +
  labs(x='Epoch', y='Coefficients') +
  theme_minimal() +
  theme(legend.title = element_blank())  # Removes the legend title
p

file_name <- paste0(fn, "_coef_epoch.pdf")
if (TRUE){
  file_path <- file.path("runs/experiment_5_all_ordinal_dummy_encoded_real_data_complex/run/", basename(file_name))
  ggsave(file_path, plot = p, width = 8, height = 6/2)
}

###### Coefficient Plot for Paper #######
# if (FALSE){
#   p = ggplot(ws, aes(x=1:nrow(ws))) + 
#     geom_line(aes(y=w12, color="beta12")) + 
#     geom_line(aes(y=w13, color="beta13")) + 
#     geom_line(aes(y=w23, color="beta23")) + 
#     geom_hline(aes(yintercept=2, color="beta12"), linetype=2) +
#     geom_hline(aes(yintercept=0.2, color="beta13"), linetype=2) +
#     geom_hline(aes(yintercept=-0.3, color="beta23"), linetype=2) +
#     scale_color_manual(
#       values=c('beta12'='skyblue', 'beta13'='red', 'beta23'='darkgreen'),
#       labels=c(expression(beta[12]), expression(beta[13]), expression(beta[23]))
#     ) +
#     labs(x='Epoch', y='Coefficients') +
#     theme_minimal() +
#     theme(
#       legend.title = element_blank(),   # Removes the legend title
#       legend.position = c(0.85, 0.17),  # Adjust this to position the legend inside the plot (lower-right)
#       legend.background = element_rect(fill="white", colour="black")  # Optional: white background with border
#     )
#   
#   p
#   file_name <- paste0(fn, "_coef_epoch.pdf")
#   if (FALSE){
#     file_path <- file.path("~/Library/CloudStorage/Dropbox/Apps/Overleaf/tramdag/figures", basename(file_name))
#     ggsave(file_path, plot = p, width = 8, height = 6/2)
#   }
# }

# estimated beta weights
param_model$get_layer(name = "beta")$get_weights() * param_model$get_layer(name = "beta")$mask


##### Checking observational distribution ####
# s = do_dag_struct(param_model, train$A, doX=c(NA, NA, NA), num_samples = 5000)
##### Checking observational distribution ####
s = do_dag_struct(param_model, MA = train$A, MA_encoded = MA_encoded, doX=c(NA, NA, NA), num_samples = 5000, data = train)
plot(table(train$df_R[,3])/sum(table(train$df_R[,3])), ylab='Probability ', 
     main='Black = Observations, Red samples from TRAM-DAG',
     xlab='X3')
table(train$df_R[,3])/sum(table(train$df_R[,3]))
points(as.numeric(table(s[,3]$numpy()))/5000, col='red', lty=2)
table(s[,3]$numpy())/5000

par(mfrow=c(1,3))
# for (i in 1:2){
#   hist(train$df_orig$numpy()[,i], freq=FALSE, 100,main=paste0("X",i, " red: ours, black: data"), xlab='samples')
#   #hist(train$df_orig$numpy()[,i], freq=FALSE, 100,main=paste0("X_",i))
#   lines(density(s[,i]$numpy()), col='red')
# }
for (i in 1:2){
  plot(table(train$df_orig$numpy()[,i])/sum(table(train$df_orig$numpy()[,i])),main=paste0("X",i, " red: ours, black: data"), xlab='samples')
  #hist(train$df_orig$numpy()[,i], freq=FALSE, 100,main=paste0("X_",i))
  points(as.numeric(table(s[,i]$numpy()))/5000, col='red')
}
plot(table(train$df_R[,3])/sum(table(train$df_R[,3])), ylab='Probability ', 
     main='Black = Observations, Red samples from TRAM-DAG',
     xlab='X3')
table(train$df_R[,3])/sum(table(train$df_R[,3]))
points(as.numeric(table(s[,3]$numpy()))/5000, col='red', lty=2)
table(s[,3]$numpy())/5000
par(mfrow=c(1,1))

######### Simulation of do-interventions #####
# doX=c(1, NA, NA)
# dx0.2 = dgp(10000, doX=doX)
# dx0.2$df_orig$numpy()[1:5,]
# 
# 
# doX=c(0.7, NA, NA)
# dx7 = dgp(10000, doX=doX)
# #hist(dx0.2$df_orig$numpy()[,2], freq=FALSE,100)
# mean(dx7$df_orig$numpy()[,2]) - mean(dx0.2$df_orig$numpy()[,2])  
# mean(dx7$df_orig$numpy()[,3]) - mean(dx0.2$df_orig$numpy()[,3])  
# 
# s_dag = do_dag_struct(param_model, train$A, doX=c(0.2, NA, NA))
# hist(dx0.2$df_orig$numpy()[,2], freq=FALSE, 50, main='X2 | Do(X1=0.2)', xlab='samples', 
#      sub='Histogram from DGP with do. red:TRAM_DAG')
# sample_dag_0.2 = s_dag[,2]$numpy()
# lines(density(sample_dag_0.2), col='red', lw=2)
# m_x2_do_x10.2 = median(sample_dag_0.2)
# 
# i = 3 
# d = dx0.2$df_orig$numpy()[,i]
# plot(table(d)/length(d), ylab='Probability ', 
#      main='X3 | do(X1=0.2)',
#      xlab='X3', ylim=c(0,0.6),  sub='Black DGP with do. red:TRAM_DAG')
# points(as.numeric(table(s_dag[,3]$numpy()))/nrow(s_dag), col='red', lty=2)

###### Figure for paper ######
if (TRUE){
  doX=c(NA, NA, NA)
  s_obs_fitted = do_dag_struct(param_model, MA = train$A, MA_encoded = MA_encoded, 
                               doX=doX, num_samples = 5000, 
                               data = train)$numpy()
  
  dx1 = 1
  # dx1 = 2
  doX=c(dx1, NA, NA)
  s_do_fitted = do_dag_struct(param_model, MA = train$A, MA_encoded = MA_encoded, 
                              doX=doX, num_samples = 5000, 
                              data = train)$numpy()
  
  df = data.frame(vals=s_obs_fitted[,1], type='Model', X=1, L='L0')
  df = rbind(df, data.frame(vals=s_obs_fitted[,2], type='Model', X=2, L='L0'))
  df = rbind(df, data.frame(vals=s_obs_fitted[,3], type='Model', X=3, L='L0'))
  
  df = rbind(df, data.frame(vals=as.numeric(train$df_R[,1]), type='Real Data', X=1, L='L0'))
  df = rbind(df, data.frame(vals=as.numeric(train$df_R[,2]), type='Real Data', X=2, L='L0'))
  df = rbind(df, data.frame(vals=as.numeric(train$df_R[,3]), type='Real Data', X=3, L='L0'))
  
  df = rbind(df, data.frame(vals=s_do_fitted[,1], type='Model', X=1, L='L1'))
  df = rbind(df, data.frame(vals=s_do_fitted[,2], type='Model', X=2, L='L1'))
  df = rbind(df, data.frame(vals=s_do_fitted[,3], type='Model', X=3, L='L1'))
  
  # d = dgp(10000, doX=doX)$df_R
  # df = rbind(df, data.frame(vals=d[,1], type='DGP', X=1, L='L1'))
  # df = rbind(df, data.frame(vals=d[,2], type='DGP', X=2, L='L1'))
  # df = rbind(df, data.frame(vals=as.numeric(d[,3]), type='DGP', X=3, L='L1'))
  
  # plot df
  
  p = ggplot() +
    # For X = 1 and X = 2, use position = "identity" (no dodging)
    # geom_histogram(data = subset(df, X != 3), 
    #                aes(x=vals, col=type, fill=type, y=..density..), 
    #                position = "identity", alpha=0.4) +
    # For X = 3, use a bar plot for discrete data
    # geom_bar(data = subset(df, X == 3), 
    #          aes(x=vals, y=..prop.. * 4,  col=type, fill=type), 
    #          position = "dodge", alpha=0.4, size = 0.5)+
    
    geom_bar(data = df, 
             aes(x=vals, y=..prop.. ,  col=type, fill=type), 
             position = "dodge", alpha=0.4, size = 0.5)+
    #limit between 0,1 but not removing the data
    coord_cartesian(ylim = c(0, 1)) +
    facet_grid(L ~ X, scales = 'free',
               labeller = as_labeller(c('1' = 'X1', '2' = 'X2', '3' = 'X3', 'L1' = paste0('Do X1=',dx1), 'L0' = 'Obs')))+ 
    labs(y = "Probability", x='')  + # Update y-axis label
    theme_minimal() +
    theme(
      legend.title = element_blank(),   # Removes the legend title
      legend.position = c(0.17, 0.25),  # Adjust this to position the legend inside the plot (lower-right)
      legend.background = element_rect(fill="white", colour="white")  # Optional: white background with border
    )
  p
  file_name <- paste0(fn, "_L0_L1.pdf")
  ggsave(file_name, plot=p, width = 8, height = 6)
  if (TRUE){
    file_path <- file.path("runs/experiment_5_all_ordinal_dummy_encoded_real_data_complex/run/", basename(file_name))
    print(file_path)
    ggsave(file_path, plot=p, width = 8/2, height = 6/2)
  }
  
}




s_dag = do_dag_struct(param_model, train$A, doX=c(0.7, NA, NA))
i = 2
ds = dx7$df_orig$numpy()[,i]
hist(ds, freq=FALSE, 50, main='X2 | Do(X1=0.7)', xlab='samples', 
     sub='Histogram from DGP with do. red:TRAM_DAG')
sample_dag_07 = s_dag[,i]$numpy()
lines(density(sample_dag_07), col='red', lw=2)
m_x2_do_x10.7 = median(sample_dag_07)
m_x2_do_x10.7 - m_x2_do_x10.2

###### Comparison of estimated f(x2) vs TRUE f(x2) #######
shift_23 = shift_13 = shift_12 = cs_12 = xs = list(c(0,0), c(1,0), c(0,1))
idx0 = 1 # c(0,0) -> reference #Index of 0 xs needs to be odd
for (i in 1:length(xs)){
  #i = 1
  x = xs[[i ]]
  # Varying x1
  # X = tf$constant(c(x, 2, 1), shape=c(1L,3L)) 
  X = tf$constant(c(x, 1, 0 , 1), shape=c(1L,5L)) 
  shift_13[i] = param_model(X)[1,3,2]$numpy() #2=LS Term X1->X3
  cs_12[i] = param_model(X)[1,2,1]$numpy() #2=CS Term X1->X2
  
  #Varying x2
  # X = tf$constant(c(1, x, 1), shape=c(1L,3L)) 
  X = tf$constant(c(0,0, x, 1), shape=c(1L,5L)) 
  # cs_23[i] = param_model(X)[1,3,1]$numpy() #1=CS Term
  shift_23[i] = param_model(X)[1,3,2]$numpy() #2-LS Term X2-->X3 (Ms. Whites Notation)
}

par(mfrow=c(2,2))

plot(1:length(xs), cs_12, main='LS-Term (black DGP, red Ours)', 
     sub = 'Effect of x1 on x2',
     xlab='x1', col='red')
# abline(0, 2)

delta_0 = shift_13[[idx0]] - 0
plot(1:length(xs), unlist(shift_13) - delta_0, main='LS-Term (black DGP, red Ours)', 
     sub = paste0('Effect of x1 on x3, delta_0 ', round(delta_0,2)),
     xlab='x1', col='red')
# abline(0, .2)

delta_0 = shift_23[[idx0]] - 0
plot(1:length(xs), unlist(shift_23) - delta_0, main='LS-Term (black DGP, red Ours)', 
     sub = paste0('Effect of x2 on x3, delta_0 ', round(delta_0,2)),
     xlab='x1', col='red')
# abline(0, .2)



if (F32 == 1){ #Linear DGP
  if (MA[2,3] == 'ls'){
    delta_0 = shift_23[idx0] - f(0)
    plot(xs, shift_23 - delta_0, main='LS-Term (black DGP, red Ours)', 
         sub = paste0('Effect of x2 on x3, delta_0 ', round(delta_0,2)),
         xlab='x2', col='red')
    #abline(shift_23[length(shift_23)/2], -0.3)
    # abline(0, -0.3)
  } 
  if (MA[2,3] == 'cs'){
    plot(xs, cs_23, main='CS-Term (black DGP, red Ours)', xlab='x2',  
         sub = 'Effect of x2 on x3',col='red')
    
    # abline(cs_23[idx0], -0.3)  
  }
} else{ #Non-Linear DGP
  if (MA[2,3] == 'ls'){
    delta_0 = shift_23[idx0] - f(0)
    plot(xs, shift_23 - delta_0, main='LS-Term (black DGP, red Ours)', 
         sub = paste0('Effect of x2 on x3, delta_0 ', round(delta_0,2)),
         xlab='x2', col='red')
    # lines(xs, f(xs))
  } else if (MA[2,3] == 'cs'){
    plot(xs, cs_23 + ( -cs_23[idx0] + f(0) ),
         ylab='CS',
         main='CS-Term (black DGP f2(x), red Ours)', xlab='x2',  
         sub = 'Effect of x2 on x3',col='red')
    # lines(xs, f(xs))
  } else{
    print(paste0("Unknown Model ", MA[2,3]))
  }
}
#plot(xs,f(xs), xlab='x2', main='DGP')
par(mfrow=c(1,1))

h_params = param_model(train$df_orig)

# check the model output and meaning
h_params[1,,] # outputs for X1, X2 and X3 for first triple of observations [x1, x2, x3]
h_params[1,1,] # outputs for X1 for first triple of observations [x1, x2, x3]
h_params[1,1,1:2] # CS and LS for X1
h_params[1,1,3:dim(h_params)[3]] # Theta for X1 
# (note that for ordinal only first (#classes-1) thetas are used in the loss,
# the others can be ignored.)




if (FALSE){
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
  
  ### Continiuous dimensions
  #### At least one continuous dimension exits
  h_I = h_dag_extra(t_i[,cont_dims, drop=FALSE], theta[,cont_dims,1:len_theta,drop=FALSE], k_min[cont_dims], k_max[cont_dims]) 
  
  h_12 = h_I + h_LS[,cont_dims, drop=FALSE] + h_CS[,cont_dims, drop=FALSE]
  
  ### Ordingal Dimensions
  B = tf$shape(t_i)[1]
  col = 3
  nol = tf$cast(k_max[col] - 1L, tf$int32) # Number of cut-points in respective dimension
  theta_ord = theta[,col,1:nol,drop=TRUE] # Intercept
  h_3 = theta_ord + h_LS[,col, drop=FALSE] + h_CS[,col, drop=FALSE]
  
  ####### DGP Transformations #######
  X_1 = t_i[,1]$numpy()
  X_2 = t_i[,2]$numpy()
  h2_DGP = 5 *X_2 + 2 * X_1
  plot(h2_DGP[1:2000], h_12[1:2000,2]$numpy())
  abline(0,1,col='red')
  
  h2_DGP_I = 5*X_2
  h2_M_I = h_I[,2]
  
  plot(h2_DGP_I, h2_M_I)
  abline(0,1,col='red')
  
  h_3 #Model
  
  ##### DGP 
  theta_k = c(-2, 0.42, 1.02)
  n_obs = B$numpy()
  h_3_DPG = matrix(, nrow=n_obs, ncol=3)
  for (i in 1:n_obs){
    h_3_DPG[i,] = theta_k + 0.2 * X_1[i] + f(X_2[i]) #- 0.3 * X_2[i]
  }
  
  plot(h_3_DPG[1:2000,3], h_3[1:2000,3]$numpy())
  abline(0,1,col='green')
  
  #LS
  plot(-0.2*X_1, h_LS[,3]$numpy())
  abline(0,1,col='green')
  
  #LS
  plot(f(X_2), h_CS[,3]$numpy())
  abline(0,1,col='green')
}






