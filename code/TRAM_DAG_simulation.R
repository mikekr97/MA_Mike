
###############################################################

# Code for Experiment 1: TRAM-DAG (simulation study) 

# simple example how a DAG can model functional relationships in a DAG

###############################################################




##### Mr Browns MAC ####
if (FALSE){
  reticulate::use_python("C:/ProgramData/Anaconda3/python.exe", required = TRUE)
}

#### Experiment 1: simple TRAM-DAG simulation ####

# X1, X2 continuous
# X3 ordinal with 4 levels


#### Libraries
library(tensorflow)
library(keras)
library(mlt)
library(tram)
library(MASS)
library(tidyverse)

#### For TF
source('code/utils/utils_tf.R')
# source('code/utils/ITE_utils.R')

#### For TFP
library(tfprobability)
source('code/utils/utils_tfp.R')



#### Saving the current version of the script into runtime
DIR = 'runs/TRAM_DAG_simulation/run'
if (!dir.exists(DIR)) {
  dir.create(DIR, recursive = TRUE)
}
# Copy this file to the directory DIR
file.copy('/TRAM_DAG_simulation.R', file.path(DIR, 'TRAM_DAG_simulation.R'), overwrite=TRUE)


### Choose type of model (in thesis -> cs from X2 to X3)
# Get command-line arguments
args <- commandArgs(trailingOnly = TRUE)
if (length(args) == 0) {
  args <- c(3, 'cs')
  args <- c(1, 'ls')
}
args <- c(3, 'cs')
F32 <- as.numeric(args[1])
M32 <- args[2]
print(paste("FS:", F32, "M32:", M32))


len_theta = 20 # Number of coefficients of the Bernstein polynomials
hidden_features_I = c(2,2,2,2) # NN for CI terms (not used in this experiment)
hidden_features_CS = c(2,2,2,2) # NN for CS terms


#########################
# Data generating process (DGP)
#########################


# CS = DPG0.5exp
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

# plot complex shift function
xs = seq(-2,2,0.1)
plot(xs, f(xs), sub=fn, xlab='x2', ylab='f(x2)', main='DGP influence of x2 on x3')


# MA = with complex shift
if (M32 == 'ls') {
  MA =  matrix(c(
    0, 'ls', 'ls', 
    0,    0, 'ls', 
    0,    0,   0), nrow = 3, ncol = 3, byrow = TRUE)
  MODEL_NAME = 'ModelLS'
} else{
  MA =  matrix(c(
    0, 'ls', 'ls', 
    0,    0, 'cs', 
    0,    0,   0), nrow = 3, ncol = 3, byrow = TRUE)
  MODEL_NAME = 'ModelCS'
}

MA


# Model path
fn = file.path(DIR, paste0(FUN_NAME, '_', MODEL_NAME))
print(paste0("Starting experiment ", fn))



##### DGP ########
dgp <- function(n_obs, doX=c(NA, NA, NA), SEED=123) {
  #n_obs = 1e5 n_obs = 10
  set.seed(SEED)
  #Sample X_1 from GMM with 2 components
  if (is.na(doX[1])){
    X_1_A = rnorm(n_obs, 0.25, 0.1)
    X_1_B = rnorm(n_obs, 0.73, 0.05)
    X_1 = ifelse(sample(1:2, replace = TRUE, size = n_obs) == 1, X_1_A, X_1_B)
  } else{
    X_1 = rep(doX[1], n_obs)
  }
  #hist(X_1)
  
  # Sampling according to colr
  if (is.na(doX[2])){
    U2 = runif(n_obs)
    x_2_dash = qlogis(U2)
    #x_2_dash = h_0(x_2) + beta * X_1
    #x_2_dash = 0.42 * x_2 + 2 * X_1
    X_2 = 1/0.42 * (x_2_dash - 2 * X_1)
    X_2 = 1/5. * (x_2_dash - 0.4 * X_1) # 0.39450
    X_2 = 1/5. * (x_2_dash - 1.2 * X_1) 
    X_2 = 1/5. * (x_2_dash - 2 * X_1)  # 
    
    
  } else{
    X_2 = rep(doX[2], n_obs)
  }
  
  #hist(X_2)
  #ds = seq(-5,5,0.1)
  #plot(ds, dlogis(ds))
  
  if (is.na(doX[3])){
    # x3 is an ordinal variable with K = 4 levels x3_1, x3_2, x3_3, x3_4
    # h(x3 | x1, x2) = h0 + gamma_1 * x1 + gamma_2 * x2
    # h0(x3_1) = theta_1, h0(x_3_2) =  theta_2, h0(x_3_3) = theta_3 
    theta_k = c(-2, 0.42, 1.02)
    
    h = matrix(, nrow=n_obs, ncol=3)
    for (i in 1:n_obs){
      h[i,] = theta_k + 0.2 * X_1[i] + f(X_2[i]) #- 0.3 * X_2[i]
    }
    
    U3 = rlogis(n_obs)
    # chooses the correct X value if U3 is smaller than -2 that is level one if it's between -2 and 0.42 it's level two answer on
    x3 = rep(1, n_obs)
    x3[U3 > h[,1]] = 2
    x3[U3 > h[,2]] = 3
    x3[U3 > h[,3]] = 4
    x3 = ordered(x3, levels=1:4)
  } else{
    x3 = rep(doX[3], n_obs)
  }
  
  #hist(X_3)
  A <- matrix(c(0, 1, 1, 0,0,1,0,0,0), nrow = 3, ncol = 3, byrow = TRUE)
  dat.orig =  data.frame(x1 = X_1, x2 = X_2, x3 = x3)
  dat.tf = tf$constant(as.matrix(dat.orig), dtype = 'float32')
  
  q1 = quantile(dat.orig[,1], probs = c(0.05, 0.95)) 
  q2 = quantile(dat.orig[,2], probs = c(0.05, 0.95))
  q3 = c(1, 4) #No Quantiles for ordinal data
  
  
  return(list(
    df_orig=dat.tf, 
    df_R = dat.orig,
    #min =  tf$reduce_min(dat.tf, axis=0L),
    #max =  tf$reduce_max(dat.tf, axis=0L),
    min = tf$constant(c(q1[1], q2[1], q3[1]), dtype = 'float32'),
    max = tf$constant(c(q1[2], q2[2], q3[2]), dtype = 'float32'),
    type = c('c', 'c', 'o'),
    A=A, 
    theta_k = theta_k))
} 

# generate train and test data
train = dgp(20000)
test  = dgp(5000)
(global_min = train$min)
(global_max = train$max)
data_type = train$type


# plot the marginal distributions of the generated data
par(mfrow=c(1,3), mar=c(4, 4, 2, 1), pty="s")
plot(density(train$df_R$x1), main = "X1", xlab="")
plot(density(train$df_R$x2), main = "X2", xlab="")
barplot(table(train$df_R$x3)/sum(table(train$df_R$x3)), main = "X3", ylab = "Probability")

par(mfrow=c(1,3))
hist(train$df_R$x1, freq=FALSE, 100, main='X1', xlab='samples')
hist(train$df_R$x2, freq=FALSE, 100, main='X2', xlab='samples')
plot(train$df_R$x3, main='X3', xlab='samples')





#########################
# Model fitting
#########################


# Maximum number of coefficients (BS and Levels - 1 for the ordinal)
len_theta_max = len_theta
for (i in 1:nrow(MA)){ 
  if (train$type[i] == 'o'){
    len_theta_max = max(len_theta_max, nlevels(train$df_R[,i]) - 1)
  }
}


# create the TRAM-DAG model
param_model = create_param_model(MA, hidden_features_I=hidden_features_I, 
                                 len_theta=len_theta, 
                                 hidden_features_CS=hidden_features_CS,
                                 activation = 'sigmoid')
optimizer = optimizer_adam(learning_rate = 0.005)
param_model$compile(optimizer, loss=struct_dag_loss)
param_model$evaluate(x = train$df_orig, y=train$df_orig, batch_size = 7L)
summary(param_model)

#show activation_264 (Activation) 
# param_model$get_layer("activation_268")$get_config()

##### Training ####
num_epochs <- 400
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
      print(paste("Epoch", e))
      hist <- param_model$fit(x = train$df_orig, y = train$df_orig, 
                              epochs = 1L, verbose = TRUE, 
                              validation_data = list(test$df_orig,test$df_orig))
      
      # Append losses to history
      train_loss <- c(train_loss, hist$history$loss)
      val_loss <- c(val_loss, hist$history$val_loss)
      
      # Extract specific weights
      w <- param_model$get_layer(name = "beta")$get_weights()[[1]]
      
      ws <- rbind(ws, data.frame(w12 = w[1, 2], w13 = w[1, 3], w23 = w[2, 3]))
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




# Loss
epochs = length(train_loss)
plot(1:epochs, train_loss, type='l', main='', ylab='Loss', xlab='Epochs', ylim = c(1, 1.5))
lines(1:epochs, val_loss, type = 'l', col = 'blue')
legend('topright', legend=c('training', 'validation'), col=c('black', 'blue'), lty=1:1, cex=0.8, bty='n')


# Loss (last 100 epochs)
diff = max(epochs - 100,0)
plot(diff:epochs, val_loss[diff:epochs], type = 'l', col = 'green', main='Last 50 epochs')
lines(diff:epochs, train_loss[diff:epochs], type='l')



#########################
# Analysis of learned model
#########################


## plot parameters over epochs, with Colr for beta12 as comparison

fit12 <- Colr(x2 ~ x1, data = train$df_R)

plot(1:epochs, ws[,1], type='l', main= 'Coefficients for LS', ylab='Betas', xlab = 'Epochs')#, ylim=c(0, 6))
lines(1:epochs, ws[,2], type='l', col='blue')
abline(h=coef(fit12), col='black', lty=2)
legend('bottomright', legend=c('beta_12', 'beta_12 with Colr(x2~x1)', 'beta_23'), col=c('black', 'black', 'blue'), lty=c(1,2,1), cex=0.8, bty='n')

# learned coefs
ws[nrow(ws),]



############### Plot for Thesis (Loss and Parameters) ##########


file_name <- file.path(DIR, "loss_parameters.png")
png(filename = file_name, width = 2350, height = 1150, res = 300)

par(mfrow = c(1,2))

# loss
epochs = length(train_loss)
plot(1:epochs, train_loss, type='l', main='', ylab='Loss', xlab='Epochs', ylim = c(1, 1.5))
lines(1:epochs, val_loss, type = 'l', col = 'blue')
legend('topright', legend=c('Training', 'Validation'), col=c('black', 'blue'), lty=1:1, cex=0.8, bty='n')



# Plot with mathematical notation for beta coefficients
plot(1:epochs, ws[,1], type='l',  
     ylab=expression(beta), xlab = 'Epochs', lwd=1.5)
lines(1:epochs, ws[,2], type='l', col='blue', lwd=1.5)

abline(h=2, col='black', lty=4, lwd=1.2)  # dgp for beta_12
abline(h=0.2, col='blue', lty=4, lwd=1.2) # dgp for beta_13


# Legend with clear mappings
legend('bottomright',
       legend = c(expression(hat(beta)[12]),
                  expression(hat(beta)[13]),
                  expression("DGP" ~ beta[12] == 2),
                  expression("DGP" ~ beta[13] == 0.2)),
       col = c('black', 'blue', 'black', 'blue'),
       lty = c(1, 1, 4, 4),
       lwd = c(1.5, 1.5, 1.5, 1.5),
       cex = 0.7,
       bty = 'n')


# Close PNG device
dev.off()




# estimated parameters in the masked adjacency matrix
param_model$get_layer(name = "beta")$get_weights() * param_model$get_layer(name = "beta")$mask






#########################
# Sampling from Observational and Interventional distributions
#########################


##### Checking observational distribution ####

# create observational samples
s = do_dag_struct(param_model, train$A, doX=c(NA, NA, NA), num_samples = 5000)

par(mfrow=c(1,3))
for (i in 1:2){
  hist(train$df_orig$numpy()[,i], freq=FALSE, 100,main="", xlab=paste0("X",i)) #paste0("X",i,' (samples)')
  #hist(train$df_orig$numpy()[,i], freq=FALSE, 100,main=paste0("X_",i))
  lines(density(s[,i]$numpy()), col='red')
  legend('topleft', legend=c('DGP', 'TRAM-DAG'), col=c('black', 'red'), lty=1:1, cex=0.8, bty='n')
}
plot(table(train$df_R[,3])/sum(table(train$df_R[,3])), ylab='Probability ', 
     main="",
     xlab='X3')
table(train$df_R[,3])/sum(table(train$df_R[,3]))
points(as.numeric(table(s[,3]$numpy()))/5000, col='red', lty=2)
legend('topright', legend=c('DGP', 'TRAM-DAG'), col=c('black', 'red'), lty=1:1, cex=0.8, bty='n')
# table(s[,3]$numpy())/5000
par(mfrow=c(1,1))





############### Plot for Thesis (Observational Distribution) ##########


file_name <- file.path(DIR, "exp1_observational_distribution.png")
png(filename = file_name, width = 2350, height = 800, res = 300)


# Set plotting layout and margins
par(mfrow = c(1, 3), mar = c(4, 4, 2, 1))  # Smaller top margin

# Plot for X1 and X2
for (i in 1:2) {
  hist(train$df_orig$numpy()[, i],
       freq = FALSE,
       breaks = 50,
       main = "",
       xlab = paste0("X", i),
       col = "grey90",
       border = "black",
       ylim = c(0, max(density(train$df_orig$numpy()[, i])$y) * 1.2))
  
  lines(density(s[, i]$numpy()), col = "red", lwd = 2)
  box()  # Ensures frame is visible
  legend("topleft",
         legend = c("DGP", "TRAM-DAG"),
         col = c("black", "red"),
         lty = 1,
         lwd = c(1, 2),
         bty = "n",
         cex = 0.8)
}
# Categorical variable X3
true_probs <- table(train$df_R[, 3]) / sum(table(train$df_R[, 3]))
est_probs <- table(factor(s[, 3]$numpy(), levels = names(true_probs))) / sum(table(s[, 3]$numpy()))

# Softer red
light_red <- adjustcolor("red", alpha.f = 0.6)

barplot(rbind(true_probs, est_probs),
        beside = TRUE,
        col = c("grey90", light_red),
        names.arg = names(true_probs),
        ylim = c(0, max(true_probs, est_probs) * 1.2),
        ylab = "Probability",
        xlab = "X3",
        legend.text = c("DGP", "TRAM-DAG"),
        args.legend = list(x = "topright", bty = "n", cex = 0.8),
        border = "black")
box()

dev.off()

# Reset layout
par(mfrow = c(1, 1))









######### Simulation of do-interventions #####


# create interventional samples X1 = 1
s_dx2_1 = do_dag_struct(param_model, train$A, doX=c(NA, 1, NA), num_samples = 5000)
dgp_dx2_1 = dgp(10000, doX=c(NA, 1, NA))




file_name <- file.path(DIR, "exp1_interventional_distribution.png")
png(filename = file_name, width = 2350, height = 800, res = 300)

# Set plotting layout and margins
par(mfrow = c(1, 3), mar = c(4, 4, 2, 1))  # Smaller top margin


# Plot for X1
i = 1
hist(dgp_dx2_1$df_orig$numpy()[, i],
     freq = FALSE,
     breaks = 50,
     main = "",
     xlab = paste0("X", i),
     col = "grey90",
     border = "black",
     ylim = c(0, max(density(dgp_dx2_1$df_orig$numpy()[, i])$y) * 1.2))

lines(density(s_dx2_1[, i]$numpy()), col = "red", lwd = 2)
box()  # Ensures frame is visible
legend("topleft",
       legend = c("DGP", "TRAM-DAG"),
       col = c("black", "red"),
       lty = 1,
       lwd = c(1, 2),
       bty = "n",
       cex = 0.8)




# plot for X2 (both TRAM-DAG and DGP has only samples at value 1) (manually just for nice plotting)

# Create a matrix with two rows: DGP and TRAM-DAG
bar_heights <- rbind(
  DGP = c(0, 0, 0, 1, 0),
  TRAM_DAG = c(0, 0, 0, 1, 0)
)

# Set x-axis labels
colnames(bar_heights) <- c("-2.0", "-1.0", "0.0", "1.0", "2.0")

# Softer red
light_red <- adjustcolor("red", alpha.f = 0.6)

# Create the barplot
bar_centers <- barplot(bar_heights,
                       beside = TRUE,
                       space = c(0.2, 3), # Increased space: first value affects space within groups, second between groups
                       col = c("grey90", light_red),
                       border = "black",
                       ylim = c(0, 1.15),
                       ylab = "Probability",
                       xlab = "X2",
                       legend.text = c("DGP", "TRAM-DAG"),
                       args.legend = list(x = "topleft", bty = "n", cex = 0.8),
                       xaxt = "n") # Suppress default x-axis


# Add tickmarks and labels to the bottom (x-axis)
# Use 'at' to specify positions and 'labels' to specify the text
axis(side = 1, at = colMeans(bar_centers), labels = colnames(bar_heights))

# Add tickmarks to the left (y-axis)
axis(side = 2)

box() # Adds a box around the plot area





# Categorical variable X3
true_probs <- table(dgp_dx2_1$df_R[, 3]) / sum(table(dgp_dx2_1$df_R[, 3]))
est_probs <- table(factor(s_dx2_1[, 3]$numpy(), levels = names(true_probs))) / sum(table(s_dx2_1[, 3]$numpy()))

# Softer red
light_red <- adjustcolor("red", alpha.f = 0.6)

barplot(rbind(true_probs, est_probs),
        beside = TRUE,
        col = c("grey90", light_red),
        names.arg = names(true_probs),
        ylim = c(0, max(true_probs, est_probs) * 1.2),
        ylab = "Probability",
        xlab = "X3",
        legend.text = c("DGP", "TRAM-DAG"),
        args.legend = list(x = "topright", bty = "n", cex = 0.8),
        border = "black")
box()


dev.off()







### other interventions

doX=c(0.2, NA, NA)
dx0.2 = dgp(10000, doX=doX)
dx0.2$df_orig$numpy()[1:5,]


doX=c(0.7, NA, NA)
dx7 = dgp(10000, doX=doX)
#hist(dx0.2$df_orig$numpy()[,2], freq=FALSE,100)
mean(dx7$df_orig$numpy()[,2]) - mean(dx0.2$df_orig$numpy()[,2])  
mean(dx7$df_orig$numpy()[,3]) - mean(dx0.2$df_orig$numpy()[,3])  

s_dag = do_dag_struct(param_model, train$A, doX=c(0.2, NA, NA))
hist(dx0.2$df_orig$numpy()[,2], freq=FALSE, 50, main='X2 | Do(X1=0.2)', xlab='samples', 
     sub='Histogram from DGP with do. red:TRAM_DAG')
sample_dag_0.2 = s_dag[,2]$numpy()
lines(density(sample_dag_0.2), col='red', lw=2)
m_x2_do_x10.2 = median(sample_dag_0.2)

i = 3 
d = dx0.2$df_orig$numpy()[,i]
plot(table(d)/length(d), ylab='Probability ', 
     main='X3 | do(X1=0.2)',
     xlab='X3', ylim=c(0,0.6),  sub='Black DGP with do. red:TRAM_DAG')
points(as.numeric(table(s_dag[,3]$numpy()))/nrow(s_dag), col='red', lty=2)




###### Figure for paper ######
if (TRUE){
  doX=c(NA, NA, NA)
  s_obs_fitted = do_dag_struct(param_model, train$A, doX, num_samples = 5000)$numpy()
  # dx1 = 1.5
  # # dx1 = 0
  # doX=c(dx1, NA, NA)
  dx2 = 1
  # dx1 = 0
  doX=c(NA, dx2, NA)
  s_do_fitted = do_dag_struct(param_model, train$A, doX=doX, num_samples = 5000)$numpy()
  
  df = data.frame(vals=s_obs_fitted[,1], type='Model', X=1, L='L0')
  df = rbind(df, data.frame(vals=s_obs_fitted[,2], type='Model', X=2, L='L0'))
  df = rbind(df, data.frame(vals=s_obs_fitted[,3], type='Model', X=3, L='L0'))
  
  df = rbind(df, data.frame(vals=train$df_R[,1], type='DGP', X=1, L='L0'))
  df = rbind(df, data.frame(vals=train$df_R[,2], type='DGP', X=2, L='L0'))
  df = rbind(df, data.frame(vals=as.numeric(train$df_R[,3]), type='DGP', X=3, L='L0'))
  
  df = rbind(df, data.frame(vals=s_do_fitted[,1], type='Model', X=1, L='L1'))
  df = rbind(df, data.frame(vals=s_do_fitted[,2], type='Model', X=2, L='L1'))
  df = rbind(df, data.frame(vals=s_do_fitted[,3], type='Model', X=3, L='L1'))
  
  d = dgp(10000, doX=doX)$df_R
  df = rbind(df, data.frame(vals=d[,1], type='DGP', X=1, L='L1'))
  df = rbind(df, data.frame(vals=d[,2], type='DGP', X=2, L='L1'))
  df = rbind(df, data.frame(vals=as.numeric(d[,3]), type='DGP', X=3, L='L1'))
  
  p = ggplot() +
    # For X = 1 and X = 2, use position = "identity" (no dodging)
    geom_histogram(data = subset(df, X != 3), 
                   aes(x=vals, col=type, fill=type, y=..density..), 
                   position = "identity", alpha=0.4) +
    # For X = 3, use a bar plot for discrete data
    geom_bar(data = subset(df, X == 3), 
             aes(x=vals, y=..prop.. * 4,  col=type, fill=type), 
             position = "dodge", alpha=0.4, size = 0.5)+
    #limit between 0,1 but not removing the data
    coord_cartesian(ylim = c(0, 4)) +
    facet_grid(L ~ X, scales = 'free',
               # labeller = as_labeller(c('1' = 'X1', '2' = 'X2', '3' = 'X3', 'L1' = paste0('Do X1=',dx1), 'L0' = 'Obs')))+ 
               labeller = as_labeller(c('1' = 'X1', '2' = 'X2', '3' = 'X3', 'L1' = paste0('Do X2=',dx2), 'L0' = 'Obs')))+ 
    labs(y = "Density / (Probability × 4)", x='')  + # Update y-axis label
    theme_minimal() +
    theme(
      legend.title = element_blank(),   # Removes the legend title
      legend.position = c(0.9, 0.35),  # Adjust this to position the legend inside the plot (lower-right)
      legend.background = element_rect(fill="white", colour="white")  # Optional: white background with border
    )
  p
  # file_name <- paste0(fn, "_L0_L1.pdf")
  # ggsave(file_name, plot=p, width = 8, height = 6)
  # if (FALSE){
  #   file_path <- file.path("~/Library/CloudStorage/Dropbox/Apps/Overleaf/tramdag/figures", basename(file_name))
  #   print(file_path)
  #   ggsave(file_path, plot=p, width = 8/2, height = 6/2)
  # }
  
}



## Observational samples P(Y | X2 < -1):

# tram dag estimate
obs_filtered <- s_obs_fitted[s_obs_fitted[,2] < -1,]
table(obs_filtered[,3]) / nrow(obs_filtered)

# dgp sampling estimate
doX=c(NA, NA, NA)
s_obs_dgp = dgp(10000, doX=doX)
obs_dgp_filtered = s_obs_dgp$df_R[s_obs_dgp$df_R[,2] < -1,]
table(obs_dgp_filtered[,3]) / nrow(obs_dgp_filtered)




## Interventional samples for P(Y | do(x2 = 1)):

#tram dag estimate
table(s_do_fitted[,3])/5000

# dgp sampling estimate
doX=c(NA, 1, NA)
s_do_dgp = dgp(10000, doX=doX)
table(s_do_dgp$df_R[,3]) / nrow(s_do_dgp$df_R)





#plot of observational and interventional

png("C:/Users/kraeh/OneDrive/Dokumente/Desktop/UZH_Biostatistik/Masterarbeit/MA_Mike/presentation_report/intermediate_presentation/img/Prob_Estimates.png",
    res = 150, width = 800, height= 400)

par(mfrow=c(1,2), mar=c(4, 4, 2, 1)) # Removed pty='s'

# Observational plot the tram-dag estimates vs dgp estimates
plot(table(obs_dgp_filtered[,3])/nrow(obs_dgp_filtered), ylab='Probability',
     main=expression(paste("P(", X[3], " | ", X[2], " < -1)")),
     xlab=expression(X[3]), ylim=c(0,0.6), cex.main=0.85)
points(as.numeric(table(obs_filtered[,3])) / nrow(obs_filtered), col='red', lty=2)
legend('topright', legend=c('DGP', 'TRAM-DAG'), col=c('black', 'red'), lty=1:1, cex=0.65, bty='n')


# Interventional plot the tram-dag estimates vs dgp estimates
plot(table(s_do_dgp$df_R[,3]) / nrow(s_do_dgp$df_R), ylab='Probability',
     main=expression(paste("P(", X[3], " | do(", X[2], "=1))")),
     xlab=expression(X[3]), ylim=c(0,0.6), cex.main=0.85)
points(as.numeric(table(s_do_fitted[,3])) / nrow(s_do_fitted), col='red', lty=2)
legend('topright', legend=c('DGP', 'TRAM-DAG'), col=c('black', 'red'), lty=1:1, cex=0.65, bty='n')

dev.off()






#########################
# Analysis of learned Shifts
#########################




###### Comparison of estimated f(x2) vs TRUE f(x2) #######
shift_12 = shift_13 = cs_23 = xs = seq(-1,1,length.out=41)
idx0 = which(xs == 0) #Index of 0 xs needs to be odd
for (i in 1:length(xs)){
  #i = 1
  x = xs[i]
  # Varying x1
  X = tf$constant(c(x, 0.5, 3), shape=c(1L,3L)) 
  shift_12[i] =   param_model(X)[1,2,2]$numpy() # LS Term X1->X2
  shift_13[i] = param_model(X)[1,3,2]$numpy()   # LS Term X1->X3
  
  #Varying x2
  X = tf$constant(c(0.5, x, 3), shape=c(1L,3L)) 
  cs_23[i] = param_model(X)[1,3,1]$numpy()      # CS Term X2->X3
}




file_name <- file.path(DIR, "exp1_shifts.png")
png(filename = file_name, width = 2350, height = 800, res = 300)


# plot the following 3 plots in one row and make the plots squared
par(mfrow=c(1,3), mar=c(4, 4, 2, 1))

plot(xs, shift_12, 
     main="", 
     # sub = 'Effect of x1 on x2',
     ylab="Effect on X2",
     xlab="X1",
     col='red')
mtext(expression("LS effect of X1 on X2"), side = 3, line = 0.5, cex = 0.8)
abline(0, 2)
legend('topleft', legend=c('DGP', 'TRAM-DAG'), col=c('black', 'red'), lty=1:1, cex=0.8, bty='n')

delta_0 = shift_13[idx0] - 0
plot(xs, shift_13 - delta_0, main="",
     # sub = paste0('Effect of x1 on x3, delta_0 ', round(delta_0,2)),
     ylab="Effect on X3",
     xlab="X1",
     col='red')
mtext(expression("LS effect of X1 on X3"), side = 3, line = 0.5, cex = 0.8)
abline(0, .2)
legend('topleft', legend=c('DGP', 'TRAM-DAG'), col=c('black', 'red'), lty=1:1, cex=0.8, bty='n')


plot(xs, cs_23 + ( -cs_23[idx0] + f(0) ),
     main = "",
     ylab="Effect on X3",
     xlab="X2",
     col='red')
# sub = 'Effect of x2 on x3',col='red')
mtext(expression("CS effect of X2 on X3"), side = 3, line = 0.5, cex = 0.8)
lines(xs, f(xs))
legend('topleft', legend=c('DGP', 'TRAM-DAG'), col=c('black', 'red'), lty=1:1, cex=0.8, bty='n')

dev.off()



# for other DGPs
if (F32 == 1){ #Linear DGP
  if (MA[2,3] == 'ls'){
    delta_0 = shift_23[idx0] - f(0)
    plot(xs, shift_23 - delta_0, main='LS-Term (black DGP, red Ours)', 
         sub = paste0('Effect of x2 on x3, delta_0 ', round(delta_0,2)),
         xlab='x2', col='red')
    #abline(shift_23[length(shift_23)/2], -0.3)
    abline(0, -0.3)
  } 
  if (MA[2,3] == 'cs'){
    plot(xs, cs_23, main='CS-Term (black DGP, red Ours)', xlab='x2',  
         sub = 'Effect of x2 on x3',col='red')
    
    abline(cs_23[idx0], -0.3)  
  }
} else{ #Non-Linear DGP
  if (MA[2,3] == 'ls'){
    delta_0 = shift_23[idx0] - f(0)
    plot(xs, shift_23 - delta_0, main='LS-Term (black DGP, red Ours)', 
         sub = paste0('Effect of x2 on x3, delta_0 ', round(delta_0,2)),
         xlab='x2', col='red')
    lines(xs, f(xs))
  } else if (MA[2,3] == 'cs'){
    plot(xs, cs_23 + ( -cs_23[idx0] + f(0) ),
         ylab='CS',
         main='CS-Term (black DGP f2(x), red Ours)', xlab='x2',  
         sub = 'Effect of x2 on x3',col='red')
    lines(xs, f(xs))
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



### plot and save the Complex Shift

layer_sizes_CS <- c(ncol(MA), hidden_features_CS, nrow(MA))
masks_CS = create_masks(adjacency =  t(MA == 'cs'), hidden_features_CS)
# h_CS = create_param_net(len_param = 1, input_layer=input_layer, layer_sizes = layer_sizes_CS, masks_CS, last_layer_bias=FALSE)
dag_maf_plot_new(masks_CS, layer_sizes_CS)

ggsave("network_plot.pdf", dag_maf_plot_new(masks_CS, layer_sizes_CS),
       width = 4.5, height = 4)



######### Intercepts (baseline h)


#### Checking the transformation ####
h_params = param_model(train$df_orig)
r = check_baselinetrafo(h_params)
Xs = r$Xs
h_I = r$h_I

par(mfrow=c(1,3))

##### X1
df = data.frame(train$df_orig$numpy())
fit.11 = Colr(X1~1,df, order=len_theta)
temp = model.frame(fit.11)[1:2, -1, drop=FALSE] #WTF!
plot(fit.11, which = 'baseline only', newdata = temp, lwd=1, col='black', 
     main='h_I(X_1)', cex.main=0.8)
lines(Xs[,1], h_I[,1], col='red', lty=3, lwd=4)
rug(train$df_orig$numpy()[,1], col='black')
legend('topleft', legend=c('Colr()', 'TRAM-DAG'), col=c('black', 'red'), lty=c(1,3), lwd=c(1,3), cex=0.8, bty='n')


##### X2
df = data.frame(train$df_orig$numpy())
fit.21 = Colr(X2~X1,df, order=len_theta)
temp = model.frame(fit.21)[1:2,-1, drop=FALSE] #WTF!
plot(fit.21, which = 'baseline only', newdata = temp, lwd=2, col='black', 
     main='h_I(X_2)', cex.main=0.8)
lines(Xs[,2], h_I[,2], col='red', lty=3, lwd=4)
rug(train$df_orig$numpy()[,2], col='black')
legend('topleft', legend=c('Colr()', 'TRAM-DAG'), col=c('black', 'red'), lty=c(1,3), lwd=c(1,3), cex=0.8, bty='n')


##### X3


theta_tilde <- h_params[,,3:dim(h_params)[3], drop = FALSE]
theta = to_theta3(theta_tilde)
# theta[1,3,1:3]



category <- seq(1,3, by = 1)
# make a plot with only the categories on x axis and not continuous
plot(category, train$theta_k + f(0), type='p', col='blue', lwd=2,
     main='Theta for X3', cex.main=0.8, ylim = c(-2.2, 2.5), xlim=c(0.9, 3.3), xlab = 'X3', ylab = "trafo",)
points(category, theta[1,3,1:3] + cs_23[idx0], col='red', lty=2, lwd=5)





file_name <- file.path(DIR, "exp1_intercepts.png")
png(filename = file_name, width = 2350, height = 800, res = 300)


# plot the following 3 plots in one row and make the plots squared
par(mfrow=c(1,3), mar=c(4, 4, 2, 1))

##### X1
df <- data.frame(train$df_orig$numpy())
fit.11 <- Colr(X1 ~ 1, df, order = len_theta)
temp <- model.frame(fit.11)[1:2, -1, drop = FALSE]

plot(fit.11, which = 'baseline only', newdata = temp, lwd = 1, col = 'black',
     main = "", cex.main = 0.9,
     xlab = "X1", ylab = expression(h["I"](X[1])))
lines(Xs[, 1], h_I[, 1], col = 'red', lty = 3, lwd = 4)
rug(train$df_orig$numpy()[, 1], col = 'black')
legend('topleft', legend = c('Colr()', 'TRAM-DAG'), col = c('black', 'red'),
       lty = c(1, 3), lwd = c(1, 3), cex = 0.8, bty = 'n')

##### X2
fit.21 <- Colr(X2 ~ X1, df, order = len_theta)
temp <- model.frame(fit.21)[1:2, -1, drop = FALSE]

plot(fit.21, which = 'baseline only', newdata = temp, lwd = 1, col = 'black',
     main = "", cex.main = 0.9,
     xlab = "X2", ylab = expression(h["I"](X[2])))
lines(Xs[, 2], h_I[, 2], col = 'red', lty = 3, lwd = 4)
rug(train$df_orig$numpy()[, 2], col = 'black')
legend('topleft', legend = c('Colr()', 'TRAM-DAG'), col = c('black', 'red'),
       lty = c(1, 3), lwd = c(1, 3), cex = 0.8, bty = 'n')

##### X3 (categorical)
theta_tilde <- h_params[, , 3:dim(h_params)[3], drop = FALSE]
theta <- to_theta3(theta_tilde)
category <- 1:3
x_offset <- c(-0.06, -0.06, -0.06)  # slight offset for visibility

plot(category, train$theta_k, type = 'p', col = 'black', pch = 16,
     main = "", cex.main = 0.9,
     xlab = "X3", ylab = expression(h["I"](X[3])),
     ylim = c(-2.2, 2.5), xlim = c(0.8, 3.2), xaxt = 'n')

points(category + x_offset, theta[1, 3, 1:3] + (cs_23[idx0]-f(0)), col = 'red', pch = 17, cex = 1.5)

axis(1, at = category, labels = category)
legend("topleft", legend = c("DGP", "TRAM-DAG"),
       col = c("black", "red"), pch = c(16, 17), bty = "n", cex = 0.8)


dev.off()





#########################
# Counterfactuals
#########################


# Counterfactuals for X2 when having different values for X1

# Test Individuum, observed values: X = c(0.5, -1.2, 2)




#### with DGP ####


# 1) determine "observed" latent value z2_dgp 

z2_dgp <- 5*(-1.2) + 2 * 0.5


# 2) determine counterfactuals for X2 when varying X1 in c(0, 0.9)

x1_values <- seq(0, 0.9, length.out=10)

# Create a matrix to store counterfactuals
counterfactuals <- matrix(NA, nrow=length(x1_values), ncol=2)
colnames(counterfactuals) <- c("X1", "X2")

counterfactuals[,1] <- x1_values

for (i in seq_along(x1_values)){
  x1 <- x1_values[i]
  # Calculate the counterfactual value for X2
  # Using the formula: z2_dgp = 5 * X2 + 2 * X1
  # Rearranging gives: X2 = (z2_dgp - 2 * X1) / 5
  counterfactuals[i,2] <- (z2_dgp - 2 * x1) / 5
}


plot(counterfactuals)





#### with TRAM-DAG ####



# 1) determine the latent value z_2 of observed X2=-1.2 | X1=0.5

X = tf$constant(c(0.5, -1.2, 2), shape=c(1L,3L)) 
t_i = X

h_params = param_model(t_i)
# param_model(X)[1,2,2]$numpy()


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

# latent value z_2 of observed
z_2 <- h_12[1,2]






# 2) for varying X1 in c(0, 0.9) compute counterfactuals for X2

X1_values = seq(0, 0.9, length.out=10)


#create a tensor of shape c(10L, 3L) with X1_values as float32
X_hat = tf$constant(cbind(X1_values, -1.2, 2), shape=c(length(X1_values),3L), 
                    dtype = tf$float32)

t_i = X_hat

# get outputs of model for X1_hat
h_params = param_model(t_i)

# compute h for the current X1_hat
k_min <- k_constant(global_min)
k_max <- k_constant(global_max)

# from the last dimension of h_params the first entry is h_cs1
# the second to |X|+1 are the LS
# the 2+|X|+1 to the end is H_I
h_cs <- h_params[,,1, drop = FALSE]
h_ls <- h_params[,,2, drop = FALSE]
#LS
h_LS = tf$squeeze(h_ls, axis=-1L)
#CS
h_CS = tf$squeeze(h_cs, axis=-1L)

theta_tilde <- h_params[,,3:dim(h_params)[3], drop = FALSE]
theta = to_theta3(theta_tilde)


# turn the observed latent z_2 to shape c(1, 1L)
latent_sample <- tf$expand_dims(z_2, axis=0L) #tf$expand_dims(z_2, axis=-1L)


# h_dag returns the intercept h (single value) at 0 and 1
h_0 =  h_LS + h_CS + h_dag(L_START, theta) #tf$expand_dims(h_LS + h_CS + h_dag(L_START, theta), axis=-1L)
h_1 =  h_LS + h_CS + h_dag(R_START, theta) #tf$expand_dims(h_LS + h_CS + h_dag(R_START, theta), axis=-1L)


# find the root of f(t_i) = h(t_i) - rlogis() == 0, those samples are the target samples
object_fkt = function(t_i){
  return(h_dag_extra_struc(t_i, theta, shift = h_LS + h_CS, k_min, k_max) - latent_sample)
}

target_sample = tfp$math$find_root_chandrupatla(object_fkt)$estimated_root

  

# Manuly calculating the inverse for the extrapolated samples
## smaller than h_0
l = latent_sample

# check if the latent sample would be below h_0 (needs extrapolation)
mask <- tf$math$less_equal(l, h_0)
slope0 <- h_dag_dash(L_START, theta)
target_sample = tf$where(mask,
                         ((l-h_0)/slope0)*(k_max - k_min) + k_min
                         ,target_sample)

## larger than h_1
mask <- tf$math$greater_equal(l, h_1)
slope1<- h_dag_dash(R_START, theta)
target_sample = tf$where(mask,
                         (((l-h_1)/slope1) + 1.0)*(k_max - k_min) + k_min,
                         target_sample)




# counterfactuals for X2|X1 with TRAM-DAG
counterfactuals_tram_dag <- target_sample$numpy()[,2]

plot(counterfactuals)
points(counterfactuals[,1], target_sample$numpy()[,2], col='red', pch=16)


# save figure for thesis



file_name <- file.path(DIR, "exp1_counterfactuals.png")
png(filename = file_name, width = 2050, height = 1300, res = 300)


# plot the following 3 plots in one row and make the plots squared
par(mfrow=c(1,1), mar=c(4, 4, 2, 1))


plot(counterfactuals[,1], counterfactuals[,2], type = 'p', col = 'black', pch = 16,
     main = "", cex.main = 0.9,
     xlab = "X1", ylab = "Counterfactuals for X2 given X1")
points(counterfactuals[,1], counterfactuals_tram_dag, col = adjustcolor("red", alpha.f = 0.8), pch = 17, cex = 1.5)

axis(1, at = category, labels = category)
legend("topright", legend = c("DGP", "TRAM-DAG"),
       col = c("black", adjustcolor("red", alpha.f = 0.8)), pch = c(16, 17), bty = "n", cex = 0.8)

#add text with individual observed X = (0.5, -1.2, 2)
text(0.05, -1.3, labels = "Observed:  X1 = 0.5,  X2 = -1.2,  X3 = 2", pos = 4, cex = 0.95, col = 'black')

dev.off()




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








