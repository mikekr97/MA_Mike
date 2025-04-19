

#================================================================================================
# Data preparation
#================================================================================================

library(ncdf4)
library(lubridate)
library(dplyr)


# Function to extract yearly averages (for certain months)
extract_yearly_avg <- function(nc_obj, months, var_name) {
  
  # Extract time values and units
  time_values <- ncvar_get(nc_obj, "time")
  time_units <- ncatt_get(nc_obj, "time", "units")$value
  
  # Extract reference date
  time_origin <- ymd_hms("1800-01-01 00:00:00", tz = "UTC")
  time_converted <- time_origin + hours(time_values)
  
  # Extract data variable (assumes variable name is "sic")
  variable_data <- ncvar_get(nc_obj, var_name)
  
  # Create dataframe
  df <- tibble(
    time = time_converted,
    value = variable_data,
    year = year(time_converted),
    month = month(time_converted)
  )
  
  # Filter by selected months
  df_filtered <- df %>% filter(month %in% months)
  
  # Compute yearly averages
  df_avg <- df_filtered %>% group_by(year) %>% summarise(value = mean(value, na.rm = TRUE))
  
  return(df_avg)
}



# load the nc objects
bk_sic <- nc_open("data/bk_sic.nc")
nh_spv <- nc_open("data/nh_spv_uwnd.nc")
ural_slp <- nc_open("data/ural_slp.nc")
np_slp <- nc_open("data/np_slp.nc")

df_avg_bk <- extract_yearly_avg(bk_sic, c(10, 11, 12), var_name = "sic")
df_avg_spv <- extract_yearly_avg(nh_spv, c(1, 2, 3), var_name = "uwnd")
df_avg_ural <- extract_yearly_avg(ural_slp, c(10, 11, 12), var_name = "slp")
df_avg_np <- extract_yearly_avg(np_slp, c(10, 11, 12), var_name = "slp")


# standardize (zero to mean, unit variance)
bk <- (df_avg_bk$value - mean(df_avg_bk$value))/sd(df_avg_bk$value)
ural <- (df_avg_ural$value - mean(df_avg_ural$value))/sd(df_avg_ural$value)
np <- (df_avg_np$value - mean(df_avg_np$value))/sd(df_avg_np$value)
spv <- (df_avg_spv$value - mean(df_avg_spv$value))/sd(df_avg_spv$value)


# detrend
bk_detrended <- residuals(lm(bk ~ df_avg_bk$year))
ural_detrended <- residuals(lm(ural ~ df_avg_ural$year))
np_detrended <- residuals(lm(np ~ df_avg_np$year))
spv_detrended <- residuals(lm(spv ~ df_avg_spv$year))


# histograms of the detrended variables

par(mfrow=c(4,1))
hist(bk_detrended, main = "BK", xlab = "BK")
hist(ural_detrended, main = "Ural", xlab = "Ural")
hist(np_detrended, main = "NP", xlab = "NP")
hist(spv_detrended, main = "SPV", xlab = "SPV")


# SPV: starting in Jan-Mar of 1951 until 2019
# X: starting in Oct-Dec of 1950 until 2018

raw_df <- data.frame(
  BK = bk_detrended[-length(bk_detrended)],
  Ural = ural_detrended[-length(ural_detrended)],
  NP = np_detrended[-length(np_detrended)],
  SPV = spv_detrended[-1]
)




#================================================================================================
# Observational distribution
#================================================================================================


# X1 ~ P(X1) no parents --> KDE
# X2 ~ P(X2) no parents --> KDE
# X3 ~ P(X3|X1, X2) --> Linear regression
# X4 ~ P(X4|X1, X2, X3) --> Linear regression


#================================================================================================
# ### KDE for X1 and X2
#================================================================================================


library(MASS)  # for kde2d
library(ggplot2)

# Assume x1_data and x2_data are observed data vectors
kde_x1 <- density(raw_df$NP, bw = "SJ")  
kde_x2 <- density(raw_df$Ural, bw = "SJ")

par(mfrow=c(1,2))
hist(raw_df$NP, main = "NP", xlab = "NP", prob=TRUE)
lines(kde_x1, col="red")
hist(raw_df$Ural, main = "Ural", xlab = "Ural", prob=TRUE)
lines(kde_x2, col="red")


# Sample from estimated KDEs
x1_samples <- sample(kde_x1$x, size=10000, replace=TRUE, prob=kde_x1$y)
x2_samples <- sample(kde_x2$x, size=10000, replace=TRUE, prob=kde_x2$y)


#================================================================================================
# ### Linear Regression for X3 and X4
#================================================================================================

# Fit linear regression models
fit_3 <- lm(BK ~ NP + Ural, data=raw_df)
fit_4 <- lm(SPV ~ NP + Ural + BK, data=raw_df)



### Linear Regression for X3 and X4

# X3 ~ P(X3|X1, X2)
# X3 ~ N(a + b1*X1 + b2*X2, sigma^2)
fit_3$coefficients

normal_pdf_3 <- function(x1, x2, p) {
  
  # we want conditional N(mu, sigma^2) where mu = a + b1*x1 + b2*x2
  # take a p (probability value) and return the value of x3 that corresponds to that probability
  
  mu <- fit_3$coefficients[1] + fit_3$coefficients[2]*x1 + fit_3$coefficients[3]*x2
  sigma <- summary(fit_3)$sigma
  
  return(qnorm(p, mean=mu, sd=sigma))
}


# X4 ~ P(X4|X1, X2, X3)
# X4 ~ N(a + b1*X1 + b2*X2 + b3*X3, sigma^2)
fit_4$coefficients

normal_pdf_4 <- function(x1, x2, x3, p) {
  
  # we want conditional N(mu, sigma^2) where mu = a + b1*x1 + b2*x2 + b3*x3
  # take a p (probability value) and return the value of x4 that corresponds to that probability
  
  mu <- fit_4$coefficients[1] + fit_4$coefficients[2]*x1 + fit_4$coefficients[3]*x2 + fit_4$coefficients[4]*x3
  sigma <- summary(fit_4)$sigma
  
  return(qnorm(p, mean=mu, sd=sigma))
}


#================================================================================================
# ### Sample X3 and X4 from the conditional distributions
#================================================================================================


# Sample from conditional distributions
x3_samples <- normal_pdf_3(x1_samples, x2_samples, runif(10000))
x4_samples <- normal_pdf_4(x1_samples, x2_samples, x3_samples, runif(10000))




#================================================================================================
# ### Simulate interventional distribution for X3 = -3.5
#================================================================================================

dx3 = -3.5
x3_samples_do <- rep(dx3, 10000)

x4_samples_do <- normal_pdf_4(x1_samples, x2_samples, x3_samples_do, runif(10000))


#================================================================================================
# ### Plot full observational distribution vs true data (vs KDE for X1 and X2) with ggplot
#================================================================================================

# Plot full observational distribution



true_df <- data.frame(
  X1 = raw_df$NP,
  X2 = raw_df$Ural,
  X3 = raw_df$BK,
  X4 = raw_df$SPV,
  type = "true", 
  L = "L0"
)


# observational distribution
doX=c(NA, NA, NA, NA)
s_obs_fitted <- data.frame(
  X1 = x1_samples,
  X2 = x2_samples,
  X3 = x3_samples,
  X4 = x4_samples, 
  type = "Model"
  
)

# do intervention
# dx3 = -3.5
# doX=c(NA, NA, dx3, NA)
s_do_fitted <- data.frame(
  X1 = x1_samples,
  X2 = x2_samples,
  X3 = x3_samples_do,
  X4 = x4_samples_do, 
  type = "Model"
  
)
# add the doX to the plot
df = data.frame(vals=s_obs_fitted[,1], type='Model', X=1, L='L0')
df = rbind(df, data.frame(vals=s_obs_fitted[,2], type='Model', X=2, L='L0'))
df = rbind(df, data.frame(vals=s_obs_fitted[,3], type='Model', X=3, L='L0'))
df = rbind(df, data.frame(vals=s_obs_fitted[,4], type='Model', X=4, L='L0'))

df = rbind(df, data.frame(vals=true_df[,1], type='DGP', X=1, L='L0'))
df = rbind(df, data.frame(vals=true_df[,2], type='DGP', X=2, L='L0'))
df = rbind(df, data.frame(vals=true_df[,3], type='DGP', X=3, L='L0'))
df = rbind(df, data.frame(vals=true_df[,4], type='DGP', X=4, L='L0'))

df = rbind(df, data.frame(vals=s_do_fitted[,1], type='Model', X=1, L='L1'))
df = rbind(df, data.frame(vals=s_do_fitted[,2], type='Model', X=2, L='L1'))
df = rbind(df, data.frame(vals=s_do_fitted[,3], type='Model', X=3, L='L1'))
df = rbind(df, data.frame(vals=s_do_fitted[,4], type='Model', X=4, L='L1'))

# d = dgp(10000, doX=doX)$df_R
# # d = dgp(nrow(train_df), doX=doX, data=train_df)$df_R  # not possible for real data
# df = rbind(df, data.frame(vals=d[,1], type='DGP', X=1, L='L1'))
# df = rbind(df, data.frame(vals=d[,2], type='DGP', X=2, L='L1'))
# df = rbind(df, data.frame(vals=as.numeric(d[,3]), type='DGP', X=3, L='L1'))
# df = rbind(df, data.frame(vals=d[,4], type='DGP', X=4, L='L1'))



