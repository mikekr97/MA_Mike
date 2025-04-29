

library(tram)

### DGP based on Fz = logistic

set.seed(123)
n <- 1000

beta <- 5
sex <- rbinom(n, size = 1, prob = 0.5)

#### CASE 1: DGP-Fz = logistic

# generate latent logistic sample
y_dash <- rlogis(n, location = 0, scale = 1)

# y_dash = h_0(y) + beta * x
# y_dash = 2.5*y + beta * x
y = (y_dash - beta * sex) / 2.5


df <- data.frame(Y = y, sex=sex)

par(mfrow=c(1,1))
boxplot(Y~as.factor(sex), data=df)



###############################

# different models

plot_ecdf <- function(model, modelname){
  
  # grid for Y
  y_grid <- seq(min(df$Y), max(df$Y), length.out = 300)
  
  # now predict: for each Y_grid value, need the covariates sex=0 or sex=1
  
  newdata0 <- data.frame(Y = y_grid, sex = 0)
  newdata1 <- data.frame(Y = y_grid, sex = 1)
  
  pred0 <- predict(model, newdata = newdata0, type = "distribution")
  pred1 <- predict(model, newdata = newdata1, type = "distribution")
  
  # ECDFs
  ecdf_sex0 <- ecdf(df$Y[df$sex == 0])
  ecdf_sex1 <- ecdf(df$Y[df$sex == 1])
  
  # Plot
  plot(y_grid, ecdf_sex0(y_grid), type = "l", lty = 2, col = "blue",
       xlab = "Y", ylab = "CDF", main = modelname, ylim = c(0,1))
  lines(y_grid, ecdf_sex1(y_grid), lty = 2, col = "red")
  
  lines(y_grid, pred0, lty = 1, col = "blue", lwd = 2)
  lines(y_grid, pred1, lty = 1, col = "red", lwd = 2)
  
  legend("topleft",
         legend = c("ECDF sex=0", "ECDF sex=1", "Model sex=0", "Model sex=1"),
         col = c("blue", "red", "blue", "red"),
         lty = c(2,2,1,1),
         lwd = c(1,1,2,2), 
         cex = 0.8,
         bty = "n")
  
}


par(mfrow=c(1,1))
# fit logistic model

fit_Colr <- Colr(Y~sex, data=df)
plot_ecdf(model = fit_Colr, modelname = "Colr" )


# now fit model with Fz = standardnormal

fit_BoxCox <- BoxCox(Y~sex, data=df)
plot_ecdf(model = fit_BoxCox, modelname = "BoxCox" )

# how fit model with hazard ratio
fit_cox <- Coxph(Y~sex, data=df)
plot_ecdf(model = fit_cox, modelname = "Cox" )

# now fit model with Fz = lehmann -> reverse time hazard
fit_lehmann <- Lehmann(Y~sex, data=df)
# plot_ecdf(model = fit_lehmann, modelname = "Lehmann" )  # not sure if flexible h()

# plot the 3 next to eachother
par(mfrow=c(1,4))
plot_ecdf(model = fit_Colr, modelname = "Colr" )
plot_ecdf(model = fit_BoxCox, modelname = "BoxCox" )
plot_ecdf(model = fit_cox, modelname = "Coxph" )
# plot_ecdf(model = fit_lehmann, modelname = "Lehmann" )
