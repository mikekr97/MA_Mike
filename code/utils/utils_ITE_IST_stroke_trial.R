
# helpers for file TRAM_DAG_ITE_Stroke_IST.R

## ---- data preparation ----------

# by Chen et al. (2025)
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

# by Chen et al. (2025)
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



## ---- data preparation(for TRAM-DAG) ----------


# Helper function to dummy encode and scale one dataset
transform_dataset <- function(df) {
  mm <- model.matrix(~ ., data = df)[, -1]  # dummy encode, drop intercept
  mm <- as.data.frame(mm)
  
  # Shift outcome encoding from {0,1} to {1,2}, if needed
  if ("OUTCOME6M" %in% colnames(mm)) {
    mm$OUTCOME6M <- mm$OUTCOME6M + 1
  }
  
  # Scale numerical variables
  num_vars <- c("AGE", "RDELAY", "RSBP")
  for (v in num_vars) {
    if (v %in% colnames(mm)) {
      mm[[v]] <- scale(mm[[v]], center = TRUE, scale = TRUE)
    }
  }
  
  if ("RXASPY" %in% colnames(mm)) { # if RXASPY is in the dataset, rename it to RXASP
    colnames(mm)[colnames(mm) == "RXASPY"] <- "RXASP"
  }
  
  return(mm)
}





## ---- models (glm and comets RF) ----------

# by Chen et al. (2025), but added prediction for outcome Y=1
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
  
  # Predict outcome for observed T and X on derivation sample
  data$data.dev$Y_pred <- predict(fit.dev.tx, newdata = pred.data.dev, type = "response") * 
    ifelse(data$data.dev$RXASP == "Y", 1, 0) + 
    predict(fit.dev.ct, newdata = pred.data.dev, type = "response") * 
    (1 - ifelse(data$data.dev$RXASP == "Y", 1, 0))
  
  
  
  
  # Predict ITE on validation sample
  pred.data.val <- data$data.val %>% dplyr::select(-c("RXASP","OUTCOME6M"))
  y.val.tx <- predict(fit.dev.tx, newdata = pred.data.val, type = "response")
  y.val.ct <- predict(fit.dev.ct, newdata = pred.data.val, type = "response")
  pred.val <- y.val.tx - y.val.ct
  

  # Predict outcome for observed T and X on validation sample
  data$data.val$Y_pred <- predict(fit.dev.tx, newdata = pred.data.val, type = "response") * 
    ifelse(data$data.val$RXASP == "Y", 1, 0) + 
    predict(fit.dev.ct, newdata = pred.data.val, type = "response") * 
    (1 - ifelse(data$data.val$RXASP == "Y", 1, 0))
  
  
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



# by Krähenbühl

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





# TRAM-DAG (param_model) needs to be trained first
tram.ITE <- function(data = test.compl.data.transformed, train = dat.train.tf, test = dat.test.tf){
  
  # debugging 
  # data = test.compl.data.transformed
  # train = dat.train.tf
  # test = dat.test.tf
  
  # Train set
  
  # Prepare Data
  # set the values of the first column of train to 0
  train_ct <- tf$concat(list(tf$zeros_like(train[, 1, drop = FALSE]), train[, 2:tf$shape(train)[2]]), axis = 1L)
  # set the values of the first column of train to 1
  train_tx <- tf$concat(list(tf$ones_like(train[, 1, drop = FALSE]), train[, 2:tf$shape(train)[2]]), axis = 1L)
  
  # Calculate potential outcomes
  h_params_ct <- param_model(train_ct)
  y_train_ct <- as.numeric(do_probability_IST(h_params_ct))
  
  h_params_tx <- param_model(train_tx)
  y_train_tx <- as.numeric(do_probability_IST(h_params_tx))
  
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
  y_test_ct <- as.numeric(do_probability_IST(h_params_ct))
  # y_test_ct <- predict(fit_gam, newdata = data.frame(pred_probability = y_test_ct), type = "response") # if recalibrate
  
  h_params_tx <- param_model(test_tx)
  y_test_tx <- as.numeric(do_probability_IST(h_params_tx))
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




# ---- evaluation helpers ----------


# by Chen et al. (2025)
calc.ATE.RR <- function(data){
  data <- as.data.frame(data)
  model <- glm(OUTCOME6M ~ RXASP, data = data, family = quasipoisson(link = "log"))
  ATE.odds <- exp(coef(model)[2])
  ATE.lb <- exp(suppressMessages(confint(model)[2,1]))
  ATE.ub <- exp(suppressMessages(confint(model)[2,2]))
  return(data.frame(ATE.odds = ATE.odds, ATE.lb = ATE.lb, ATE.ub = ATE.ub, 
                    n.total = nrow(data), 
                    n.tr = sum(data$RXASP == "Y"), n.ct = sum(data$RXASP == "N")))
}



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



# by M. Krähenbühl
calc.ATE.RiskDiff <- function(data) {
  data <- as.data.frame(data)
  # data$OUTCOME6M <- data$OUTCOME6M - 1  # transform to 0/1
  
  # Ensure RXASP is a factor
  # data$RXASP <- factor(data$RXASP, levels = c("N", "Y"))
  
  # Calculate proportions
  p1 <- mean(data$OUTCOME6M[data$RXASP ==  "Y"])  # treated
  p0 <- mean(data$OUTCOME6M[data$RXASP == "N"])  # control
  
  # Risk difference
  ATE.RiskDiff <- p1 - p0
  
  # Standard error for RD (Wald)
  n1 <- sum(data$RXASP == "Y")
  n0 <- sum(data$RXASP == "N")
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


# same for transformed dataset
calc.ATE.RiskDiff_transformed <- function(data) {
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



# bootstrap CI for the ATE in terms of mean(ITE_pred)

ITE_CI_bootstrap <- function(ITE, n_boot){
  
  B <- n_boot
  
  bootstrap_means <- numeric(B)
  # Bootstrap resampling: compute mean for each resample
  set.seed(123)  # for reproducibility
  for (i in 1:B) {
    # Resample with replacement
    resample <- sample(ITE, length(ITE), replace = TRUE)
    
    # Store the mean of the resample
    bootstrap_means[i] <- mean(resample)
  }
  
  ci_lower <- quantile(bootstrap_means, 0.025)
  ci_upper <- quantile(bootstrap_means, 0.975)
  return(list(
    mean = mean(ITE),
    ci_lower = ci_lower,
    ci_upper = ci_upper
  ))
}



# ---- evaluation plots ----------


# by Chen et al. (2025)
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



plot_outcome_ITE_tuned_rf <- function(data.dev.rs, data.val.rs , x_lim = c(-0.5,0.5)){
  
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


## risk diff base R - by Krähenbühl

plot_ATE_vs_ITE_base_RiskDiff <- function(model.results = model.results, breaks, 
                                          delta_horizontal = 0.02, 
                                          ylim_delta = 0.1,
                                          xlim_delta = 0.05) {
  
  
  
  data.dev.grouped.ATE <- model.results$data.dev.rs %>% 
    mutate(ITE.Group = cut(ITE, breaks = breaks, include.lowest = T)) %>%
    dplyr::filter(!is.na(ITE.Group)) %>%
    group_by(ITE.Group) %>% 
    group_modify(~ calc.ATE.RiskDiff(.x)) %>% ungroup()
  data.val.grouped.ATE <- model.results$data.val.rs %>% 
    mutate(ITE.Group = cut(ITE, breaks = breaks, include.lowest = T)) %>%
    dplyr::filter(!is.na(ITE.Group)) %>%
    group_by(ITE.Group) %>%
    group_modify(~ calc.ATE.RiskDiff(.x)) %>% ungroup() 
  
  dev.data <- data.dev.grouped.ATE
  val.data <- data.val.grouped.ATE
  
  bin_centers <- (head(breaks, -1) + tail(breaks, -1)) / 2
  group_labels <- levels(dev.data$ITE.Group)
  
  delta <- delta_horizontal     # Horizontal offset between training and test
  cap_width <- 0.02 # Width of CI caps
  
  # Set up empty plot
  plot(NULL,
       xlim = range(bin_centers) + c(-xlim_delta, xlim_delta),
       ylim = range(c(dev.data$ATE.lb, dev.data$ATE.ub)) + c(-ylim_delta, ylim_delta),
       xlab = "ITE Group", ylab = "ATE in Risk Difference",
       xaxt = "n")
  mtext(expression("Observed ATE per ITE-subgroup: " * pi[treated] - pi[control]),
        side = 3, line = 0.5, cex = 0.95)
  
  
  ### Training CI bars + points
  x_train <- bin_centers - delta
  for (i in seq_along(bin_centers)) {
    arrows(x_train[i], dev.data$ATE.lb[i], x_train[i], dev.data$ATE.ub[i],
           angle = 90, code = 3, length = cap_width, col = "orange", lwd = 2)
    points(x_train[i], dev.data$ATE.RiskDiff[i], pch = 16, col = "orange")
  }
  
  ### Test CI bars + points
  x_test <- bin_centers + delta
  for (i in seq_along(bin_centers)) {
    arrows(x_test[i], val.data$ATE.lb[i], x_test[i], val.data$ATE.ub[i],
           angle = 90, code = 3, length = cap_width, col = "#36648B", lwd = 2)
    points(x_test[i], val.data$ATE.RiskDiff[i], pch = 16, col = "#36648B")
  }
  
  # Connect points with lines
  lines(x_train, dev.data$ATE.RiskDiff, col = "orange", lwd = 2)
  lines(x_test, val.data$ATE.RiskDiff, col = "#36648B", lwd = 2)
  
  # Reference lines
  abline(h = 0, lty = "dotted", col = "gray")
  lines(bin_centers, bin_centers, lty = 3)  # Theoretical
  
  # Rug plots (stripchart)
  stripchart(model.results$data.dev.rs$ITE, method = "jitter", at = par("usr")[4],
             add = TRUE, pch = "|", col = "orange", jitter = 0.002, cex = 0.6)
  stripchart(model.results$data.val.rs$ITE, method = "jitter", at = par("usr")[3],
             add = TRUE, pch = "|", col = "#36648B", jitter = 0.002, cex = 0.6)
  
  # Custom x-axis
  # axis(1, at = bin_centers, labels = group_labels)
  
  # Custom x-axis
  axis(1, at = bin_centers, labels = FALSE)  # suppress default labels
  
  # Add staggered labels manually
  for (i in seq_along(bin_centers)) {
    offset <- ifelse(i %% 2 == 0, -1.5, -3)  # stagger in two rows
    text(x = bin_centers[i], y = par("usr")[3] + offset * strheight("M"),
         labels = group_labels[i], srt = 0, xpd = TRUE)
  }
  
  
  # Legend
  legend("topleft", inset = c(0.02, 0.02),  # move slightly downward
         legend = c("Training", "Test", "Theoretical ATE"),
         col = c("orange", "#36648B", "black"), 
         pch = c(16, 16, NA), lty = c(1, 1, 3),
         bty = "n", cex =0.8, lwd = c(2, 2, 1))
}




# same with ggplot

plot_ATE_ITE_in_group_RiskDiff_ggplot <- function(dev.data = data.dev.rs, val.data = data.val.rs, ylb=0, yub=2){
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







plot_calibration <- function(y_obs, y_pred, bins = 10, conf = 0.95, title_suffix = "Set") {
  library(dplyr)
  library(binom)
  library(ggplot2)
  
  df <- data.frame(
    Y_prob = y_pred,
    Y = y_obs
  )
  
  # Bin the predicted probabilities
  df <- df %>%
    mutate(prob_bin = cut(
      Y_prob,
      breaks = seq(min(Y_prob), max(Y_prob), length.out = bins + 1),
      include.lowest = TRUE
    ))
  
  # Aggregate data within bins
  agg_bin <- df %>%
    group_by(prob_bin) %>%
    summarise(
      pred_probability = mean(Y_prob),
      obs_proportion = mean(Y),
      n_pos = sum(Y == 1),
      n_total = n(),
      .groups = "drop"
    )
  
  # Compute confidence intervals using Wilson method
  bin_cis <- mapply(
    function(x, n) binom.confint(x, n, conf.level = conf, methods = "wilson")[, c("lower", "upper")],
    agg_bin$n_pos, agg_bin$n_total, SIMPLIFY = FALSE
  )
  cis_df <- do.call(rbind, bin_cis)
  agg_bin$lo_CI_obs_prop <- cis_df[, 1]
  agg_bin$up_CI_obs_prop <- cis_df[, 2]
  agg_bin$width_CI <- abs(agg_bin$up_CI_obs_prop - agg_bin$lo_CI_obs_prop)
  
  # Create calibration plot
  ggplot(agg_bin, aes(x = pred_probability, y = obs_proportion)) +
    geom_point(color = "blue", size = 2) +
    geom_errorbar(aes(ymin = lo_CI_obs_prop, ymax = up_CI_obs_prop), width = 0.03) +
    geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "red") +
    labs(
      title = paste("Calibration Plot (", title_suffix, ", ", bins, " bins)", sep = ""),
      x = "Predicted Probability",
      y = "Observed Proportion"
    ) +
    coord_equal() +
    theme_minimal()
}





plot_roc <- function(y_obs, y_pred, title_suffix = "Set") {
  library(pROC)
  
  # Create ROC object
  roc_obj <- roc(y_obs, y_pred)
  
  # Compute AUC
  auc_value <- auc(roc_obj)
  
  # Plot ROC curve
  plot(
    roc_obj, 
    col = "#2c7fb8", 
    lwd = 2, 
    main = paste("ROC Curve -", title_suffix)
  )
  
  # Add AUC as legend
  legend(
    "bottomright", 
    legend = paste("AUC =", round(auc_value, 3)), 
    col = "#2c7fb8", 
    lwd = 2
  )
  
  # Return AUC invisibly if needed for reporting
  invisible(auc_value)
}



# plots the ITE density on the left and ITE-ATE plot on the right

plot_for_slides_IST <- function(model.results, breaks, delta_horizontal = 0.02,
                                ylim_delta = 0.1, xlim_delta = 0.1) {
  
  
  layout_matrix <- matrix(c(
    1, 2, 2,
    1, 2, 2
  ), nrow = 2, byrow = TRUE)
  
  layout(mat = layout_matrix, heights = c(1, 1.3))
  par(mar = c(4.5, 4.5, 2, 1), mgp = c(2.2, 0.6, 0))
  
  
  
  #### Plot 1: ITE Densities
  plot(density(model.results$data.dev.rs$ITE), 
       main = "", 
       xlab = "ITE (Predicted)", 
       ylab = "Density", 
       lty = 1, 
       col = "orange", 
       lwd = 2, 
       ylim = range(0, density(model.results$data.dev.rs$ITE)$y,
                    density(model.results$data.val.rs$ITE)$y))
  mtext(expression("ITE Density"), side = 3, line = 0.5, cex = 0.95)
  
  lines(density(model.results$data.val.rs$ITE), col = "#36648B", lwd = 2)
  legend("topright", legend = c("Training", "Test"),
         col = c("orange", "#36648B"), lwd = 2, bty = "n")
  
  #### Plot 2: ATE vs ITE
  plot_ATE_vs_ITE_base_RiskDiff(model.results = model.results, 
                                breaks = breaks, 
                                delta_horizontal = delta_horizontal, 
                                ylim_delta = ylim_delta,
                                xlim_delta = xlim_delta)
}



# ---- functions for problem identification ----------

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





## not used? 

# plot_ATE_ITE_in_group_risks <- function(dev.data = data.dev.rs, val.data = data.val.rs, ylb=0, yub=2){
#   data <- rbind(dev.data %>% mutate(sample = "derivation"), val.data %>%  mutate(sample = "validation"))
#   result <- ggplot(data, aes(x = ITE.Group, y = ATE.RiskDiff)) +
#     geom_line(aes(group = sample, color = sample), linewidth = 1, 
#               position = position_dodge(width = 0.2)) +
#     geom_point(aes(color = sample), size = 1.5, 
#                position = position_dodge(width = 0.2)) +
#     geom_errorbar(aes(ymin = ATE.lb, ymax = ATE.ub, color = sample), width = 0.2,
#                   position = position_dodge(width = 0.2))+
#     geom_hline(yintercept = 0, linetype = "dashed", color = "black") +
#     scale_color_manual(name = "Group",
#                        labels = c("derivation" = "Training Data", "validation" = "Test Data"),
#                        values = c("orange", "#36648B"))+
#     scale_x_discrete(guide = guide_axis(n.dodge = 2))+
#     ylim(ylb,yub)+
#     xlab("ITE Group")+
#     ylab("ATE in Risk Difference")+
#     theme_minimal()+
#     theme(
#       legend.position.inside = c(0.9, 0.9),
#       legend.justification = c("right", "top"),
#       legend.box.just = "right",
#       panel.grid.major = element_blank(),  # Removes major grid lines
#       panel.grid.minor = element_blank(),  # Removes minor grid lines
#       panel.background = element_blank(),  # Removes panel background
#       plot.background = element_blank(),
#       text = element_text(size = 14),
#       axis.line = element_line(color = "black"),
#       axis.ticks = element_line(color = "black")
#     )
#   
#   return(result)
# }
# 
# 


# plot_ITE_density <- function(test.results){
#   result <- ggplot()+
#     geom_density(aes(x = test.results$data.dev.rs$ITE, fill = "ITE.dev", color = "ITE.dev"), alpha = 0.5, linewidth=1) +
#     geom_density(aes(x = test.results$data.val.rs$ITE, fill = "ITE.val", color = "ITE.val"), alpha = 0.5, linewidth=1) +
#     #  geom_vline(aes(xintercept = mean(test.results$data.dev.rs$ITE)), 
#     #             color = "orange", linetype = "dashed") +
#     #  geom_vline(aes(xintercept = mean(test.results$data.val.rs$ITE)), 
#     #             color = "#36648B", linetype = "dashed") +
#     geom_vline(aes(xintercept = 0), color = "black", linetype = "dashed", linewidth=1) +
#     xlab("Individualized Treatment Effect") +
#     ylab("Density") +
#     scale_color_manual(name = "Group", 
#                        labels = c("ITE.dev" = "Training Data", "ITE.val" = "Test Data"), 
#                        values = c("orange", "#36648B")) +
#     scale_fill_manual(name = "Group", 
#                       labels = c("ITE.dev" = "Training Data", "ITE.val" = "Test Data"), 
#                       values = c("orange", "#36648B")) +
#     theme_minimal()+
#     theme(
#       legend.position.inside = c(1, 1),
#       legend.justification = c("right", "top"),
#       legend.box.just = "right",
#       panel.grid.major = element_blank(),  # Removes major grid lines
#       panel.grid.minor = element_blank(),  # Removes minor grid lines
#       panel.background = element_blank(),  # Removes panel background
#       plot.background = element_blank(),    # Removes plot background (outside the plot)
#       text = element_text(size = 14),
#       axis.line = element_line(color = "black"),
#       axis.ticks = element_line(color = "black")
#     )
#   return(result)
# }



# calc.ATE.Odds.tram <- function(data, log.odds = T){
#   data <- as.data.frame(data)
#   data$OUTCOME6M <- data$OUTCOME6M - 1 #transform back to 0, 1 coding
#   model <- glm(OUTCOME6M ~ RXASP, data = data, family = binomial(link = "logit"))
#   if (!log.odds){
#     ATE.odds <- exp(coef(model)[2])
#     ATE.lb <- exp(suppressMessages(confint(model)[2,1]))
#     ATE.ub <- exp(suppressMessages(confint(model)[2,2]))
#   } else {
#     ATE.odds <- coef(model)[2]
#     ATE.lb <- suppressMessages(confint(model)[2,1])
#     ATE.ub <- suppressMessages(confint(model)[2,2])
#   }
#   return(data.frame(ATE.odds = ATE.odds, ATE.lb = ATE.lb, ATE.ub = ATE.ub, 
#                     n.total = nrow(data), 
#                     n.tr = sum(data$RXASP == 1), n.ct = sum(data$RXASP == 0)))
# }




# plot_ATE_ITE_in_group <- function(dev.data = data.dev.rs, val.data = data.val.rs, log.odds = T, ylb=0, yub=2){
#   data <- rbind(dev.data %>% mutate(sample = "derivation"), val.data %>%  mutate(sample = "validation"))
#   result <- ggplot(data, aes(x = ITE.Group, y = ATE.odds)) +
#     geom_line(aes(group = sample, color = sample), linewidth = 1, 
#               position = position_dodge(width = 0.2)) +
#     geom_point(aes(color = sample), size = 1.5, 
#                position = position_dodge(width = 0.2)) +
#     geom_errorbar(aes(ymin = ATE.lb, ymax = ATE.ub, color = sample), width = 0.2,
#                   position = position_dodge(width = 0.2))+
#     geom_hline(yintercept = ifelse(log.odds, 0, 1), linetype = "dashed", color = "black") +
#     scale_color_manual(name = "Group",
#                        labels = c("derivation" = "Training Data", "validation" = "Test Data"),
#                        values = c("orange", "#36648B"))+
#     scale_x_discrete(guide = guide_axis(n.dodge = 2))+
#     ylim(ylb,yub)+
#     xlab("ITE Group")+
#     ylab(ifelse(log.odds, "ATE in Log Odds Ratio", "ATE in Odds Ratio"))+
#     theme_minimal()+
#     theme(
#       legend.position.inside = c(0.9, 0.9),
#       legend.justification = c("right", "top"),
#       legend.box.just = "right",
#       panel.grid.major = element_blank(),  # Removes major grid lines
#       panel.grid.minor = element_blank(),  # Removes minor grid lines
#       panel.background = element_blank(),  # Removes panel background
#       plot.background = element_blank(),
#       text = element_text(size = 14),
#       axis.line = element_line(color = "black"),
#       axis.ticks = element_line(color = "black")
#     )
#   
#   return(result)
# }
