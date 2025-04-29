

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



logis.ITE.simple <- function(data, p){
  variable_names <- paste0("X", 1:p)
  form <- as.formula(paste("Y ~ ", paste(c("Treatment", variable_names), collapse = " + ")))
  
  fit.dev <- glm(form, data = data$data.dev, family = binomial(link = "logit"))
  
  # Predict ITE on derivation sample
  pred.data.dev.N <- data$data.dev %>%
    dplyr::select(c("Treatment", variable_names)) %>%
    dplyr::mutate(Treatment = "N")
  pred.data.dev.Y <- data$data.dev %>%
    dplyr::select(c("Treatment", variable_names)) %>%
    dplyr::mutate(Treatment = "Y")
  
  pred.dev <- predict(fit.dev, newdata = pred.data.dev.Y, type = "response") -
    predict(fit.dev, newdata = pred.data.dev.N, type = "response")
  
  # Predict ITE on validation sample
  pred.data.val.N <- data$data.val%>%
    dplyr::select(c("Treatment", variable_names)) %>%
    dplyr::mutate(Treatment = "N")
  pred.data.val.Y <- data$data.val %>%
    dplyr::select(c("Treatment", variable_names)) %>%
    dplyr::mutate(Treatment = "Y")
  
  pred.val <- predict(fit.dev, newdata = pred.data.val.Y, type = "response") -
    predict(fit.dev, newdata = pred.data.val.N, type = "response")
  
  # generate data
  data.dev.rs <- data$data.dev %>% 
    mutate(ITE = pred.dev, RS = ifelse(ITE < 0, "benefit", "harm")) %>%
    mutate(RS = as.factor(RS))
  
  data.val.rs <- data$data.val %>% 
    mutate(ITE = pred.val, RS = ifelse(ITE < 0, "benefit", "harm")) %>%
    mutate(RS = as.factor(RS))
  
  return(list(data.dev.rs = data.dev.rs, data.val.rs = data.val.rs, 
              model.dev = fit.dev))
}




## ---- Calculate ATE -----
calc.ATE <- function(data){
  data <- as.data.frame(data)
  model <- glm(Y ~ Treatment, data = data, family = binomial(link = "logit"))
  ATE1 <- predict(model, newdata = data.frame(Treatment = "Y"), type = "response") - predict(model, newdata = data.frame(Treatment = "N"), type = "response")
  ATE2 <- mean(data$Y[data$Treatment == "Y"], na.rm = T) -
    mean(data$Y[data$Treatment == "N"], na.rm = T)
  
  return(data.frame(ATE1 = ATE1, ATE2 = ATE2))
}

calc.ATE.Odds <- function(data, log.odds = Treatment){
  data <- as.data.frame(data)
  model <- glm(Y ~ Treatment, data = data, family = binomial(link = "logit"))
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
                    n.tr = sum(data$Treatment == "Y"), n.ct = sum(data$Treatment == "N")))
}

calc.ATE.RR <- function(data){
  data <- as.data.frame(data)
  model <- glm(Y ~ Treatment, data = data, family = quasipoisson(link = "log"))
  ATE.odds <- exp(coef(model)[2])
  ATE.lb <- exp(suppressMessages(confint(model)[2,1]))
  ATE.ub <- exp(suppressMessages(confint(model)[2,2]))
  return(data.frame(ATE.odds = ATE.odds, ATE.lb = ATE.lb, ATE.ub = ATE.ub, 
                    n.total = nrow(data), 
                    n.tr = sum(data$Treatment == "Y"), n.ct = sum(data$Treatment == "N")))
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



## ---- ATE-ITE Visualization by groups for derivation and validation data ----------
plot_ATE_ITE_in_group <- function(dev.data = data.dev.rs, val.data = data.val.rs, train.data.name = train_data_name, test.data.name = test_data_name, log.odds = T, ylb=0, yub=2){
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
                       labels = c("derivation" = train.data.name, "validation" = test.data.name),
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

# ------------ New loss funciton for tram DAG with binary treatment



struct_dag_loss_ITE = function (t_i, h_params, binary_treatment = TRUE){
  # binary_treatment = TRUE means that we amend the loss for the first variable. 
  # only used for ITE see below in ordinal loss calculation
  #t_i = train$df_orig # (40000, 3)    # original data x1, x2, x3 for each obs
  # t_i = dgp_data$df_orig_train
  #h_params = h_params                 # NN outputs (CS, LS, theta') for each obs
  k_min <- k_constant(global_min)
  k_max <- k_constant(global_max)
  
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
  
  NLL = 0
  ### Continiuous dimensions
  #### At least one continuous dimension exits
  if (length(cont_dims) != 0){
    
    # inputs in h_dag_extra:
    # data=(40000, 3), 
    # theta=(40000, 3, 20), k_min=(3), k_max=(3))
    
    # creates the value of the Bernstein at each observation
    # and current parameters: output shape=(40000, 3)
    h_I = h_dag_extra(t_i[,cont_dims, drop=FALSE], theta[,cont_dims,1:len_theta,drop=FALSE], k_min[cont_dims], k_max[cont_dims]) 
    
    # adding the intercepts and shifts: results in shape=(40000, 3)
    # basically the estimated value of the latent variable
    h = h_I + h_LS[,cont_dims, drop=FALSE] + h_CS[,cont_dims, drop=FALSE]
    
    #Compute terms for change of variable formula
    
    # log of standard logistic density at h
    log_latent_density = -h - 2 * tf$math$softplus(-h) #log of logistic density at h
    
    ## h' dh/dtarget is 0 for all shift terms
    log_hdash = tf$math$log(tf$math$abs(
      h_dag_dash_extra(t_i[,cont_dims, drop=FALSE], theta[,cont_dims,1:len_theta,drop=FALSE], k_min[cont_dims], k_max[cont_dims]))
    ) - 
      tf$math$log(k_max[cont_dims] - k_min[cont_dims])  #Chain rule! See Hathorn page 12 
    
    NLL = NLL - tf$reduce_mean(log_latent_density + log_hdash)
  }
  
  ### Ordinal dimensions
  if (length(cont_ord) != 0){
    B = tf$shape(t_i)[1]
    for (col in cont_ord){
      # col=1
      
      if (binary_treatment == TRUE & col==1){
        nol = tf$cast(k_max[col], tf$int32)
      } else {
        nol = tf$cast(k_max[col] - 1L, tf$int32) # Number of cut-points in respective dimension
      }
      theta_ord = theta[,col,1:nol,drop=TRUE] # Intercept (2 values per observation if 2 cutpoints)
      
      
      h = theta_ord + h_LS[,col, drop=FALSE] + h_CS[,col, drop=FALSE]
      # putting -Inf and +Inf to the left and right of the cutpoints
      neg_inf = tf$fill(c(B,1L), -Inf)
      pos_inf = tf$fill(c(B,1L), +Inf)
      h_with_inf = tf$concat(list(neg_inf, h, pos_inf), axis=-1L)
      logistic_cdf_values = logistic_cdf(h_with_inf)
      #cdf_diffs <- tf$subtract(logistic_cdf_values[, 2:ncol(logistic_cdf_values)], logistic_cdf_values[, 1:(ncol(logistic_cdf_values) - 1)])
      cdf_diffs <- tf$subtract(logistic_cdf_values[, 2:tf$shape(logistic_cdf_values)[2]], logistic_cdf_values[, 1:(tf$shape(logistic_cdf_values)[2] - 1)])
      # Picking the observed cdf_diff entry
      
      if (binary_treatment == TRUE & col==1){
        class_indices <- tf$cast(t_i[, col], tf$int32)  # already zero based index
      } else {
        class_indices <- tf$cast(t_i[, col] - 1, tf$int32)  # Convert to zero-based index
      }
      
      # Create batch indices to pair with class indices
      batch_indices <- tf$range(tf$shape(class_indices)[1])
      # Combine batch_indices and class_indices into pairs of indices
      gather_indices <- tf$stack(list(batch_indices, class_indices), axis=1)
      cdf_diff_picked <- tf$gather_nd(cdf_diffs, gather_indices)
      # Gather the corresponding values from cdf_diffs
      NLL = NLL -tf$reduce_mean(tf$math$log(cdf_diff_picked))
    }
  }
  
  ### DEBUG 
  #if (sum(is.infinite(log_lik$numpy())) > 0){
  #  print("Hall")
  #}
  return (NLL)
}



