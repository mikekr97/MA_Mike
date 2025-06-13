

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
  
  x_lim_lower <- min(c(data.dev.rs$ITE, data.val.rs$ITE), na.rm = TRUE)-0.05
  x_lim_upper <- max(c(data.dev.rs$ITE, data.val.rs$ITE), na.rm = TRUE)+0.05
  
  p1 <- ggplot(data=data.dev.rs, aes(x=ITE, y=Y))+
    geom_point(aes(color=Treatment))+
    geom_smooth(aes(color=Treatment, fill=Treatment), method = "glm", method.args = list(family = "binomial"), alpha=0.5)+
    coord_cartesian(xlim=c(x_lim_lower, x_lim_upper), ylim = c(0,1))+
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
    coord_cartesian(xlim=c(x_lim_lower, x_lim_upper), ylim = c(0,1))+
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

















##### ------- MA Mike funcitons for ITE_observatinoal_simulation.R -------------


# baseline transformation function for dgp

# Define the interval boundaries for the core function
x_left_boundary <- -2
x_right_boundary <- 2
# Define the constant divisor as per the problem description (1/0.2)
constant_divisor <- 0.2

# --- Pre-calculate constants for linear extrapolation ---

# Calculate the y-values of the core function at the boundaries
# f(x) = tan(x/2) / 0.2
y_at_x_left_boundary <- tan(x_left_boundary / 2) / constant_divisor
y_at_x_right_boundary <- tan(x_right_boundary / 2) / constant_divisor

# Calculate the slopes (derivatives) of the core function at the boundaries
# f'(x) = d/dx (tan(x/2) / 0.2)
#       = (1 / 0.2) * (1/2) * sec^2(x/2)
#       = (1 / 0.4) * (1 / cos(x/2)^2)
slope_at_x_left_boundary <- (1 / constant_divisor) * (1 / 2) * (1 / cos(x_left_boundary / 2)^2)
slope_at_x_right_boundary <- (1 / constant_divisor) * (1 / 2) * (1 / cos(x_right_boundary / 2)^2)

# --- Define the h_y function ---

h_y <- function(x) {
  # Initialize a numeric vector to store the results, matching the length of x
  result <- numeric(length(x))
  
  # Identify indices for each segment
  indices_left_extrapolation <- which(x < x_left_boundary)
  indices_middle_segment <- which(x >= x_left_boundary & x <= x_right_boundary)
  indices_right_extrapolation <- which(x > x_right_boundary)
  
  # Apply the function logic for each segment
  if (length(indices_left_extrapolation) > 0) {
    # Linear extrapolation for x < -2: y = y_at_x_left_boundary + slope_at_x_left_boundary * (x - x_left_boundary)
    result[indices_left_extrapolation] <- y_at_x_left_boundary + slope_at_x_left_boundary * (x[indices_left_extrapolation] - x_left_boundary)
  }
  
  if (length(indices_middle_segment) > 0) {
    # Core function for -2 <= x <= 2: y = tan(x/2) / 0.4
    result[indices_middle_segment] <- tan(x[indices_middle_segment] / 2) / constant_divisor
  }
  
  if (length(indices_right_extrapolation) > 0) {
    # Linear extrapolation for x > 2: y = y_at_x_right_boundary + slope_at_x_right_boundary * (x - x_right_boundary)
    result[indices_right_extrapolation] <- y_at_x_right_boundary + slope_at_x_right_boundary * (x[indices_right_extrapolation] - x_right_boundary)
  }
  
  return(result)
}

# --- Define the h_y_inverse function ---

h_y_inverse <- function(y) {
  # Initialize a numeric vector to store the results, matching the length of y
  result <- numeric(length(y))
  
  # The y-values at the tangent points define the boundaries for the inverse function
  # These are the y-values at x = -2 and x = 2
  y_boundary_left_tangent_point <- y_at_x_left_boundary
  y_boundary_right_tangent_point <- y_at_x_right_boundary
  
  # Identify indices for each inverse segment based on y-values
  indices_inverse_left_extrapolation <- which(y <= y_boundary_left_tangent_point)
  indices_inverse_middle_segment <- which(y > y_boundary_left_tangent_point & y < y_boundary_right_tangent_point)
  indices_inverse_right_extrapolation <- which(y >= y_boundary_right_tangent_point)
  
  # Apply the inverse function logic for each segment
  if (length(indices_inverse_left_extrapolation) > 0) {
    # Inverse of left extrapolation: x = (y - y_at_x_left_boundary) / slope_at_x_left_boundary + x_left_boundary
    result[indices_inverse_left_extrapolation] <- (y[indices_inverse_left_extrapolation] - y_at_x_left_boundary) / slope_at_x_left_boundary + x_left_boundary
  }
  
  if (length(indices_inverse_middle_segment) > 0) {
    # Inverse of core function: y = tan(x/2) / 0.4 => 0.4 * y = tan(x/2) => x/2 = atan(0.4 * y) => x = 2 * atan(0.4 * y)
    result[indices_inverse_middle_segment] <- 2 * atan(constant_divisor * y[indices_inverse_middle_segment])
  }
  
  if (length(indices_inverse_right_extrapolation) > 0) {
    # Inverse of right extrapolation: x = (y - y_at_x_right_boundary) / slope_at_x_right_boundary + x_right_boundary
    result[indices_inverse_right_extrapolation] <- (y[indices_inverse_right_extrapolation] - y_at_x_right_boundary) / slope_at_x_right_boundary + x_right_boundary
  }
  
  return(result)
}


# --- Example usage of the functions ---

# Example usage of h_y and h_y_inverse functions
example_x_values <- seq(-4, 4, by = 0.1)  # Example x values for testing
example_y_values <- h_y(example_x_values)  # Calculate h_y for the example x values


# plot to check if it works:
# par(mfrow = c(1, 2))  
# plot(example_x_values, example_y_values, type = "l", col = "blue", lwd = 2,
#      main = "Plot of h_y(x)",
#      xlab = "x", ylab = "h_y(x)",
#      xlim = c(-6, 6), ylim = c(-10, 10))
# plot(example_x_values, example_y_values, type = "l", col = "blue", lwd = 2,
#      main = "Plot of h_y(x)",
#      xlab = "x", ylab = "h_y(x)",
#      xlim = c(-2, 2), ylim = c(-10, 10))
#
# # inverse of y value
#
# example_y_values_inverse <- h_y_inverse(example_y_values)  # Calculate h_y_inverse for the example y values
# plot(example_x_values, example_y_values_inverse)





calculate_ITE_median <- function(data){
  # data <- train
  
  # from the observed patient characteristics determine the latent value
  
  
  # NN outputs (CS, LS, theta') at the observed values (T=0)
  h_params_obs <- param_model(data$dat.tf)
  
  # combine outputs to the transformation function
  h_obs <- construct_h(t_i = data$dat.tf, h_params = h_params_obs)
  
  # this is the cut point for the Treatment (X4), it is not used because we will intervene on this variable
  # h_obs$h_ord_vars
  
  # these are the latent values for the continuous variables (X1, X2, X3, X5, X6, X7 (outcome, not used))
  # h_obs$h_cont_vars
  
  ### Note that we only use the latent values for the observed patient characteristics
  ### (X1, X2, X3, X5, X6) , where X5 and X6 depended on the treatment received (here T=0), 
  ### the received treatment was considered when h was constructed with construct_h
  ### because I assume that the patients were not yet treated, all is constructed with T=0
  
  ### in ITE estimation, the outcome is not known already, we only use the patient characteristics
  
  
  # potential outcome for T=0 (not treated)
  
  # h_obs$h_cont_vars
  
  ######## ITE as difference of median of Potential Outcomes
  
  # estimate the variables under Control and Treatment (last variable Y is the median of P(Y|X, T=0) and P(Y|X, T=1))
  outcome_ct <- do_dag_struct_ITE_obs_sim(param_model, train$A, h_obs$h_combined, 
                                          doX = c(NA, NA, NA, 0, NA, NA, NA), 
                                          num_samples=dim(h_obs$h_cont_vars)[1])
  y_obsZ_ct <- as.numeric(outcome_ct[, 7])
  
  outcome_tx <- do_dag_struct_ITE_obs_sim(param_model, train$A, h_obs$h_combined, 
                                          doX = c(NA, NA, NA, 1, NA, NA, NA), 
                                          num_samples=dim(h_obs$h_cont_vars)[1])
  y_obsZ_tx <- as.numeric(outcome_tx[, 7])
  
  ITE_obsZ_pred <- y_obsZ_tx - y_obsZ_ct
  # plot(ITE_obsZ_pred, data$simulated_full_data$ITE_true)
  
  
  ######## ITE as difference of Potential Outcomes at observed latent values
  # yields same ITE as with median when transformation function for Y in DGP is linear
  
  # i = 7  # target node Y (X7)
  # 
  # # generate samples for target node under T=0 and T=1 with the observed latent value for Y
  # ts_ct = sample_from_target_MAF_struct_ITE_obs_sim(param_model, latent_observed = h_obs$h_combined,
                                                    # doX = c(NA, NA, NA, 0, NA, NA, NA), i, parents=outcome_ct)
  # ts_tx = sample_from_target_MAF_struct_ITE_obs_sim(param_model,  latent_observed = h_obs,
  #                                                   doX = c(NA, NA, NA, 1, NA, NA, NA), i, parents=outcome_tx)
  # 
  # ITE_obsZ_pred <- as.numeric(ts_tx) - as.numeric(ts_ct)
  
  # plot(as.numeric(ts_ct), as.numeric(outcome_ct[, 7]))
  
  i = 7  # target node Y (X7)
  
  # set column 7 of h_obs to 0 (median of latent distribution) 
  
  # Step 1: Convert tensor to R matrix
  h_mat <- as.matrix(h_obs$h_combined)
  
  # Step 2: Set last column to 0
  h_mat[, ncol(h_mat)] <- 0
  
  # Step 3: Convert back to TensorFlow tensor
  h_obs_y_median <- tf$convert_to_tensor(h_mat, dtype = tf$float32)
  
  
  # generate samples for target node under T=0 and T=1 with the median latent value for Y
  outcome_ct_median = sample_from_target_MAF_struct_ITE_obs_sim(param_model, latent_observed = h_obs_y_median, 
                                                    doX = c(NA, NA, NA, 0, NA, NA, NA), node=i, parents=outcome_ct)
  outcome_tx_median = sample_from_target_MAF_struct_ITE_obs_sim(param_model,  latent_observed = h_obs_y_median,
                                                    doX = c(NA, NA, NA, 1, NA, NA, NA), node=i, parents=outcome_tx)
  
  ITE_median_pred <- as.numeric(outcome_tx_median) - as.numeric(outcome_ct_median)
  
  # plot(ITE_median_pred, ITE_obsZ_pred)
  
  data$simulated_full_data <- data$simulated_full_data %>%
    dplyr::mutate(ITE_median_pred = ITE_median_pred,
                  ITE_obsZ_pred = ITE_obsZ_pred)
  
  return(list(
    outcome_ct = outcome_ct$numpy(),             # variables at obsZ under Control (T=0)
    outcome_tx = outcome_tx$numpy(),             # variables at obsZ under Treatment (T=1)
    outcome_ct_median = outcome_ct_median$numpy(), # only Y at obsZ under Control (T=0) with median latent value
    outcome_tx_median = outcome_tx_median$numpy(), # only Y at obsZ under Treatment (T=1) with median latent value
    ITE_median_pred = ITE_median_pred,
    ITE_obsZ_pred = ITE_obsZ_pred,
    data = data,
    latent_obs = h_obs$h_combined$numpy() # latent value observed (for Treatment (X4) this is just the cut-point)
    
  ))
  
  
}






construct_h <- function (t_i, h_params){
  
  #t_i <- data$dat.tf   # original data x1, x2, x3 for each obs
  #h_params = h_params_obs    # NN outputs (CS, LS, theta') for each obs
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
  #Thetas for intercept -> to_theta3 to make them increasing
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
  
  ### Continiuous dimensions
  #### At least one continuous dimension exits
  if (length(cont_dims) != 0){
    
    # inputs in h_dag_extra:
    # data=(40000, 3), 
    # theta=(40000, 3, 20), k_min=(3), k_max=(3))
    
    # creates the value of the Bernstein at each observation
    # and current parameters: output shape=(40000, 3)
    # h_I = h_dag_extra(t_i[,cont_dims, drop=FALSE], theta[,cont_dims,1:len_theta,drop=FALSE], k_min[cont_dims], k_max[cont_dims])
    h_I = h_dag_extra(tf$gather(t_i, as.integer(cont_dims-1L), axis = 1L), 
                      tf$gather(theta, as.integer(cont_dims-1L), axis = 1L)[,,1:len_theta,drop=FALSE],
                      tf$gather(k_min, as.integer(cont_dims-1L)),
                      tf$gather(k_max, as.integer(cont_dims-1L)))
    
    
    # adding the intercepts and shifts: results in shape=(40000, 3)
    # basically the estimated value of the latent variable
    h_cont_vars = h_I + tf$gather(h_LS, as.integer(cont_dims-1L), axis = 1L) + 
      tf$gather(h_CS, as.integer(cont_dims-1L), axis = 1L)
    
    
  }
  
  ### Ordinal dimensions
  if (length(cont_ord) != 0){
    B = tf$shape(t_i)[1]
    for (col in cont_ord){
      # col=4
      # nol = tf$cast(k_max[col] - 1L, tf$int32) # Number of cut-points in respective dimension
      nol = tf$cast(k_max[col], tf$int32) # Number of cut-points in respective dimension (binary encoded)
      
      theta_ord = theta[,col,1:nol,drop=TRUE] # Intercept (2 values per observation if 2 cutpoints)
      
      
      h_ord_vars = theta_ord + h_LS[,col, drop=FALSE] + h_CS[,col, drop=FALSE]
    }
  }
  
  
  
  # combine continuous and ordinal variables to tensor according to data_type = c( "c" "c" "c" "o" "c" "c" "c")
  
  # Split the continuous tensor before and after the ordinal variable
  h_cont_before = h_cont_vars[, 1:3]  # columns 0, 1, 2
  h_cont_after = h_cont_vars[, 4:6]   # columns 3, 4, 5
  
  # Concatenate in the order: c c c o c c c
  h_combined <- tf$concat(list(h_cont_before, h_ord_vars, h_cont_after), axis = 1L)
  
  
  ### DEBUG 
  #if (sum(is.infinite(log_lik$numpy())) > 0){
  #  print("Hall")
  #}
  return (list(
    h_cont_vars = h_cont_vars, 
    h_ord_vars = h_ord_vars,
    h_combined = h_combined))
}









do_dag_struct_ITE_obs_sim = function(param_model, MA, latent_observed, doX = c(NA, NA, NA, 0, NA, NA, NA), num_samples=1042){
  num_samples = as.integer(num_samples)
  # MA <- train$A
  
  # observed latent variables
  # latent_observed <- h_obs$h_combined
  
  N = length(doX) #NUmber of nodes
  
  #### Checking the input #####
  stopifnot(is_upper_triangular(MA)) #MA needs to be upper triangular
  stopifnot(param_model$input$shape[2L] == N) #Same number of variables
  stopifnot(nrow(MA) == N)           #Same number of variables
  stopifnot(sum(is.na(doX)) >= N-1) #Currently only one Variable with do(might also work with more but not tested)
  
  
  # if doX contains 0 at index 4 then, get the samples for T=0
  # if (doX[4] == 0) {
  #   latent_observed
  #   sample_from_target_MAF_struct_ITE_obs_sim(param_model, latent_observed, i, s)
  # }
  
  
  # Looping over the variables assuming causal ordering
  #Sampling (or replacing with do) of the current variable x
  xl = list() 
  s = tf$ones(c(num_samples, N))
  for (i in 1:N){
    # i = 7
    ts = NA
    parents = which(MA[,i] != "0")
    if (length(parents) == 0) { #Root node?
      ones = tf$ones(shape=c(num_samples,1L),dtype=tf$float32)
      if(is.na(doX[i])){ #No do ==> replace with samples (conditioned on 1)
        ts = sample_from_target_MAF_struct_ITE_obs_sim(param_model, latent_observed, doX, i, s)
      } else{
        ts = doX[i] * ones #replace with do
      }
    } else { #No root node ==> the parents are present 
      if(is.na(doX[i])){ #No do ==> replace with samples (conditioned on 1)
        ts = sample_from_target_MAF_struct_ITE_obs_sim(param_model, latent_observed, doX, i, s)
        if (is.na(doX[4])){
          cat("Attention: Treatment (X4) is NA, but has to be specified for ITE")
        }
      } else{ #Replace with do
        ones = tf$ones(shape=c(num_samples,1L),dtype=tf$float32)
        ts = doX[i] * ones #replace with do
        
      }
    }
    #We want to add the samples to the ith column i.e. s[,i,drop=FALSE] = ts 
    mask <- tf$one_hot(indices = as.integer(i - 1L), depth = tf$shape(s)[2], on_value = 1.0, off_value = 0.0, dtype = tf$float32)
    # Adjust 'ts' to have the same second dimension as 's'
    ts_expanded <- tf$broadcast_to(ts, tf$shape(s))
    # Subtract the i-th column from 's' and add the new values
    s <- s - mask + ts_expanded * mask
  }
  return(s)
}





sample_from_target_MAF_struct_ITE_obs_sim = function(param_model, latent_observed, doX, node, parents){
  DEBUG_NO_EXTRA = FALSE
  # parents = s
  # node = 7
  
  
  # if no parents, then h_params is model output for x1=1, x2=1, x3=1
  h_params = param_model(parents)
  
  # Extracting the CS & LS for each Sample and Variable
  h_cs <- h_params[,,1, drop = FALSE]
  h_ls <- h_params[,,2, drop = FALSE]
  
  # Extracting the theta' parameters and convert to (increasing) theta
  theta_tilde <- h_params[,,3:dim(h_params)[3], drop = FALSE]
  theta = to_theta3(theta_tilde)
  h_LS = tf$squeeze(h_ls, axis=-1L)
  h_CS = tf$squeeze(h_cs, axis=-1L)
  k_min <- k_constant(global_min)
  k_max <- k_constant(global_max)
  
  if(node %in% which(data_type == 'o')) {
    B = tf$shape(h_cs)[1]
    nol = tf$cast(k_max[node], tf$int32) # 1 cut-point (binary encoded)
    # theta_ord = theta[,node,1:nol,drop=TRUE] # Intercept
    # h = theta_ord + h_LS[,node, drop=FALSE] + h_CS[,node, drop=FALSE]
    
    # neg_inf = tf$fill(c(B,1L), -Inf)
    # pos_inf = tf$fill(c(B,1L), +Inf)
    # h_with_inf = tf$concat(list(neg_inf, h, pos_inf), axis=-1L)
    # logistic_cdf_values = logistic_cdf(h_with_inf)
    # #cdf_diffs <- tf$subtract(logistic_cdf_values[, 2:ncol(logistic_cdf_values)], logistic_cdf_values[, 1:(ncol(logistic_cdf_values) - 1)])
    # cdf_diffs <- tf$subtract(logistic_cdf_values[, 2:tf$shape(logistic_cdf_values)[2]], logistic_cdf_values[, 1:(tf$shape(logistic_cdf_values)[2] - 1)])
    # samples <- tf$random$categorical(logits = tf$math$log(cdf_diffs), num_samples = 1L)
    
    # set Treatment to 0 if doX[4] is 0
    if (doX[4] == 0) {
      # generate tensor of dim shape=(B, 1) with all values at 0
      samples <- tf$zeros(shape = c(B, 1L), dtype = tf$float32)
      
    } else if (doX[4] == 1) { # doX[4] == 1 means Treatment = 1
      # generate tensor of dim shape=(B, 1) with all values at 1
      samples <- tf$ones(shape = c(B, 1L), dtype = tf$float32)
      
    } else {
      stop("doX[4] must be 0, 1")
    }
    return(samples)
    # Picking the observed cdf_diff entry
  } else {
    #h_0_old =  tf$expand_dims(h_dag(L_START, theta), axis=-1L)
    #h_1 = tf$expand_dims(h_dag(R_START, theta), axis=-1L)
    
    # h_dag returns the intercept h (single value) at 0 and 1
    h_0 =  h_LS + h_CS + h_dag(L_START, theta) #tf$expand_dims(h_LS + h_CS + h_dag(L_START, theta), axis=-1L)
    h_1 =  h_LS + h_CS + h_dag(R_START, theta) #tf$expand_dims(h_LS + h_CS + h_dag(R_START, theta), axis=-1L)
    # if (DEBUG_NO_EXTRA){
    #   s = sample_logistics_within_bounds(h_0$numpy(), h_1$numpy())
    #   latent_sample = tf$constant(s)
    #   stop("Not IMplemented") #latent_sample = latent_dist$sample(theta_tilde$shape[1])
    # } else { #The normal case allowing extrapolations
    #   latent_sample = sample_standard_logistic(parents$shape)
    # }
    #ddd = target_sample$numpy() #hist(ddd[,1],100)
    
    #t_i = tf$ones_like(h_LS) *0.5
    #h_dag_extra_struc(t_i, theta, shift = h_LS + h_CS)
    #h_dag_extra(t_i, theta)
    # h_dag_extra_struc(target_sample, theta, shift, k_min, k_max) - latent_sample
    
    # We want to know for which t_i, h(t_i) is equal to the latent_sample
    # h(t_i) = rlogis()
    
    # for this we define function f(t_i) that is zero when the observation t_i fulfills the condition:
    # f(t_i) = h(t_i) - rlogis() == 0
    
    # for all explanatory variables, use the observed latent sample,
    # else (for the outcome Y) use the latent value 0 (median)
    
    
    # if(node !=length(doX)){
    #   latent_sample = latent_observed[, node, drop=FALSE]
    # } else {
    #   latent_sample = tf$zeros(shape = c(B, 1L), dtype = tf$float32) # median of logistic distribution
    # }
    

    latent_sample = latent_observed    #[, node, drop=FALSE]

    
    
    
    
    object_fkt = function(t_i){
      return(h_dag_extra_struc(t_i, theta, shift = h_LS + h_CS, k_min, k_max) - latent_sample)
    }
    #object_fkt(t_i)
    #shape = tf$shape(parents)[1]
    #target_sample = tfp$math$find_root_chandrupatla(object_fkt, low = -1E5*tf$ones(c(shape,1L)), high = 1E5*tf$ones(c(shape,1L)))$estimated_root
    #TODO better checking
    
    # find the root of f(t_i) = h(t_i) - rlogis() == 0, those samples are the target samples
    target_sample = tfp$math$find_root_chandrupatla(object_fkt)$estimated_root
    #target_sample = tfp$math$find_root_chandrupatla(object_fkt, low = -10000., high = 10000.)$estimated_root
    #wtfness = object_fkt(target_sample)$numpy()
    #summary(wtfness)
    
    
  

    # Manuly calculating the inverse for the extrapolated samples
    ## smaller than h_0
    l = latent_sample#tf$expand_dims(latent_sample, -1L)

    # check if the latent sample would be below h_0 (needs extrapolation)
    mask <- tf$math$less_equal(l, h_0)
    #cat(paste0('~~~ sample_from_target  Fraction of extrapolated samples < 0 : %f \n', tf$reduce_mean(tf$cast(mask, tf$float32))))
    #tf$where(mask, beta_dist_h$prob(y_i)* theta_im, h)
    slope0 <- h_dag_dash(L_START, theta)#tf$expand_dims(h_dag_dash(L_START, theta), axis=-1L)

    target_sample = tf$where(mask,
                             ((l-h_0)/slope0)*(k_max - k_min) + k_min
                             ,target_sample)

    ## larger than h_1
    mask <- tf$math$greater_equal(l, h_1)
    #tf$where(mask, beta_dist_h$prob(y_i)* theta_im, h)
    slope1<- h_dag_dash(R_START, theta)

    target_sample = tf$where(mask,
                             (((l-h_1)/slope1) + 1.0)*(k_max - k_min) + k_min,
                             target_sample)
    cat(paste0('sample_from_target Fraction of extrapolated samples > 1 : %f \n', tf$reduce_mean(tf$cast(mask, tf$float32))))
    return(target_sample[,node, drop=FALSE])
  }
}







predict_outcome = function(param_model, data){
  # parents = s
  # node = 7
  
  
  # if no parents, then h_params is model output for x1=1, x2=1, x3=1
  h_params = param_model(data)
  
  # Extracting the CS & LS for each Sample and Variable
  h_cs <- h_params[,,1, drop = FALSE]
  h_ls <- h_params[,,2, drop = FALSE]
  
  # Extracting the theta' parameters and convert to (increasing) theta
  theta_tilde <- h_params[,,3:dim(h_params)[3], drop = FALSE]
  theta = to_theta3(theta_tilde)
  h_LS = tf$squeeze(h_ls, axis=-1L)
  h_CS = tf$squeeze(h_cs, axis=-1L)
  k_min <- k_constant(global_min)
  k_max <- k_constant(global_max)
  
  B = dim(h_params)[1]
  
  # h_dag returns the intercept h (single value) at 0 and 1
  h_0 =  h_LS + h_CS + h_dag(L_START, theta) #tf$expand_dims(h_LS + h_CS + h_dag(L_START, theta), axis=-1L)
  h_1 =  h_LS + h_CS + h_dag(R_START, theta) #tf$expand_dims(h_LS + h_CS + h_dag(R_START, theta), axis=-1L)

  latent_sample = tf$zeros(shape = c(B, 1L), dtype = tf$float32) # median of logistic distribution
      

  object_fkt = function(t_i){
    return(h_dag_extra_struc(t_i, theta, shift = h_LS + h_CS, k_min, k_max) - latent_sample)
  }


  target_sample = tfp$math$find_root_chandrupatla(object_fkt)$estimated_root
    
  # Manualy calculating the inverse for the extrapolated samples
  ## smaller than h_0
  l = latent_sample#tf$expand_dims(latent_sample, -1L)
    
  # check if the latent sample would be below h_0 (needs extrapolation)
  mask <- tf$math$less_equal(l, h_0)

  slope0 <- h_dag_dash(L_START, theta)#tf$expand_dims(h_dag_dash(L_START, theta), axis=-1L)
  
  target_sample = tf$where(mask,
                           ((l-h_0)/slope0)*(k_max - k_min) + k_min
                           ,target_sample)
    
  ## larger than h_1
  mask <- tf$math$greater_equal(l, h_1)
  #tf$where(mask, beta_dist_h$prob(y_i)* theta_im, h)
  slope1<- h_dag_dash(R_START, theta)
  
  target_sample = tf$where(mask,
                           (((l-h_1)/slope1) + 1.0)*(k_max - k_min) + k_min,
                           target_sample)
  
  return(target_sample[,7, drop=FALSE])
}













# Function to calculate ATE (difference in means) for continuous outcomes
calc.ATE.Continuous <- function(data) {
  data <- as.data.frame(data)
  
  # Mean outcomes in treated and control
  mean1 <- mean(data$Y[data$Tr == 1])
  mean0 <- mean(data$Y[data$Tr == 0])
  
  # ATE
  ATE <- mean1 - mean0
  
  # Standard error for difference in means
  n1 <- sum(data$Tr == 1)
  n0 <- sum(data$Tr == 0)
  sd1 <- sd(data$Y[data$Tr == 1])
  sd0 <- sd(data$Y[data$Tr == 0])
  se <- sqrt((sd1^2 / n1) + (sd0^2 / n0))
  
  # 95% Confidence Interval
  z <- qnorm(0.975)
  ATE.lb <- ATE - z * se
  ATE.ub <- ATE + z * se
  
  return(data.frame(
    ATE = ATE,
    ATE.lb = ATE.lb,
    ATE.ub = ATE.ub,
    n.total = nrow(data),
    n.tr = n1,
    n.ct = n0
  ))
}


# Function to create the CATE vs ITE group plot
plot_CATE_vs_ITE_group <- function(dev.data, val.data) {
  data <- rbind(
    dev.data %>% mutate(sample = "derivation"),
    val.data %>% mutate(sample = "validation")
  )
  
  result <- ggplot(data, aes(x = ITE.Group, y = ATE)) +
    geom_line(aes(group = sample, color = sample), linewidth = 1, position = position_dodge(width = 0.2)) +
    geom_point(aes(color = sample), size = 1.5, position = position_dodge(width = 0.2)) +
    geom_errorbar(aes(ymin = ATE.lb, ymax = ATE.ub, color = sample), width = 0.2, position = position_dodge(width = 0.2)) +
    geom_hline(yintercept = 0, linetype = "dashed", color = "black") +
    scale_color_manual(
      name = "Group",
      labels = c("derivation" = "Training Data", "validation" = "Test Data"),
      values = c("orange", "#36648B")
    ) +
    scale_x_discrete(guide = guide_axis(n.dodge = 2)) +
    ylim(min(dev.data$ATE.lb) - 0.1, max(dev.data$ATE.ub) + 0.1) +
    xlab("ITE Group") +
    ylab("ATE (Difference in Means)") +
    theme_minimal() +
    theme(
      legend.position = c(0.9, 0.9),
      legend.justification = c("right", "top"),
      legend.box.just = "right",
      panel.grid.major = element_blank(),
      panel.grid.minor = element_blank(),
      panel.background = element_blank(),
      plot.background = element_blank(),
      text = element_text(size = 14),
      axis.line = element_line(color = "black"),
      axis.ticks = element_line(color = "black")
    )
  
  return(result)
}








# Function to calculate ATE (difference in means) for continuous outcomes
calc.ATE.Continuous.median <- function(data) {
  data <- as.data.frame(data)
  
  y_tx <- data$Y[data$Tr == 1]
  y_ct <- data$Y[data$Tr == 0]
  
  # Mean outcomes in treated and control
  median1 <- median(y_tx)
  median0 <- median(y_ct)
  
  # ATE (difference of medians)
  ATE <- median1 - median0
  
  # Number of observations in this bin
  n1 <- sum(data$Tr == 1)
  n0 <- sum(data$Tr == 0)
  

  # Set number of bootstrap iterations
  n_boot <- 10000
  
  # Bootstrap distribution of difference in medians
  boot_diff <- replicate(n_boot, {
    sample_tx <- sample(y_tx, length(y_tx), replace = TRUE)
    sample_ct <- sample(y_ct, length(y_ct), replace = TRUE)
    median(sample_tx) - median(sample_ct)
  })
  
  # Compute 95% confidence interval
  ATE.lb <- quantile(boot_diff, 0.025)
  ATE.ub <- quantile(boot_diff, 0.975)
  
  
  return(data.frame(
    ATE = ATE,
    ATE.lb = ATE.lb,
    ATE.ub = ATE.ub,
    n.total = nrow(data),
    n.tr = n1,
    n.ct = n0
  ))
}


# Function to create the CATE vs ITE group plot
plot_CATE_vs_ITE_group_median <- function(dev.data, val.data) {
  data <- rbind(
    dev.data %>% mutate(sample = "derivation"),
    val.data %>% mutate(sample = "validation")
  )
  
  result <- ggplot(data, aes(x = ITE.Group, y = ATE)) +
    geom_line(aes(group = sample, color = sample), linewidth = 1, position = position_dodge(width = 0.2)) +
    geom_point(aes(color = sample), size = 1.5, position = position_dodge(width = 0.2)) +
    geom_errorbar(aes(ymin = ATE.lb, ymax = ATE.ub, color = sample), width = 0.2, position = position_dodge(width = 0.2)) +
    geom_hline(yintercept = 0, linetype = "dashed", color = "black") +
    scale_color_manual(
      name = "Group",
      labels = c("derivation" = "Training Data", "validation" = "Test Data"),
      values = c("orange", "#36648B")
    ) +
    scale_x_discrete(guide = guide_axis(n.dodge = 2)) +
    ylim(min(c(dev.data$ATE.lb, val.data$ATE.lb)) - 0.1, max(c(dev.data$ATE.ub, val.data$ATE.ub)) + 0.1) +
    xlab("ITE Group") +
    ylab("ATE (Difference in Medians)") +
    theme_minimal() +
    theme(
      legend.position = c(0.9, 0.9),
      legend.justification = c("right", "top"),
      legend.box.just = "right",
      panel.grid.major = element_blank(),
      panel.grid.minor = element_blank(),
      panel.background = element_blank(),
      plot.background = element_blank(),
      text = element_text(size = 14),
      axis.line = element_line(color = "black"),
      axis.ticks = element_line(color = "black")
    )
  
  return(result)
}








# Function to create the CATE vs ITE group plot
plot_CATE_vs_ITE_group_median_with_theoretical <- function(dev.data, val.data, bin_centers) {
  data <- rbind(
    dev.data %>% mutate(sample = "derivation"),
    val.data %>% mutate(sample = "validation")
  )
  
  result <- ggplot(data, aes(x = ITE.Group, y = ATE)) +
    geom_line(aes(group = sample, color = sample), linewidth = 1, position = position_dodge(width = 0.2)) +
    geom_point(aes(color = sample), size = 1.5, position = position_dodge(width = 0.2)) +
    geom_errorbar(aes(ymin = ATE.lb, ymax = ATE.ub, color = sample), width = 0.2, position = position_dodge(width = 0.2)) +
    geom_hline(yintercept = 0, linetype = "dashed", color = "black") +
    scale_color_manual(
      name = "Group",
      labels = c("derivation" = "Training Data", "validation" = "Test Data", "theoretical" = "Theoretical Center"),
      values = c("orange", "#36648B", "black")
    ) +
    scale_x_discrete(guide = guide_axis(n.dodge = 2)) +
    geom_line(data = bin_centers, aes(x = ITE.Group, y = theoretical.center, group = 1, color = "theoretical"),
              linetype = "dashed")+
    ylim(min(c(dev.data$ATE.lb, val.data$ATE.lb)) - 0.1, max(c(dev.data$ATE.ub, val.data$ATE.ub)) + 0.1) +
    xlab("ITE Group") +
    ylab("ATE (Difference in Medians)") +
    theme_minimal() +
    theme(
      legend.position = c(0.9, 0.9),
      legend.justification = c("right", "top"),
      legend.box.just = "right",
      panel.grid.major = element_blank(),
      panel.grid.minor = element_blank(),
      panel.background = element_blank(),
      plot.background = element_blank(),
      text = element_text(size = 14),
      axis.line = element_line(color = "black"),
      axis.ticks = element_line(color = "black")
    )
  
  return(result)
}





