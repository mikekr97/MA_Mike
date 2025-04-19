

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


