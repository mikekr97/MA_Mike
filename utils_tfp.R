



do_dag_struct = function(param_model, MA, doX = c(0.5, NA, NA, NA), num_samples=1042){
  num_samples = as.integer(num_samples)
  
  
  N = length(doX) #NUmber of nodes
  
  #### Checking the input #####
  stopifnot(is_upper_triangular(MA)) #MA needs to be upper triangular
  stopifnot(param_model$input$shape[2L] == N) #Same number of variables
  stopifnot(nrow(MA) == N)           #Same number of variables
  stopifnot(sum(is.na(doX)) >= N-1) #Currently only one Variable with do(might also work with more but not tested)
  
  # Looping over the variables assuming causal ordering
  #Sampling (or replacing with do) of the current variable x
  xl = list() 
  s = tf$ones(c(num_samples, N))
  for (i in 1:N){
    # i = 1
    ts = NA
    parents = which(MA[,i] != "0")
    if (length(parents) == 0) { #Root node?
      ones = tf$ones(shape=c(num_samples,1L),dtype=tf$float32)
      if(is.na(doX[i])){ #No do ==> replace with samples (conditioned on 1)
        ts = sample_from_target_MAF_struct(param_model, i, s)
      } else{
        ts = doX[i] * ones #replace with do
      }
    } else { #No root node ==> the parents are present 
      if(is.na(doX[i])){ #No do ==> replace with samples (conditioned on 1)
        ts = sample_from_target_MAF_struct(param_model, i, s)
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




sample_from_target_MAF_struct = function(param_model, node, parents){
  DEBUG_NO_EXTRA = FALSE
  # parents = s
  # node = 1
  
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
    nol = tf$cast(k_max[node] - 1L, tf$int32) # Number of cut-points in respective dimension
    theta_ord = theta[,node,1:nol,drop=TRUE] # Intercept
    h = theta_ord + h_LS[,node, drop=FALSE] + h_CS[,node, drop=FALSE]
    neg_inf = tf$fill(c(B,1L), -Inf)
    pos_inf = tf$fill(c(B,1L), +Inf)
    h_with_inf = tf$concat(list(neg_inf, h, pos_inf), axis=-1L)
    logistic_cdf_values = logistic_cdf(h_with_inf)
    #cdf_diffs <- tf$subtract(logistic_cdf_values[, 2:ncol(logistic_cdf_values)], logistic_cdf_values[, 1:(ncol(logistic_cdf_values) - 1)])
    cdf_diffs <- tf$subtract(logistic_cdf_values[, 2:tf$shape(logistic_cdf_values)[2]], logistic_cdf_values[, 1:(tf$shape(logistic_cdf_values)[2] - 1)])
    samples <- tf$random$categorical(logits = tf$math$log(cdf_diffs), num_samples = 1L)
    samples = tf$cast(samples * 1.0 + 1, dtype='float32')
    return(samples)
    # Picking the observed cdf_diff entry
  } else {
    #h_0_old =  tf$expand_dims(h_dag(L_START, theta), axis=-1L)
    #h_1 = tf$expand_dims(h_dag(R_START, theta), axis=-1L)
    
    # h_dag returns the intercept h (single value) at 0 and 1
    h_0 =  h_LS + h_CS + h_dag(L_START, theta) #tf$expand_dims(h_LS + h_CS + h_dag(L_START, theta), axis=-1L)
    h_1 =  h_LS + h_CS + h_dag(R_START, theta) #tf$expand_dims(h_LS + h_CS + h_dag(R_START, theta), axis=-1L)
    if (DEBUG_NO_EXTRA){
      s = sample_logistics_within_bounds(h_0$numpy(), h_1$numpy())
      latent_sample = tf$constant(s)
      stop("Not IMplemented") #latent_sample = latent_dist$sample(theta_tilde$shape[1])
    } else { #The normal case allowing extrapolations
      latent_sample = sample_standard_logistic(parents$shape)
    }
    #ddd = target_sample$numpy() #hist(ddd[,1],100)
    
    #t_i = tf$ones_like(h_LS) *0.5
    #h_dag_extra_struc(t_i, theta, shift = h_LS + h_CS)
    #h_dag_extra(t_i, theta)
    # h_dag_extra_struc(target_sample, theta, shift, k_min, k_max) - latent_sample
    
    # We want to know for which t_i, h(t_i) is equal to the latent_sample
    # h(t_i) = rlogis()
    
    # for this we define function f(t_i) that is zero when the observation t_i fulfills the condition:
    # f(t_i) = h(t_i) - rlogis() == 0
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

