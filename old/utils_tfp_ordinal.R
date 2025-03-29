



do_dag_struct = function(param_model, MA, MA_encoded, doX = c(NA, NA, NA), num_samples=1042, data = train){
  num_samples = as.integer(num_samples)
  
  # N_original <- ncol(MA)
  
  
  N = length(doX) #NUmber of nodes
  
  #### Checking the input #####
  stopifnot(is_upper_triangular(MA)) #MA needs to be upper triangular
  # stopifnot(param_model$input$shape[2L] == N) #Same number of variables       ############## not the case with 5 vars here
  stopifnot(nrow(MA) == N)           #Same number of variables
  stopifnot(sum(is.na(doX)) >= N-1) #Currently only one Variable with do(might also work with more but not tested)
  
  # Looping over the variables assuming causal ordering
  #Sampling (or replacing with do) of the current variable x
  xl = list() 
  # s = tf$ones(c(num_samples, N))
  s_e = tf$ones(c(num_samples, nrow(MA_encoded)))                               ###### change to 5 columns instead of 3
  for (i in 1:N){
    # i = 1
    ts = NA
    parents = which(MA_encoded[,i] != "0")                                      # Use parents from 5x3
    # parents[,i]
    if (length(parents) == 0) { #Root node?
      ones = tf$ones(shape=c(num_samples,1L),dtype=tf$float32)
      if(is.na(doX[i])){ #No do ==> replace with samples (conditioned on 1)
        ts = sample_from_target_MAF_struct(param_model, i, s_e)                 # sample with N x 5 parents matrix
      } else{
        ts = doX[i] * ones #replace with do
      }
    } else { #No root node ==> the parents are present 
      if(is.na(doX[i])){ #No do ==> replace with samples (conditioned on 1)
        ts = sample_from_target_MAF_struct(param_model, i, s_e)
      } else{ #Replace with do
        ones = tf$ones(shape=c(num_samples,1L),dtype=tf$float32) 
        ts = doX[i] * ones #replace with do
      }
    }
    
    # ts is now a matrix 5000 x 1 (samples are levels of ordinal variable)
    # ts
    
    # convert ts (5000 x 1) to encoded (5000 x2) (bzw. levels -1)
    
    # levels - 1 
    cuts <- as.numeric(max(levels(train$df_R[,i])))-1
    
    # Apply threshold encoding
    if (cuts == 2){
      ts_lower <- as.numeric(as.array(ts) %in% c(cuts, cuts+1))  # First threshold (2 or lower)
      ts_upper <- as.numeric(as.array(ts) == cuts+1)  # Second threshold (3)
      
      ts_encoded =  data.frame(ts_lower, ts_upper)
    } else if (cuts==1){
      ts_encoded = as.numeric(as.array(ts) == cuts+1)  # Second threshold (3)
    } else {
      stop("Amount of levels currently not possible in code")
    }
    

    # Put samples in a tensor
    ts_encoded_tf = tf$constant(as.matrix(ts_encoded), dtype = 'float32')
    
    
    # s_e
    #We want to add the samples to the ith column i.e. s[,i,drop=FALSE] = ts 
    # mask <- tf$one_hot(indices = as.integer(i - 1L), depth = tf$shape(s)[2], on_value = 1.0, off_value = 0.0, dtype = tf$float32)
    


    
    create_custom_mask <- function(i, depth = 5L) {
      #' Creates a custom mask tensor of shape 5 based on the input i in R, using one_hot.
      #'
      #' Args:
      #'     i: The index (1, 2, or 3).
      #'     depth: The desired depth of the output tensor (5).
      #'
      #' Returns:
      #'     A TensorFlow tensor of shape (5,) representing the custom mask.
      #'
      if (i == 1) {
        indices <- tf$constant(c(0L, 1L), dtype = tf$int32)
        mask <- tf$reduce_sum(tf$one_hot(indices = indices, depth = depth, on_value = 1.0, off_value = 0.0, dtype = tf$float32), axis = 0L)
        
      } else if (i == 2) {
        indices <- tf$constant(c(2L, 3L), dtype = tf$int32)
        mask <- tf$reduce_sum(tf$one_hot(indices = indices, depth = depth, on_value = 1.0, off_value = 0.0, dtype = tf$float32), axis = 0L)
        
      } else if (i == 3) {
        mask <- tf$one_hot(indices = as.integer(4L), depth = 5L, on_value = 1.0, off_value = 0.0, dtype = tf$float32)
        
      }  else {
        stop("i must be 1, 2, or 3.")
      }
      


      
      return(mask)
    }
    # 
    # # Example usage:
    # i1_mask <- create_custom_mask(1)
    # i2_mask <- create_custom_mask(2)
    # i3_mask <- create_custom_mask(3)
    # 
    
    mask <- create_custom_mask(i)

    
    
    # Adjust 'ts' to have the same second dimension as 's'
    # ts_expanded <- tf$broadcast_to(ts, tf$shape(s))
    
    # Subtract the i-th column from 's' and add the new values
    # s <- s - mask + ts_expanded * mask
    
    
    # replace the selected columns in s_e with ts_encoded_tf
    
    # Convert tensors to R matrices
    s_e_matrix <- as.matrix(s_e)
    ts_encoded_matrix <- as.matrix(ts_encoded_tf)
    mask_vector <- as.array(mask)  # Ensure mask is an R vector
    
    # Find column indices where mask == 1
    replace_cols <- which(mask_vector == 1)  # Should return c(1,2) in this case
    
    # Replace the selected columns in s_e_matrix
    s_e_matrix[, replace_cols] <- ts_encoded_matrix
    
    # Convert back to TensorFlow tensor
    s_e <- tf$convert_to_tensor(s_e_matrix, dtype = tf$float32)
    

  }
  
  # s_e is encoded and of shape=(5000, 5)
  
  # can be reconstructed as (eg. samples for X1):
  # train$df_R_encoded$x1_t1 + train$df_R_encoded$x1_t2 + 1
  
  # reconstruct s with decoded levels shape=(5000, 3)
  s_e_matrix <- as.matrix(s_e)
  s <- as.matrix(s)
  
  s[,1] <- s_e_matrix[,1] + s_e_matrix[,2] + 1
  s[,2] <- s_e_matrix[,3] + s_e_matrix[,4] + 1
  s[,3] <- s_e_matrix[,5] + 1
  
  # back to tensor
  s <- tf$convert_to_tensor(s, dtype = tf$float32)
  
  
  return(s)
}




sample_from_target_MAF_struct = function(param_model, node, parents){
  DEBUG_NO_EXTRA = FALSE
  # parents = s_e
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

