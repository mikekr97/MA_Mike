


# bern_utils #####
R_START = 1-0.0001 #1.0-1E-1
L_START = 0.0001


# Creates Autoregressive masks for a given adjency matrix and hidden features
create_masks <- function(adjacency, hidden_features=c(64, 64)) {
  out_features <- nrow(adjacency)
  in_features <- ncol(adjacency)
  
  #adjacency_unique <- unique(adjacency, MARGIN = 1)
  #inverse_indices <- match(as.matrix(adjacency), as.matrix(adjacency_unique))
  
  #np.dot(adjacency.astype(int), adjacency.T.astype(int)) == adjacency.sum(axis=-1, keepdims=True).T
  d = tcrossprod(adjacency * 1L) #  "transposed cross-product", equivalent to X %*% t(X)
  precedence <-  d == matrix(rowSums(adjacency * 1L), ncol=nrow(adjacency), nrow=nrow(d), byrow = TRUE)
  
  masks <- list()
  for (i in seq_along(c(hidden_features, out_features))) {
    if (i > 1) {
      mask <- precedence[, indices, drop = FALSE]
    } else {
      mask <- adjacency
    }
    
    if (all(!mask)) {
      stop("The adjacency matrix leads to a null Jacobian.")
    }
    
    if (i <= length(hidden_features)) {
      reachable <- which(rowSums(mask) > 0)
      if (length(reachable) > 0) {
        indices <- reachable[(seq_len(hidden_features[i]) - 1) %% length(reachable) + 1]
      } else {
        indices <- integer(0)
      }
      mask <- mask[indices, , drop = FALSE]
    } 
    #else {
    #  mask <- mask[inverse_indices, , drop = FALSE]
    #}
    masks[[i]] <- mask
  }
  return(masks)
}


###### LinearMasked ####
LinearMasked(keras$layers$Layer) %py_class% {
  
  initialize <- function(units = 32, mask = NULL, bias=TRUE, name = NULL, trainable = NULL, dtype = NULL) {
    super$initialize(name = name)
    self$units <- units
    self$mask <- mask  # Add a mask parameter
    self$bias = bias
    # The additional arguments (name, trainable, dtype) are not used but are accepted to prevent errors during deserialization
  }
  
  build <- function(input_shape) {
    self$w <- self$add_weight(
      name = "w",
      shape = shape(input_shape[[2]], self$units),
      initializer = "random_normal",
      trainable = TRUE
    )
    if (self$bias) {
      self$b <- self$add_weight(
        name = "b",
        shape = shape(self$units),
        initializer = "random_normal",
        trainable = TRUE
      )
    } else{
      self$b <- NULL
    }
    
    # Handle the mask conversion if it's a dictionary (when loaded from a saved model)
    if (!is.null(self$mask)) {
      np <- import('numpy')
      if (is.list(self$mask) || "AutoTrackable" %in% class(self$mask)) {
        # Extract the mask value and dtype from the dictionary
        mask_value <- self$mask$config$value
        mask_dtype <- self$mask$config$dtype
        print("Hallo Gallo")
        mask_dtype = 'float32'
        print(mask_dtype)
        # Convert the mask value back to a numpy array
        mask_np <- np$array(mask_value, dtype = mask_dtype)
        # Convert the numpy array to a TensorFlow tensor
        self$mask <- tf$convert_to_tensor(mask_np, dtype = mask_dtype)
      } else {
        # Ensure the mask is the correct shape and convert it to a tensor
        if (!identical(dim(self$mask), dim(self$w))) {
          stop("Mask shape must match weights shape")
        }
        self$mask <- tf$convert_to_tensor(self$mask, dtype = self$w$dtype)
      }
    }
  }
  
  call <- function(inputs) {
    if (!is.null(self$mask)) {
      # Apply the mask
      masked_w <- self$w * self$mask
    } else {
      masked_w <- self$w
    }
    if(!is.null(self$b)){
      tf$matmul(inputs, masked_w) + self$b
    } else{
      tf$matmul(inputs, masked_w)
    }
  }
  
  get_config <- function() {
    config <- super$get_config()
    config$units <- self$units
    config$mask <- if (!is.null(self$mask)) tf$make_ndarray(tf$make_tensor_proto(self$mask)) else NULL
    config
  }
}




create_param_net <- function(len_param, input_layer, layer_sizes, masks, 
                             last_layer_bias=TRUE,
                             dropout = FALSE, batchnorm = FALSE, activation = "relu") {
  outs = list()
  for (r in 1:len_param){
    d = input_layer
    if (length(layer_sizes) > 2){ #Hidden Layers
      for (i in 2:(length(layer_sizes) - 1)) {
        
        # d = layer_batch_normalization()(d)  # Batch Normalization
        d = LinearMasked(units=layer_sizes[i], mask=t(masks[[i-1]]))(d)
        if (batchnorm) {
          d = layer_batch_normalization()(d)  # Batch Normalization
        }
        # d = layer_activation(activation='sigmoid')(d)
        d = layer_activation(activation=activation)(d)
        if (dropout){
          d = layer_dropout(rate = 0.1)(d)  # Dropout
        }

      }
    } #add output layers
    out = LinearMasked(units=layer_sizes[length(layer_sizes)], mask=t(masks[[length(layer_sizes) - 1]]),bias=last_layer_bias)(d)
    outs = append(outs,tf$expand_dims(out, axis=-1L)) #Expand last dim for concatenating
  }
  outs_c = keras$layers$concatenate(outs, axis=-1L)
}


# Creates a keras layer which takes as input (None, |x|) and returns (None, |x|, 1) which are all zero 
create_null_net <- function(input_layer) {
  output_layer <- layer_lambda(input_layer, function(x) {
    # Create a tensor of zeros with the same shape as x
    zeros_like_x <- k_zeros_like(x)
    # Add an extra dimension to match the desired output shape (None, |x|, 1)
    expanded_zeros_like_x <- k_expand_dims(zeros_like_x, -1)
    return(expanded_zeros_like_x)
  })
  return(output_layer)
}

create_param_model = function(MA, hidden_features_I = c(2,2), len_theta=30, 
                              hidden_features_CS = c(2,2),
                              ...){
  
  # number of variable as input shape
  input_layer <- layer_input(shape = list(ncol(MA)))
  
  ##### Creating the Intercept Model
  if ('ci' %in% MA == TRUE) { # At least one 'ci' in model
    layer_sizes_I <- c(ncol(MA), hidden_features_I, nrow(MA))
    masks_I = create_masks(adjacency =  t(MA == 'ci'), hidden_features_I)
    h_I = create_param_net(len_param = len_theta, input_layer=input_layer, 
                           layer_sizes = layer_sizes_I, masks_I, 
                           last_layer_bias=TRUE,
                           dropout = FALSE, batchnorm = FALSE, activation = "relu")
    #dag_maf_plot(masks_I, layer_sizes_I)
    #model_ci = keras_model(inputs = input_layer, h_I)
  } else { # Adding simple intercepts
    layer_sizes_I = c(ncol(MA), nrow(MA))
    masks_I = list(matrix(FALSE, nrow=nrow(MA), ncol=ncol(MA)))
    h_I = create_param_net(len_param = len_theta, input_layer=input_layer, 
                           layer_sizes = layer_sizes_I, masks_I, 
                           last_layer_bias=TRUE,
                           dropout = FALSE, batchnorm = FALSE, activation = "relu")
    #dag_maf_plot(masks_I, layer_sizes_I)
  }
  
  ##### Creating the Complex Shift Model
  if ('cs' %in% MA == TRUE) { # At least one 'cs' in model
    layer_sizes_CS <- c(ncol(MA), hidden_features_CS, nrow(MA))
    masks_CS = create_masks(adjacency =  t(MA == 'cs'), hidden_features_CS)
    h_CS = create_param_net(len_param = 1, input_layer=input_layer, 
                            layer_sizes = layer_sizes_CS, masks_CS, 
                            last_layer_bias=FALSE,
                            dropout = FALSE, batchnorm = FALSE, activation = "relu")
    #dag_maf_plot(masks_CS, layer_sizes_CS)
    #dag_maf_plot_new(masks_CS, layer_sizes_CS)
    # model_cs = keras_model(inputs = input_layer, h_CS)
  } else { #No 'cs' term in model --> return zero
    h_CS = create_null_net(input_layer)
  }
  
  ##### Creating the Linear Shift Model
  if ('ls' %in% MA == TRUE) {
    #h_LS = keras::layer_dense(input_layer, use_bias = FALSE, units = 1L)
    layer_sizes_LS <- c(ncol(MA), nrow(MA))
    masks_LS = create_masks(adjacency =  t(MA == 'ls'), c())
    out = LinearMasked(units=layer_sizes_LS[2], mask=t(masks_LS[[1]]), bias=FALSE, name='beta')(input_layer) 
    h_LS = tf$expand_dims(out, axis=-1L)#keras$layers$concatenate(outs, axis=-1L)
    #dag_maf_plot(masks_LS, layer_sizes_LS)
    #model_ls = keras_model(inputs = input_layer, h_LS)
  } else {
    h_LS = create_null_net(input_layer)
  }
  #Keras does not work with lists (only in eager mode)
  #model = keras_model(inputs = input_layer, outputs = list(h_I, h_CS, h_LS))
  #Dimensions h_I (B,3,30) h_CS (B, 3, 1) h_LS(B, 3, 3)
  # Convention for stacking
  # 1       CS
  # 2->|X|+1 LS
  # |X|+2 --> Ende M 
  outputs_tensor = keras$layers$concatenate(list(h_CS, h_LS, h_I), axis=-1L)
  param_model = keras_model(inputs = input_layer, outputs = outputs_tensor)
  return(param_model)
}



###### to_theta3 ####
# See zuko but fixed for order 3
# Used in Loss

# ensures that theta's are increasing (theta1 left as it is, then increasing)
to_theta3 = function(theta_tilde){
  shift = tf$convert_to_tensor(log(2) * dim(theta_tilde)[[length(dim(theta_tilde))]] / 2)
  order = tf$shape(theta_tilde)[3]
  widths = tf$math$softplus(theta_tilde[,, 2L:order, drop=FALSE])
  widths = tf$concat(list(theta_tilde[,, 1L, drop=FALSE], widths), axis = -1L)
  return(tf$cumsum(widths, axis = -1L) - shift)
}

# Used in Loss

### h_dag
# returns bernstein polynomial from equation 4 (paper)
h_dag = function(t_i, theta){
  len_theta = tf$shape(theta)[3L] #TODO tied to 3er Tensors
  Be = bernstein_basis(t_i, len_theta-1L) 
  return (tf$reduce_mean(theta * Be, -1L))  
}

### h_dag_dash
h_dag_dash = function(t_i, theta){
  len_theta = tf$shape(theta)[3L] #TODO tied to 3er Tensors
  Bed = bernstein_basis(t_i, len_theta-2L) 
  dtheta = (theta[,,2:len_theta,drop=FALSE]-theta[,,1:(len_theta-1L), drop=FALSE])
  return (tf$reduce_sum(dtheta * Bed, -1L))
}


h_dag_extra = function(t_i, theta, k_min, k_max){
  DEBUG = FALSE
  t_i = (t_i - k_min)/(k_max - k_min) # Scaling
  t_i3 = tf$expand_dims(t_i, axis=-1L)
  # for t_i < 0 extrapolate with tangent at h(0)
  b0 <- tf$expand_dims(h_dag(L_START, theta),axis=-1L)
  slope0 <- tf$expand_dims(h_dag_dash(L_START, theta), axis=-1L) 
  # If t_i < 0, use a linear extrapolation
  mask0 <- tf$math$less(t_i3, L_START)
  h <- tf$where(mask0, slope0 * (t_i3 - L_START) + b0, t_i3)
  #if (DEBUG) printf('~~~ eval_h_extra  Fraction of extrapolated samples < 0 : %f \n', tf$reduce_mean(tf$cast(mask0, tf$float32)))
  
  #(for t_i > 1)
  b1 <- tf$expand_dims(h_dag(R_START, theta),axis=-1L)
  slope1 <-  tf$expand_dims(h_dag_dash(R_START, theta), axis=-1L)
  # If t_i > 1, use a linear extrapolation
  mask1 <- tf$math$greater(t_i3, R_START)
  h <- tf$where(mask1, slope1 * (t_i3 - R_START) + b1, h)
  if (DEBUG) printf('~~~ eval_h_extra  Fraction of extrapolated samples > 1 : %f \n', tf$reduce_mean(tf$cast(mask1, tf$float32)))
  
  # For values in between, use the original function
  mask <- tf$math$logical_and(tf$math$greater_equal(t_i3, L_START), tf$math$less_equal(t_i3, R_START))
  h <- tf$where(mask, tf$expand_dims(h_dag(t_i, theta), axis=-1L), h)
  # Return the mean value
  return(tf$squeeze(h))
}


# returns the shifted h() for the given t_i, with the quantiles linearly interpolated

h_dag_extra_struc = function(t_i, theta, shift, k_min, k_max){
  #Throw unsupported error
  DEBUG = FALSE
  #stop('Please check before removing')
  #k_min <- k_constant(global_min)
  #k_max <- k_constant(global_max)
  t_i = (t_i - k_min)/(k_max - k_min) # Scaling
  t_i3 = tf$expand_dims(t_i, axis=-1L)
  # if (length(t_i$shape) == 2) {
  #   t_i3 = tf$expand_dims(t_i, axis=-1L)
  # } 
  # for t_i < 0 extrapolate with tangent at h(0)
  b0 <- tf$expand_dims(h_dag(L_START, theta) + shift,axis=-1L)
  slope0 <- tf$expand_dims(h_dag_dash(L_START, theta), axis=-1L) 
  # If t_i < 0, use a linear extrapolation
  mask0 <- tf$math$less(t_i3, L_START)
  h <- tf$where(mask0, slope0 * (t_i3 - L_START) + b0, t_i3)
  #if (DEBUG) printf('~~~ eval_h_extra  Fraction of extrapolated samples < 0 : %f \n', tf$reduce_mean(tf$cast(mask0, tf$float32)))
  
  #(for t_i > 1)
  b1 <- tf$expand_dims(h_dag(R_START, theta) + shift,axis=-1L)
  slope1 <-  tf$expand_dims(h_dag_dash(R_START, theta), axis=-1L)
  # If t_i > 1, use a linear extrapolation
  mask1 <- tf$math$greater(t_i3, R_START)
  h <- tf$where(mask1, slope1 * (t_i3 - R_START) + b1, h)
  if (DEBUG) printf('~~~ eval_h_extra  Fraction of extrapolated samples > 1 : %f \n', tf$reduce_mean(tf$cast(mask1, tf$float32)))
  
  # For values in between, use the original function
  mask <- tf$math$logical_and(tf$math$greater_equal(t_i3, L_START), tf$math$less_equal(t_i3, R_START))
  h <- tf$where(mask, tf$expand_dims(h_dag(t_i, theta) + shift, axis=-1L), h)
  # Return the mean value
  return(tf$squeeze(h))
}

h_dag_dash_extra = function(t_i, theta, k_min, k_max){
  t_i = (t_i - k_min)/(k_max - k_min) # Scaling
  t_i3 = tf$expand_dims(t_i, axis=-1L)
  
  #Left extrapolation
  slope0 <- tf$expand_dims(h_dag_dash(L_START, theta), axis=-1L) # creates the slope by 
  mask0 <- tf$math$less(t_i3, L_START) 
  h_dash <- tf$where(mask0, slope0, t_i3)
  
  #Right extrapolation
  slope1 <-  tf$expand_dims(h_dag_dash(R_START, theta), axis=-1L)
  mask1 <- tf$math$greater(t_i3, R_START)
  h_dash <- tf$where(mask1, slope1, h_dash)
  
  #Interpolation
  mask <- tf$math$logical_and(tf$math$greater_equal(t_i3, L_START), tf$math$less_equal(t_i3, R_START))
  h_dash <- tf$where(mask, tf$expand_dims(h_dag_dash(t_i,theta),axis=-1L), h_dash)
  
  return (tf$squeeze(h_dash))
}


### Bernstein Basis Polynoms of order M (i.e. M+1 coefficients)
# return (B,Nodes,M+1)
bernstein_basis <- function(tensor, M) {
  # Ensure tensor is a TensorFlow tensor
  tensor <- tf$convert_to_tensor(tensor)
  dtype <- tensor$dtype
  M = tf$cast(M, dtype)
  # Expand dimensions to allow broadcasting
  tensor_expanded <- tf$expand_dims(tensor, -1L)
  # Ensuring tensor_expanded is within the range (0,1) 
  tensor_expanded = tf$clip_by_value(tensor_expanded, tf$keras$backend$epsilon(), 1 - tf$keras$backend$epsilon())
  k_values <- tf$range(M + 1L) #from 0 to M
  
  # Calculate the Bernstein basis polynomials
  log_binomial_coeff <- tf$math$lgamma(M + 1.) - 
    tf$math$lgamma(k_values + 1.) - 
    tf$math$lgamma(M - k_values + 1.)
  log_powers <- k_values * tf$math$log(tensor_expanded) + 
    (M - k_values) * tf$math$log(1 - tensor_expanded)
  log_bernstein <- log_binomial_coeff + log_powers
  
  return(tf$exp(log_bernstein))
}



#### Loss NLL

# Define the function to calculate the logistic CDF
logistic_cdf <- function(x) {
  return(tf$math$reciprocal(tf$math$add(1, tf$math$exp(-x))))
}


struct_dag_loss = function (t_i, h_params){
  #t_i = train$df_orig # (40000, 3)    # original data x1, x2, x3 for each obs
  #t_i <- dgp_data$df_orig_train
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
      nol = tf$cast(k_max[col] - 1L, tf$int32) # Number of cut-points in respective dimension
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
      class_indices <- tf$cast(t_i[, col] - 1, tf$int32)  # Convert to zero-based index
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




struct_dag_loss_ITE_interaction = function (t_i, h_params){
  #t_i = train$df_orig # (40000, 3)    # original data x1, x2, x3 for each obs
  #t_i <- dgp_data$df_orig_train
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
  

  col=5
  B = tf$shape(t_i)[1]
  nol = tf$cast(k_max[col] - 1L, tf$int32) # Number of cut-points in respective dimension
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
  class_indices <- tf$cast(t_i[, col] - 1, tf$int32)  # Convert to zero-based index
  # Create batch indices to pair with class indices
  batch_indices <- tf$range(tf$shape(class_indices)[1])
  # Combine batch_indices and class_indices into pairs of indices
  gather_indices <- tf$stack(list(batch_indices, class_indices), axis=1)
  cdf_diff_picked <- tf$gather_nd(cdf_diffs, gather_indices)
  # Gather the corresponding values from cdf_diffs
  NLL = NLL -tf$reduce_mean(tf$math$log(cdf_diff_picked))

  return (NLL)
}



struct_dag_loss_ITE_IST = function (t_i, h_params){
  #t_i = dat.train.tf # (40000, 3)    # original data x1, x2, x3 for each obs
  #t_i <- dgp_data$df_orig_train
  #h_params = h_params                 # NN outputs (CS, LS, theta') for each obs
  # k_min <- k_constant(global_min)
  # k_max <- k_constant(global_max)
  data_type <- type
  
  
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
  
  
  col=length(data_type) # outcome index
  B = tf$shape(t_i)[1]
  # nol = tf$cast(k_max[col] - 1L, tf$int32) # Number of cut-points in respective dimension
  nol <- 1 # 1 cut point for binary outcome
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
  class_indices <- tf$cast(t_i[, col] - 1, tf$int32)  # Convert to zero-based index
  # Create batch indices to pair with class indices
  batch_indices <- tf$range(tf$shape(class_indices)[1])
  # Combine batch_indices and class_indices into pairs of indices
  gather_indices <- tf$stack(list(batch_indices, class_indices), axis=1)
  cdf_diff_picked <- tf$gather_nd(cdf_diffs, gather_indices)
  # Gather the corresponding values from cdf_diffs
  NLL = NLL -tf$reduce_mean(tf$math$log(cdf_diff_picked))
  
  return (NLL)
}



struct_dag_loss_ITE_observational = function (t_i, h_params){
  #t_i = train$df_orig # (40000, 3)    # original data x1, x2, x3 for each obs
  #t_i <- dgp_data$df_orig_train
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
  
  NLL = 0
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
    h = h_I + tf$gather(h_LS, as.integer(cont_dims-1L), axis = 1L) + 
      tf$gather(h_CS, as.integer(cont_dims-1L), axis = 1L)
    
    #Compute terms for change of variable formula
    
    # log of standard logistic density at h
    log_latent_density = -h - 2 * tf$math$softplus(-h) #log of logistic density at h
    
    ## h' dh/dtarget is 0 for all shift terms
    # log_hdash = tf$math$log(tf$math$abs(
    #   h_dag_dash_extra(t_i[,cont_dims, drop=FALSE], theta[,cont_dims,1:len_theta,drop=FALSE], k_min[cont_dims], k_max[cont_dims]))
    # ) - 
    #   tf$math$log(k_max[cont_dims] - k_min[cont_dims])  #Chain rule! See Hathorn page 12 
    # 
    log_hdash = tf$math$log(tf$math$abs(
      h_dag_dash_extra(tf$gather(t_i, as.integer(cont_dims-1L), axis = 1L), 
                       tf$gather(theta, as.integer(cont_dims-1L), axis = 1L)[,,1:len_theta,drop=FALSE], 
                       tf$gather(k_min, as.integer(cont_dims-1L)),
                       tf$gather(k_max, as.integer(cont_dims-1L))))
    ) - 
      tf$math$log(tf$gather(k_max, as.integer(cont_dims-1L)) - tf$gather(k_min, as.integer(cont_dims-1L)))  #Chain rule! See Hathorn page 12 
    
    
    
    NLL = NLL - tf$reduce_mean(log_latent_density + log_hdash)
  }
  
  ### Ordinal dimensions
  if (length(cont_ord) != 0){
    B = tf$shape(t_i)[1]
    for (col in cont_ord){
      # col=4
      # nol = tf$cast(k_max[col] - 1L, tf$int32) # Number of cut-points in respective dimension
      nol = tf$cast(k_max[col], tf$int32) # Number of cut-points in respective dimension (binary encoded)
      
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
      # class_indices <- tf$cast(t_i[, col] - 1, tf$int32)  # Convert to zero-based index
      class_indices <- tf$cast(t_i[, col], tf$int32)  # already binary encoded 0,1
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



########### Interventions ######
#### Helper ####
is_upper_triangular <- function(mat) {
  # Ensure it's a square matrix
  if (nrow(mat) != ncol(mat)) {
    return(FALSE)
  }
  
  # Check if elements below the diagonal are zero
  for (i in 1:nrow(mat)) {
    for (j in 1:ncol(mat)) {
      if (j < i && mat[i, j] != 0) {
        return(FALSE)
      }
      if (j == i && mat[i, j] != 0) {
        return(FALSE)
      }
    }
  }
  
  return(TRUE)
}



########## Transformations ############
check_baselinetrafo = function(h_params){
  #h_params = param_model(train$df_orig)
  k_min <- k_constant(global_min)
  k_max <- k_constant(global_max)
  
  k_min.df <- k_min$numpy()
  k_max.df <- k_max$numpy()
  
  theta_tilde <- h_params[,,3:dim(h_params)[3], drop = FALSE]
  theta = to_theta3(theta_tilde)
  length.out  = nrow(theta_tilde)
  
  # generate dataframe for x1, x2, x3 ranging from lowest to highest
  X = matrix(NA, nrow=length.out, ncol=length(k_min))
  for(i in 1:length(k_min.df)){
    X[,i] = seq(k_min.df[i], k_max.df[i], length.out = length.out)
  }
  # convert to tensor
  t_i = k_constant(X)
  
  h_I = h_dag_extra(t_i, theta, k_min, k_max) 
  return(list(h_I = h_I$numpy(), Xs=X))
} 

sample_standard_logistic <- function(shape, epsilon=1e-7) {
  uniform_samples <- tf$random$uniform(shape, minval=0, maxval=1)
  clipped_uniform_samples <- tf$clip_by_value(uniform_samples, epsilon, 1 - epsilon)
  logistic_samples <- tf$math$log(clipped_uniform_samples / (1 - clipped_uniform_samples))
  return(logistic_samples)
}




# Load the required library
library(ggplot2)
library(grid)

# Function to draw the network
dag_maf_plot <- function(layer_masks, layer_sizes) {
  max_nodes <- max(layer_sizes)
  width <- max_nodes * 100
  min_x <- 0
  max_x <- width  # Adjust max_x to include input layer
  min_y <- Inf
  max_y <- -Inf
  
  # Create a data frame to store node coordinates
  nodes <- data.frame(x = numeric(0), y = numeric(0), label = character(0))
  
  # Draw the nodes for all layers
  for (i in 1:length(layer_sizes)) {
    size <- layer_sizes[i]
    layer_top <- max_nodes / 2 - size / 2
    
    for (j in 1:size) {
      x <- (i-1) * width
      y <- layer_top + j * 100
      label <- ifelse(i == 1, paste("x_", j, sep = ""), "")  # Add labels for the first column
      nodes <- rbind(nodes, data.frame(x = x, y = y, label=label))
      max_x <- max(max_x, x)
      min_y <- min(min_y, y)
      max_y <- max(max_y, y)
    }
  }
  
  # Create a data frame to store connection coordinates
  connections <- data.frame(x_start = numeric(0), y_start = numeric(0),
                            x_end = numeric(0), y_end = numeric(0))
  
  # Draw the connections
  for (i in 1:length(layer_masks)) {
    mask <- t(layer_masks[[i]])
    input_size <- nrow(mask)
    output_size <- ncol(mask)
    
    for (j in 1:input_size) {
      for (k in 1:output_size) {
        if (mask[j, k]) {
          start_x <- (i - 1) * width
          start_y <- max_nodes / 2 - input_size / 2 + j * 100
          end_x <- i * width
          end_y <- max_nodes / 2 - output_size / 2 + k * 100
          
          connections <- rbind(connections, data.frame(x_start = start_x, y_start = start_y,
                                                       x_end = end_x, y_end = end_y))
        }
      }
    }
  }
  
  
  # Create the ggplot object
  network_plot <- ggplot() +
    geom_segment(data = connections, aes(x = x_start, y = -y_start, xend = x_end, yend = -y_end),
                 color = 'black', size = 1,
                 arrow = arrow()) +
    geom_point(data = nodes, aes(x = x, y = -y), color = 'blue', size = 8,alpha = 0.5) +
    geom_text(data = nodes, aes(x = x, y = -y, label = label), vjust = 0, hjust = 0.5) +  # Add labels
    theme_void() 
  
  return(network_plot)
}




# make the plots a bit nicer (works for CS)
dag_maf_plot_new <- function(layer_masks, layer_sizes) {
  max_nodes <- max(layer_sizes)
  node_spacing <- 100
  layer_spacing <- 200
  width <- max_nodes * node_spacing
  
  # Create a data frame to store node coordinates
  nodes <- data.frame(x = numeric(0), y = numeric(0), label = character(0))
  
  for (i in 1:length(layer_sizes)) {
    size <- layer_sizes[i]
    layer_top <- max_nodes / 2 - size / 2
    
    for (j in 1:size) {
      x <- (i - 1) * layer_spacing
      y <- layer_top * node_spacing + j * node_spacing
      if (i == 1 || i == length(layer_sizes)) {
        label <- paste("x_", j, sep = "")
      } else {
        label <- ""
      }
      nodes <- rbind(nodes, data.frame(x = x, y = y, label = label))
    }
  }
  
  # Create a data frame to store connection coordinates
  connections <- data.frame(x_start = numeric(0), y_start = numeric(0),
                            x_end = numeric(0), y_end = numeric(0))
  
  for (i in 1:length(layer_masks)) {
    mask <- t(layer_masks[[i]])
    input_size <- nrow(mask)
    output_size <- ncol(mask)
    
    for (j in 1:input_size) {
      for (k in 1:output_size) {
        if (mask[j, k]) {
          start_x <- (i - 1) * layer_spacing
          start_y <- (max_nodes / 2 - input_size / 2 + j) * node_spacing
          end_x <- i * layer_spacing
          end_y <- (max_nodes / 2 - output_size / 2 + k) * node_spacing
          
          connections <- rbind(connections, data.frame(x_start = start_x, y_start = start_y,
                                                       x_end = end_x, y_end = end_y))
        }
      }
    }
  }
  
  # Plot using ggplot2: output plot should be squared and small
  network_plot <- ggplot() +
    geom_segment(data = connections,
                 aes(x = x_start, y = -y_start, xend = x_end, yend = -y_end),
                 color = 'black', size = 1, arrow = arrow(length = unit(0.2, "cm"))) +
    # a thick edge aroun the points
    # geom_point(data = nodes,
    #            aes(x = x, y = -y), color = 'steelblue', size = 10, alpha = 0.8, shape=21) +
    # geom_text(data = nodes,
    #           aes(x = x, y = -y, label = label), vjust = 0.5, size = 3.5) +
    geom_point(data = nodes,
               aes(x = x, y = -y),
               shape = 21,           # Circle with border
               fill = 'steelblue',   # Node fill color
               color = 'black',      # Border color
               stroke = 1,         # Border thickness
               size = 10, alpha = 0.3) +
    geom_text(data = nodes,
              aes(x = x, y = -y, label = label),
              vjust = 0.5, hjust = 0.5,
              size = 3.5)+
    theme_void()+
    coord_cartesian(xlim = c(min(nodes$x) - 50, max(nodes$x) + 50),
                    ylim = c(-max(nodes$y) - 50, -min(nodes$y) + 50))
  
  
  return(network_plot)
}

# dag_maf_plot_new(masks_CS, layer_sizes_CS)

