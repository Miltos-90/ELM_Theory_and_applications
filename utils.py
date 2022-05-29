from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
import torch
from extreme_learning_machine import ELM
import numpy as np
import time

def to_tensor(arr, dev): 
    # Convert array to tensor and send to device
    return torch.from_numpy(arr).float().to(dev)

def train_predict(df_train, df_val, no_targets, device, hidden_size, err_fcn):
    ''' Train on a train set / predict on a test set '''
    
    # Split X,y and convert to tensors
    X_train = to_tensor(df_train[:, :-no_targets], device)
    y_train = to_tensor(df_train[:, -no_targets:], device)
    X_val   = to_tensor(df_val[:, :-no_targets], device)
    y_val   = to_tensor(df_val[:, -no_targets:], device)
    
    # Instantiate ELM
    elm = ELM(input_size  = X_train.shape[1],
              hidden_size = hidden_size,
              activation  = 'tanh',
              device      = device)

    # Train and time it
    start = time.time()
    elm.fit(X_train, y_train)
    end  = time.time()
    dur = end - start
    
    # Predict and compute error
    y_val_hat = elm.predict(X_val).cpu().detach().numpy()
    y_val     = y_val.cpu().detach().numpy()
    error     = err_fcn(y_val, y_val_hat)
    
    return dur, error

def cross_validation(df, no_targets, kfold, device, hidden_size, error_metric, scale):
    ''' Perform cross validation procedure '''
    
    # Arrays to hold results for this set
    errors, train_times = [], []

    # Loop over folds
    for train_idx, val_idx in kfold.split(df):

        # Scale
        if scale:
            scaler   = MinMaxScaler(feature_range = (0, 1))
            df_train = scaler.fit_transform(df[train_idx])
            df_val   = scaler.transform(df[val_idx])
        else:
            df_train = df[train_idx]
            df_val   = df[val_idx]

        # Train / predict
        tr_time, error = train_predict(df_train, df_val, no_targets, device, hidden_size, error_metric)
            
        # Append to lists
        errors.append(error)
        train_times.append(tr_time)
        
    return errors, train_times
            


def model_selection(df, no_targets, no_CV, device, hyperparams, scale, error_metric, seed):
    ''' Model Selection Process '''
    # Best results so far
    best_error   = np.Inf
    best_err_sd  = None
    best_time    = None
    best_neurons = None

    # Configure k-fold
    kfold = KFold(n_splits = no_CV, shuffle = True, random_state = seed)

    # Loop over the hyperprameter sets
    for hidden_size in hyperparams:

        errors, train_times = cross_validation(df, no_targets, kfold, device, hidden_size, error_metric, scale)

        # Compute median error and training time for all folds
        mean_error = np.mean(errors)
        std_error  = np.std(errors)
        mean_time  = np.mean(train_times)

        # Check if this is the lowest error obtained so far
        if mean_error < best_error:
            best_time    = mean_time
            best_error   = mean_error
            best_err_sd  = std_error
            best_neurons = hidden_size
            
    return best_error, best_err_sd, best_time, best_neurons


def error_estimation(df, no_targets, no_CV, hyperparam_set, error_metric, device, seed):
    ''' Error Estimation Process '''
    
    # Configure k-fold procedure
    kf = KFold(n_splits = no_CV, shuffle = True, random_state = seed)

    test_errors = [] # List to hold results

    for train_idx, test_idx in kf.split(df):

        # Split
        df_train = df[train_idx, :]
        df_test  = df[test_idx, :]

        # Run inner CV (model selection)
        cv_error, cv_sd_error, train_time, hidden_size = model_selection(df_train, no_targets,
                                                                         no_CV  = no_CV - 1, 
                                                                         scale  = True,
                                                                         seed   = seed,
                                                                         device = device, 
                                                                         hyperparams  = hyperparam_set,
                                                                         error_metric = error_metric)

        # Scale
        scaler   = MinMaxScaler(feature_range = (0, 1))
        df_train = scaler.fit_transform(df_train)
        df_test  = scaler.transform(df_test)

        # Train / predict
        test_time, test_error = train_predict(df_train, df_test, no_targets, device, hidden_size, error_metric)

        # Append to list of errors
        test_errors.append(test_error)

    # Grab mean and std 
    mean_test_error = np.mean(test_errors)
    std_test_error  = np.std(test_errors)
    
    return mean_test_error, std_test_error, cv_error, cv_sd_error, train_time, test_time
    