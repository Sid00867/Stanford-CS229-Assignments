import numpy as np

class LogisticRegressor:
    
    '''
    Logistic Regressor
    =================

    When to use:
    ------------
    Binary Classification
    
    Prerequisites:
    -------------
    - Model assumes the Label Column (column that needs prediction) contains binary values [1/0]
    - Model assumes all columns in the dataset have numerical values
    
    How to use:
    -----------
    1. Instantiate model object with the dataset (as Pandas DataFrame)
    2. Remove unnecessary columns using ``removeColumns()`` with a list of column names
    3. Standardize data using ``Standardize_data_Zscore()`` with a list of column names
       (Do not include columns with binary values)
    4. Prepare data with ``PrepareData()`` specifying the label column name
    5. Train the model with ``Train()``
       - *alpha*: value in range [0.0, 1.0] (determines learning rate)
       - Training methods: ``"newtons_convergance"`` (fast but computationally expensive) 
         or ``"batch_gradient_ascent"`` (slower but computationally inexpensive)
         or ``"gaussian_discriminant_analysis"`` (fastest, computationally inexpensive but assumes data is gaussian)
    6. Evaluate with ``Test()`` to check accuracy
    7. Use ``Predict()`` to make predictions on new data
    '''


    def __init__(self, dataset) -> None:
        self.params = np.zeros((1,1))
        self.dataset = dataset
        self.partitioned_data = []
    
    # ===============================================================

    def getParams(self):
        return self.params

    def getDataset(self):
        return self.dataset
    
    def setDataset(self, dataset):
        self.dataset = dataset

    def PrepareData(self,label_column_name,split_ratio_training=0.8):
        self.AddBiasColumn()
        X = self.dataset.drop(columns=[label_column_name]).values
        Y = self.dataset[label_column_name].values.reshape(X.shape[0], 1)
        np.random.seed()
        indices = np.random.permutation(X.shape[0])
        train_size = int(split_ratio_training * X.shape[0])
        X_train, X_test = X[indices[:train_size]], X[indices[train_size:]]
        Y_train, Y_test = Y[indices[:train_size]], Y[indices[train_size:]]

        self.partitioned_data =  [X_train, X_test, Y_train, Y_test]
        self.params = np.zeros((X.shape[1],1))

    def Standardize_data_Zscore(self, feature_list):
        if not hasattr(self, 'standardization_params'):
            self.standardization_params = {}
        
        for feature in feature_list:
            mean = np.mean(self.dataset[feature])
            std_dev = np.std(self.dataset[feature])
            self.dataset.loc[:, feature] = (self.dataset[feature] - mean) / std_dev
            self.standardization_params[feature] = (mean, std_dev)
    
    def AddBiasColumn(self):
        self.dataset.insert(0, 'bias', 1)
    
    def removeColumns(self,  column_names):
        self.dataset = self.dataset.drop(columns=column_names)

    # ===============================================================

    def sigmoid(self, theta, X):
        z = np.dot(X, theta)
        return  1/(1 + np.exp(-z))
    
    def likelihood(self, X, Y, theta):
        m = X.shape[0]
        epsilon = 1e-15  # Small value to prevent log(0)
        sigmoid_values = self.sigmoid(theta, X)
        return (1/m) * (np.sum(Y * np.log(sigmoid_values + epsilon)) + np.sum((1 - Y) * np.log(1 - sigmoid_values + epsilon)))
    
    def liklihood_gradient(self, X, Y , theta):
        m = X.shape[0]
        return (1/m)*np.dot(X.T, Y - self.sigmoid(theta, X))

    def newtons_convergance(self, X, Y, theta,alpha):
        m = X.shape[0]
        hessian = (-1/m) * np.dot(X.T, self.sigmoid(theta, X) * (1 - self.sigmoid(theta, X)) * X)
        # Add a small regularization term to the diagonal
        hessian += np.eye(hessian.shape[0]) * 1e-5  
        new_theta = theta - alpha*np.dot(np.linalg.inv(hessian), self.liklihood_gradient(X, Y, theta))
        return new_theta
    
    def batch_gradient_ascent(self, X, Y, theta, alpha):
        gradient = self.liklihood_gradient(X, Y, theta)
        new_theta = theta + alpha * gradient
        return new_theta

    def gaussian_discriminant_analysis(self, X, Y):
        Y = Y.flatten()
        self.gda_phi = np.sum(Y) / len(Y)
        self.gda_mu0 = np.mean(X[Y == 0], axis=0)
        self.gda_mu1 = np.mean(X[Y == 1], axis=0)
        mus = np.where(Y[:, np.newaxis] == 0, self.gda_mu0, self.gda_mu1)  
        diff = X - mus                                            
        self.gda_covMat = np.dot(diff.T, diff) / len(Y)           


    def Train(self, training_method, alpha=0.01, convergence_threshold=1e-5):
        self.training_method = training_method
        X = self.partitioned_data[0]
        Y = self.partitioned_data[2]
        if training_method != "gaussian_discriminant_analysis":
            theta = self.params
            likelihood = self.likelihood(X, Y, theta)
            prev_likelihood = likelihood + 2 * convergence_threshold  # Ensure the first iteration is not skipped

            while abs(likelihood - prev_likelihood) > convergence_threshold:
                prev_likelihood = likelihood
                if training_method == "batch_gradient_ascent":
                    theta = self.batch_gradient_ascent(X, Y, theta, alpha)
                elif training_method == "newtons_convergance":    
                    theta = self.newtons_convergance(X, Y, theta, alpha)
                likelihood = self.likelihood(X, Y, theta)
                # print(f"Likelihood: {likelihood}")

            self.params = theta
        else:
            self.gaussian_discriminant_analysis(X,Y)    

    def Test(self):
        X = self.partitioned_data[1]
        Y = self.partitioned_data[3]

        if self.training_method != "gaussian_discriminant_analysis":
            predictions = self.sigmoid(self.params, X)
            predictions = predictions >= 0.5  # Convert probabilities to binary predictions
        else:
            # Ensure regularization term is added for invertibility
            self.gda_covMat += np.eye(self.gda_covMat.shape[0]) * 1e-5  # Small regularization for numerical stability
            sigma_inv = np.linalg.inv(self.gda_covMat)  # Shape (n, n)

            # Calculate decision values for all samples
            diff0 = X - self.gda_mu0  # shape (m, n)
            diff1 = X - self.gda_mu1  # shape (m, n)

            term0 = np.sum(np.dot(diff0, sigma_inv) * diff0, axis=1)
            term1 = np.sum(np.dot(diff1, sigma_inv) * diff1, axis=1)

            # Decision rule
            decision = term1 - term0 + 2 * np.log(self.gda_phi / (1 - self.gda_phi))
            
            # Predict 1 if decision < 0, otherwise predict 0
            predictions = (decision < 0).astype(int)

        # Compute overall accuracy
        overall_accuracy = np.mean(predictions == Y)
        print(f"Overall Test Accuracy: {overall_accuracy * 100:.2f}%")


    def Predict(self, input_data):
        if isinstance(input_data, dict):
            if hasattr(self, 'standardization_params'):
                filtered_data = {k: v for k, v in input_data.items() if k in self.standardization_params}
                standardized_values = [
                    (filtered_data[k] - self.standardization_params[k][0]) / self.standardization_params[k][1]
                    if k in self.standardization_params else input_data[k]
                    for k in input_data.keys()
                ]
            else:
                standardized_values = [input_data[k] for k in input_data.keys()]
            
            input_data = np.array(standardized_values).reshape(1, -1)
        
        # Now input_data is (1, n) or (m, n)

        if hasattr(self, 'training_method') and self.training_method == "gaussian_discriminant_analysis":
            # --- GDA prediction ---
            diff0 = input_data - self.gda_mu0  # (m, n)
            diff1 = input_data - self.gda_mu1  # (m, n)
            self.gda_covMat += np.eye(self.gda_covMat.shape[0]) * 1e-5
            sigma_inv = np.linalg.inv(self.gda_covMat)  # (n, n)

            term0 = np.sum(np.dot(diff0, sigma_inv) * diff0, axis=1)
            term1 = np.sum(np.dot(diff1, sigma_inv) * diff1, axis=1)

            decision = term1 - term0 + 2 * np.log(self.gda_phi / (1 - self.gda_phi))
            
            predictions = (decision < 0).astype(int)
            return predictions

        else:
            # --- Normal regression prediction ---
            if input_data.shape[1] + 1 == self.params.shape[0]:
                input_data = np.insert(input_data, 0, 1, axis=1)  # add bias term
            
            return np.dot(input_data, self.params)

    

        



class LinearRegressor:

    '''
    Linear Regressor
    ===============

    When to use:
    ------------
    Prediction of a continuous value
    
    Prerequisites:
    -------------
    - Model assumes all columns in the dataset have numerical values
    
    How to use:
    -----------
    1. Instantiate model object with the dataset (as Pandas DataFrame)
    2. Remove unnecessary columns using ``removeColumns()`` with a list of column names
    3. Standardize data using ``Standardize_data_Zscore()`` with a list of column names
       (Do not include columns with binary values)
    4. Prepare data with ``PrepareData()`` specifying the label column name
    5. Train the model with ``Train()``
       - *alpha*: value in range [0.0, 1.0] (determines learning rate)
       - Training methods: ``"normal_equation"`` (instant but computationally expensive) 
         or ``"gradient_descent"`` (slower but computationally inexpensive)
    6. Evaluate with ``Test()`` to check accuracy
    7. Use ``Predict()`` to make predictions on new data
    '''


    def __init__(self, dataset) -> None:
        self.params = np.zeros((1, 1))
        self.dataset = dataset
        self.partitioned_data = []
    
    # ===============================================================

    def getParams(self):
        return self.params

    def getDataset(self):
        return self.dataset
    
    def setDataset(self, dataset):
        self.dataset = dataset

    def PrepareData(self, label_column_name, split_ratio_training=0.8):
        self.AddBiasColumn()
        X = self.dataset.drop(columns=[label_column_name]).values
        Y = self.dataset[label_column_name].values.reshape(X.shape[0], 1)
        np.random.seed()
        indices = np.random.permutation(X.shape[0])
        train_size = int(split_ratio_training * X.shape[0])
        X_train, X_test = X[indices[:train_size]], X[indices[train_size:]]
        Y_train, Y_test = Y[indices[:train_size]], Y[indices[train_size:]]

        self.partitioned_data = [X_train, X_test, Y_train, Y_test]
        self.params = np.zeros((X.shape[1], 1))

    def Standardize_data_Zscore(self, feature_list):
        if not hasattr(self, 'standardization_params'):
            self.standardization_params = {}
        
        for feature in feature_list:
            mean = np.mean(self.dataset[feature])
            std_dev = np.std(self.dataset[feature])
            self.dataset.loc[:, feature] = (self.dataset[feature] - mean) / std_dev
            self.standardization_params[feature] = (mean, std_dev) 

    def AddBiasColumn(self):
        self.dataset.insert(0, 'bias', 1)
    
    def removeColumns(self, column_names):
        self.dataset = self.dataset.drop(columns=column_names)

    # ===============================================================
    
    def compute_cost(self, X, Y, theta):
        m = X.shape[0]
        predictions = np.dot(X, theta)
        errors = predictions - Y
        return (1 / (2 * m)) * np.sum(errors ** 2)
    
    def gradient_descent(self, X, Y, theta, alpha, convergence_threshold=1e-5):
        m = X.shape[0]
        prev_cost = self.compute_cost(X, Y, theta) + 2 * convergence_threshold
        cost = self.compute_cost(X, Y, theta)
        
        while abs(prev_cost - cost) > convergence_threshold:
            prev_cost = cost
            gradient = (1 / m) * np.dot(X.T, np.dot(X, theta) - Y)
            theta = theta - alpha * gradient
            cost = self.compute_cost(X, Y, theta)
            # print(f"Cost: {cost}")
        
        return theta
    
    def normal_equation(self, X, Y):
        return np.dot(np.linalg.pinv(np.dot(X.T, X)), np.dot(X.T, Y))
    
    def Train(self, alpha=0.01, training_method="gradient_descent", convergence_threshold=1e-5):
        X = self.partitioned_data[0]
        Y = self.partitioned_data[2]
        
        if training_method == "gradient_descent":
            self.params = self.gradient_descent(X, Y, self.params, alpha, convergence_threshold)
        elif training_method == "normal_equation":
            self.params = self.normal_equation(X, Y)
        
    def Test(self):
        X = self.partitioned_data[1]
        Y = self.partitioned_data[3]
        predictions = np.dot(X, self.params)
        mse = np.mean((predictions - Y) ** 2)
        mae = np.mean(np.abs(predictions - Y))
        r2 = 1 - (np.sum((Y - predictions) ** 2) / np.sum((Y - np.mean(Y)) ** 2))
        print(f"Mean Squared Error on Test Data: {mse}")
        print(f"Mean Absolute Error on Test Data: {mae}")
        print(f"R² Score (Accuracy): {r2 * 100:.2f}%")


    def Predict(self, input_data):
        if isinstance(input_data, dict):
            if hasattr(self, 'standardization_params'):
                filtered_data = {k: v for k, v in input_data.items() if k in self.standardization_params}
                standardized_values = [
                    (filtered_data[k] - self.standardization_params[k][0]) / self.standardization_params[k][1]
                    if k in self.standardization_params else input_data[k]
                    for k in input_data.keys()
                ]
            else:
                standardized_values = [input_data[k] for k in input_data.keys()]
            
            input_data = np.array(standardized_values).reshape(1, -1)
        
        if input_data.shape[1] + 1 == self.params.shape[0]:
            input_data = np.insert(input_data, 0, 1, axis=1)  
        
        return np.dot(input_data, self.params)
  

class NaiveBayesClassifier:

    '''
    Naive Bayes Classifier
    =====================
    
    When to use:
    ------------
    Multiclass/Binary Classification with binary features
    
    Prerequisites:
    -------------
    - Model assumes all input columns (except the label column) have binary values [0/1]
    
    Note:
    -----
    - Label column can be either integers or strings - the model handles both automatically
    
    How to use:
    -----------
    1. Instantiate model object
    2. Call ``split_and_encode_data()`` with X (all columns except label) and y (label column)
    3. Call ``fit()`` to train the model
    4. Evaluate with ``test()`` to check accuracy
    5. Use ``predict()`` to make predictions on new data
       - Set *return_classname=True* to return the actual class names instead of indices
    '''

    def __init__(self) -> None:
        self.splitdata = []
        self.xdataset = None
        self.ydataset = None
        self.labelnames = None

    def fit(self, X=None, y=None):
        if X is None:
            X = self.splitdata[0]
        if y is None:
            y = self.splitdata[2]
        self.classes = np.unique(y)
        self.n_classes = len(self.classes)
        
        # Calculate prior probabilities P(y)
        self.priors = {c: np.sum(y == c) / len(y) for c in self.classes}

        # Calculate likelihoods P(x|y) using Laplace smoothing
        self.likelihoods = {}
        for c in self.classes:
            X_c = X[y == c]
            self.likelihoods[c] = (np.sum(X_c, axis=0) + 1) / (X_c.shape[0] + 2)  # Laplace smoothing

    def predict(self, X,  return_classname = False):
        y_pred = []
        for x in X: # X is a 2D numpy array with multiple examples (probably)
            posteriors = {}
            for c in self.classes:
                # Compute log P(y) + sum(log P(x|y))
                log_prior = np.log(self.priors[c])
                log_likelihood = np.sum(x * np.log(self.likelihoods[c]) + (1 - x) * np.log(1 - self.likelihoods[c]))
                posteriors[c] = log_prior + log_likelihood
            y_pred.append(max(posteriors, key=posteriors.get))  # Choose class with max posterior
        if return_classname:
            return [self.labelnames[i] for i in np.array(y_pred)]
        return np.array(y_pred)
    
    def test(self, X_test=None, y_test=None):
        if X_test is None:
            X_test = self.splitdata[0]
        if y_test is None:
            y_test = self.splitdata[2]
        y_pred = self.predict(X_test)
        accuracy = np.mean(y_pred == y_test) * 100
        return accuracy

    def split_and_encode_data(self, X, y, test_size=0.2, random_state=None): 
        self.xdataset = X
        self.ydataset = y
        
        # Skip encoding if y is already integers
        if isinstance(y[0], str):  # Only encode if y is strings
            unique_labels = np.unique(y)
            label_mapping = {label: index for index, label in enumerate(unique_labels)}
            y = np.array([label_mapping[label] for label in y])
        
        # Custom train-test split
        np.random.seed(random_state)
        indices = np.arange(X.shape[0])
        np.random.shuffle(indices)
        split_index = int(X.shape[0] * (1 - test_size))
        
        X_train, X_test = X[indices[:split_index]], X[indices[split_index:]]
        y_train, y_test = y[indices[:split_index]], y[indices[split_index:]]

        self.splitdata = [X_train, X_test, y_train, y_test]




class SoftmaxRegressor:

    '''
    Softmax Regressor
    ================
    
    When to use:
    ------------
    Multiclass Classification with continuous features
    
    Prerequisites:
    -------------
    - Model assumes all columns have numerical values
    - Model assumes the label/output column contains class names as strings
    
    How to use:
    -----------
    1. Instantiate model object with the dataset (as Pandas DataFrame)
    2. Remove unnecessary columns using ``removeColumns()`` with a list of column names
    3. Standardize data using ``Standardize_data_Zscore()`` with a list of column names
       (Do not include columns with binary values)
    4. Prepare data with ``PrepareData()`` specifying the label column name
    5. Train the model with ``Train()``
       - *alpha*: value in range [0.0, 1.0] (determines learning rate)
       - *batch_size*: size of training batches (default: 32)
       - *verbose*: set to True to see training progress
    6. Evaluate with ``Test()`` to check accuracy
    7. Use ``Predict()`` to make predictions on new data
    '''



    def __init__(self, dataset) -> None:
        self.dataset = dataset
        self.params = None
        self.partitioned_data = []

    def Standardize_data_Zscore(self, feature_list):
        if not hasattr(self, 'standardization_params'):
            self.standardization_params = {}
        
        for feature in feature_list:
            mean = np.mean(self.dataset[feature])
            std_dev = np.std(self.dataset[feature])
            self.dataset.loc[:, feature] = (self.dataset[feature] - mean) / std_dev
            self.standardization_params[feature] = (mean, std_dev)


    def PrepareData(self, label_column_name, split_ratio_training=0.8):
        self.AddBiasColumn()
        X = self.dataset.drop(columns=[label_column_name]).values
        Y = self.dataset[label_column_name].values.reshape(X.shape[0], 1)

        unique_labels = np.unique(Y)
        self.label_mapping = {label: index for index, label in enumerate(unique_labels)}
        
        # Ensure Y is a 1D array for proper iteration
        Y = Y.flatten()  # Convert to 1D array

        # Create one-hot encoding
        y_onehot = np.array([[1 if i == self.label_mapping[label] else 0 for i in range(len(unique_labels))] for label in Y])

        np.random.seed()
        indices = np.random.permutation(X.shape[0])
        train_size = int(split_ratio_training * X.shape[0])
        X_train, X_test = X[indices[:train_size]], X[indices[train_size:]]
        Y_train, Y_test = y_onehot[indices[:train_size]], y_onehot[indices[train_size:]]

        self.params = np.array([np.zeros((self.dataset.shape[1] - 1))] * len(unique_labels))
        self.partitioned_data = [X_train, X_test, Y_train, Y_test]

    def removeColumns(self, column_names):
        self.dataset = self.dataset.drop(columns=column_names)

    def AddBiasColumn(self):
        self.dataset.insert(0, 'bias', 1)    

    def Cost(self, X, y):

        logits = np.dot(X, self.params.T)  # (m, K)

        # Apply numerical stability trick
        logits -= np.max(logits, axis=1, keepdims=True)

        exp_logits = np.exp(logits)
        P = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)  # (m, K)

        m = X.shape[0]  
        epsilon = 1e-15
        log_probs = -np.sum(y * np.log(P + epsilon), axis=1) 
        cost = np.mean(log_probs)  
        
        return cost
        
    def Softmax(self, X):

        logits = np.dot(X, self.params.T)  # (m, K)

        # Apply numerical stability trick
        logits -= np.max(logits, axis=1, keepdims=True)

        exp_logits = np.exp(logits)  # (m, K)

        P = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)  # (m, K)

        return P


    def Gradient_Descent(self,X, y, alpha=1):
        self.params = self.params - alpha*np.dot((self.Softmax(X)- y).T,X)
        

    def Train(self, alpha=1, batch_size=32, verbose=False, convergence_threshold=1e-5):
        """
        Trains the softmax regression model using mini-batch gradient descent until convergence.
        
        Parameters:
        X - (m, n) Feature matrix
        y - (m, K) One-hot encoded labels
        alpha - Learning rate (default: 1)
        batch_size - Size of mini-batches (default: 32)
        verbose - Whether to print training progress (default: True)
        convergence_threshold - Threshold for convergence check (default: 1e-5)
        """
        X = self.partitioned_data[0]
        y = self.partitioned_data[2]


        m = X.shape[0]
        iterations = m // batch_size
        
        prev_cost = float('inf')
        epoch = 0
        
        while True:
            # Shuffle data
            shuffle_idx = np.random.permutation(m)
            X_shuffled = X[shuffle_idx]
            y_shuffled = y[shuffle_idx]
            
            epoch_cost = 0
            
            # Mini-batch training
            for i in range(iterations):
                start_idx = i * batch_size
                end_idx = start_idx + batch_size
                
                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]
                
                batch_cost = self.Cost(X_batch, y_batch)
                epoch_cost += batch_cost
                
                self.Gradient_Descent(X_batch, y_batch, alpha)
            
            epoch_cost /= iterations
            
            # Check for convergence
            if abs(epoch_cost - prev_cost) < convergence_threshold:
                if verbose:
                    print(f"Converged after {epoch + 1} epochs with cost: {epoch_cost:.4f}")
                break
                
            prev_cost = epoch_cost
            epoch += 1
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}, Cost: {epoch_cost:.4f}")


    def Test(self):

        X_test = self.partitioned_data[1]  
        y_test = self.partitioned_data[3]  
        
        probabilities = self.Softmax(X_test)
        
        y_pred_indices = np.argmax(probabilities, axis=1)  
        num_classes = probabilities.shape[1]  

        # Create a one-hot encoded vector
        y_pred_one_hot = np.zeros((y_pred_indices.size, num_classes)) 
        y_pred_one_hot[np.arange(y_pred_indices.size), y_pred_indices] = 1 
        
        accuracy = np.mean(y_pred_one_hot == y_test)

        print(f"\nTest Accuracy: {accuracy*100:.4f}%")

    def Predict(self, input_data, return_probabilities=False):
        # Handle case where input_data is a list containing an array
        if isinstance(input_data, list) and len(input_data) > 0:
            if hasattr(input_data[0], 'shape'):  # Check if the first element is an array-like object
                input_data = input_data[0]
        
        if isinstance(input_data, dict):
            if hasattr(self, 'standardization_params'):
                filtered_data = {k: v for k, v in input_data.items() if k in self.standardization_params}
                standardized_values = [
                    (filtered_data[k] - self.standardization_params[k][0]) / self.standardization_params[k][1]
                    if k in self.standardization_params else input_data[k]
                    for k in input_data.keys()
                ]
            else:
                standardized_values = [input_data[k] for k in input_data.keys()]
            
            input_data = np.array(standardized_values).reshape(1, -1)
        
        # Ensure input_data is a numpy array
        input_data = np.array(input_data)
        
        # Add a dimension if input is 1D
        if len(input_data.shape) == 1:
            input_data = input_data.reshape(1, -1)
        
        if input_data.shape[1] + 1 == self.params.shape[1]:
            input_data = np.insert(input_data, 0, 1, axis=1)  # Add bias term
        
        probabilities = self.Softmax(input_data)
        
        if return_probabilities:
            return probabilities
        else:
            predicted_indices = np.argmax(probabilities, axis=1)
            if hasattr(self, 'label_mapping') and self.label_mapping:
                reverse_mapping = {v: k for k, v in self.label_mapping.items()}
                return [reverse_mapping[idx] for idx in predicted_indices]
            else:
                return predicted_indices  # Return class indices if no mapping exists

import pandas as pd # Added for type hinting and potential DataFrame input

class NeuralNetwork:
    def __init__(self, dataset, neuron_map, learning_rate=0.01,
                 activation_function="sigmoid",
                 output_activation_function="linear", # Added for output layer
                 cost_function="mse",
                 absolute_gradient_clipping = 1.0,
                 regularization_parameter = 0) -> None:
        self.dataset = dataset.copy() # Work on a copy to avoid modifying original DataFrame passed
        self.partitioned_data = []

        self.learning_rate = learning_rate
        self.absolute_gradient_clipping = absolute_gradient_clipping

        self.regularization_parameter = regularization_parameter

        # Activation for hidden layers
        self.hidden_activation_type = activation_function
        # Activation for output layer (crucial for regression)
        self.output_activation_type = output_activation_function

        self.neuron_map = neuron_map
        if len(neuron_map) < 2:
            raise ValueError("neuron_map must have at least an input and an output layer.")
        # Derive number of hidden layers from neuron_map
        self.num_hidden_layers = len(neuron_map) - 2
        if self.num_hidden_layers < 0: # Should be caught by len < 2, but for clarity
            self.num_hidden_layers = 0


        self.cost_function_type = cost_function
        # if self.cost_function_type == "cross_entropy":
        #     print("Warning: Cross-entropy is typically for classification. For regression, 'mse' is recommended.")


        self.hidden_layer_biases = []
        self.hidden_layer_weights = []

        # Initialize weights and biases for hidden layers
        # neuron_map[0] is input layer size
        # neuron_map[1:-1] are hidden layer sizes
        # neuron_map[-1] is output layer size
        current_input_size = neuron_map[0]
        for i in range(self.num_hidden_layers):
            num_neurons_in_layer = neuron_map[i+1]
            if self.hidden_activation_type == "relu":
                # He initialization for ReLU
                self.hidden_layer_weights.append(np.random.randn(num_neurons_in_layer, current_input_size) * np.sqrt(2.0/current_input_size))
            else:
                # Xavier/Glorot initialization for other activations
                self.hidden_layer_weights.append(np.random.randn(num_neurons_in_layer, current_input_size) * 0.01)
            self.hidden_layer_biases.append(np.zeros((num_neurons_in_layer, 1))) # Initialize biases to zero
            current_input_size = num_neurons_in_layer

        # Initialize weights and biases for the output layer
        # Input to output layer is the size of the last hidden layer, or input layer if no hidden layers
        size_prev_layer_to_output = neuron_map[-2] # neuron_map[self.num_hidden_layers]
        if self.hidden_activation_type == "relu":
            # He initialization for output layer if using ReLU
            self.output_neurons_weights = np.random.randn(neuron_map[-1], size_prev_layer_to_output) * np.sqrt(2.0/size_prev_layer_to_output)
        else:
            # Xavier/Glorot initialization for other activations
            self.output_neurons_weights = np.random.randn(neuron_map[-1], size_prev_layer_to_output) * 0.01
        self.output_neurons_biases = np.zeros((neuron_map[-1], 1))

        # To store standardization parameters
        self.standardization_params = {}
        self.target_standardization_params = {}
        self.ordered_target_label_names = [] # To keep track of order for unstandardization

    def Standardize_data_Zscore(self, feature_list, label_column_names=None):
        if not hasattr(self, 'standardization_params'): # Should always exist from __init__
            self.standardization_params = {}
        if not hasattr(self, 'target_standardization_params'): # Should always exist from __init__
            self.target_standardization_params = {}

        # Standardize features
        for feature in feature_list:
            mean = np.mean(self.dataset[feature])
            std_dev = np.std(self.dataset[feature])
            self.standardization_params[feature] = (mean, std_dev)
            if std_dev > 1e-8: # Avoid division by zero
                self.dataset.loc[:, feature] = (self.dataset[feature] - mean) / std_dev
            else:
                self.dataset.loc[:, feature] = 0.0 # If no variance, set to 0 (mean-centered)

        # Standardize target/label columns if provided
        if label_column_names is not None:
            if isinstance(label_column_names, str):
                label_column_names = [label_column_names]
            self.ordered_target_label_names = label_column_names # Store order

            for label_col in label_column_names:
                mean = np.mean(self.dataset[label_col])
                std_dev = np.std(self.dataset[label_col])
                self.target_standardization_params[label_col] = (mean, std_dev)
                if std_dev > 1e-8: # Avoid division by zero
                    self.dataset.loc[:, label_col] = (self.dataset[label_col] - mean) / std_dev
                else:
                    self.dataset.loc[:, label_col] = 0.0

    def PrepareData(self, label_column_names, split_ratio_training=0.8, random_state=None, Architecture_test = False, A_test_samples = 10):
        if isinstance(label_column_names, str):
            label_column_names = [label_column_names]

        X = self.dataset.drop(columns=label_column_names).values
        Y = self.dataset[label_column_names].values

        if random_state is not None:
            np.random.seed(random_state) # For reproducible splits

        if not Architecture_test:    
            indices = np.random.permutation(X.shape[0])
            train_size = int(split_ratio_training * X.shape[0])

            X_train, X_test = X[indices[:train_size]], X[indices[train_size:]]
            Y_train, Y_test = Y[indices[:train_size]], Y[indices[train_size:]]

            self.partitioned_data = [X_train, X_test, Y_train, Y_test]
        else:
            indices = [i for i in range(A_test_samples)]

            X_train, X_test = X[indices], X[indices]
            Y_train, Y_test = Y[indices], Y[indices]

            self.partitioned_data = [X_train, X_test, Y_train, Y_test]  

    def removeColumns(self, column_names):
        self.dataset = self.dataset.drop(columns=column_names)

    def _apply_activation(self, z, activation_type):
        if activation_type == "sigmoid":
            return 1 / (1 + np.exp(-z))
        elif activation_type == "tanh":
            return np.tanh(z)
        elif activation_type == "relu":
            return np.maximum(0, z)
        elif activation_type == "linear":
            return z
        elif activation_type == "softmax":
            # Subtract max for numerical stability
            exp_z = np.exp(z - np.max(z, axis=0, keepdims=True))
            return exp_z / np.sum(exp_z, axis=0, keepdims=True)
        else:
            raise ValueError(f"Unknown activation function: {activation_type}")

    def _apply_activation_derivative(self, z, activation_type):
        a = self._apply_activation(z, activation_type) # Get activation for sigmoid/tanh
        if activation_type == "sigmoid":
            return a * (1 - a)
        elif activation_type == "tanh":
            return 1 - a**2 # 1 - np.tanh(z)**2
        elif activation_type == "relu":
            return np.where(z > 0, 1, 0)
        elif activation_type == "linear":
            return np.ones_like(z)
        elif activation_type == "softmax":
            # For softmax, we only need the diagonal elements of the Jacobian
            # which is a_i * (1 - a_i) for each output
            a = self._apply_activation(z, activation_type)
            return a * (1 - a)  # This will return shape (output_size, batch_size)
        else:
            raise ValueError(f"Unknown activation function for derivative: {activation_type}")

    def forward_propagation(self, X_batch):
        # X_batch is a numpy array of shape (m, n), where m is samples, n is features
        self.z_values = []
        self.a_values = []

        current_a = X_batch.T # Shape (features, samples)

        # print(X_batch.shape)
        # print(X_batch)

        # Hidden layers
        for i in range(self.num_hidden_layers):
            z = np.dot(self.hidden_layer_weights[i], current_a) + self.hidden_layer_biases[i]
            current_a = self._apply_activation(z, self.hidden_activation_type)
            self.z_values.append(z)
            self.a_values.append(current_a)

        # Output layer
        # If no hidden layers, current_a is still X_batch.T
        # If hidden layers, current_a is the activation of the last hidden layer
        z_output = np.dot(self.output_neurons_weights, current_a) + self.output_neurons_biases
        a_output = self._apply_activation(z_output, self.output_activation_type)
        self.z_values.append(z_output)
        self.a_values.append(a_output)

        # a_output shape: (num_output_neurons, num_samples)
        return a_output

    def _calculate_cost(self, Y_pred, Y_true):
        # Y_pred, Y_true shape: (num_output_neurons, num_samples)
        # For MSE: Y_pred and Y_true can be any real numbers
        # For categorical_cross_entropy:
        #   - Y_pred should be probabilities (values between 0 and 1)
        #   - Y_true should be one-hot encoded (each column has exactly one 1 and rest 0s)
        #   - Each column represents one sample, each row represents one class
        m = Y_true.shape[1]
        # L2 regularization: sum of squares of all weights (hidden and output)
        l2_sum = 0.0
        for w in self.hidden_layer_weights:
            l2_sum += np.sum(np.square(w))
        l2_sum += np.sum(np.square(self.output_neurons_weights))
        l2_term = (self.regularization_parameter / (2 * m)) * l2_sum

        if self.cost_function_type == "mse":
            cost = (1/m) * np.sum((Y_pred - Y_true)**2)
        elif self.cost_function_type == "categorical_cross_entropy":
            # Add small epsilon to avoid log(0)
            epsilon = 1e-15
            Y_pred = np.clip(Y_pred, epsilon, 1 - epsilon)
            # Returns a scalar value representing the average cross-entropy loss across all samples
            cost = -(1/m) * np.sum(Y_true * np.log(Y_pred))
        else:
            raise ValueError(f"Unknown cost function: {self.cost_function_type}")

        return cost + l2_term , cost

    def _calculate_cost_derivative(self, Y_pred, Y_true):
        # Y_pred, Y_true shape: (num_output_neurons, num_samples)
        # For MSE: Returns gradient of same shape as inputs
        # For categorical_cross_entropy: Returns gradient of same shape as inputs
        #   - Each element represents how much the cost changes with respect to that prediction
        m = Y_true.shape[1]
        if self.cost_function_type == "mse":
            return (2/m) * (Y_pred - Y_true)
            # If cost was 1/(2m) * sum(...), derivative is (1/m) * (Y_pred - Y_true)
            # If cost was np.mean(...), derivative needs careful shape handling. Sticking to (2/m) for now.
        elif self.cost_function_type == "categorical_cross_entropy":
            # Add small epsilon to avoid division by zero
            epsilon = 1e-15
            Y_pred = np.clip(Y_pred, epsilon, 1 - epsilon)
            # Returns gradient matrix of same shape as inputs
            # Each element (i,j) represents ∂L/∂Y_pred[i,j]
            return -(1/m) * (Y_true / Y_pred)
        else:
            raise ValueError(f"Unknown cost function for derivative: {self.cost_function_type}")

    def back_propagation(self, X_batch, Y_batch_true):
        # X_batch shape: (num_samples, num_features)
        # Y_batch_true shape: (num_output_neurons, num_samples)
        m = Y_batch_true.shape[1] # Number of samples in the batch

        # --- Output Layer ---
        # a_values[-1] is activation of output layer, z_values[-1] is z of output layer
        
        # Special case for softmax activation with categorical cross-entropy loss
        if self.output_activation_type == "softmax" and self.cost_function_type == "categorical_cross_entropy":
            # The combined derivative of loss w.r.t. z_output simplifies to (Y_pred - Y_true)
            dZ_output = self.a_values[-1] - Y_batch_true
        else:
            # Regular logic using the chain rule for all other cases
            # Derivative of cost w.r.t. activation of output layer
            dA_output = self._calculate_cost_derivative(self.a_values[-1], Y_batch_true)
            # Derivative of output activation function
            d_activation_output = self._apply_activation_derivative(self.z_values[-1], self.output_activation_type)
            dZ_output = dA_output * d_activation_output

        # Gradients for output layer weights and biases
        # Activation of the layer before output (last hidden layer, or input if no hidden layers)
        a_prev_to_output = self.a_values[-2] if self.num_hidden_layers > 0 else X_batch.T

        dW_output = (1/m) * np.dot(dZ_output, a_prev_to_output.T)
        db_output = (1/m) * np.sum(dZ_output, axis=1, keepdims=True)

        # --- Propagate to Hidden Layers (if any) ---
        dZ_next_layer = dZ_output
        weights_next_layer = self.output_neurons_weights

        # Loop backwards from the last hidden layer to the first.
        for i in range(self.num_hidden_layers - 1, -1, -1):
            # dA_hidden is dCost/dA[i]
            dA_hidden = np.dot(weights_next_layer.T, dZ_next_layer)
            d_activation_hidden = self._apply_activation_derivative(self.z_values[i], self.hidden_activation_type)
            dZ_hidden = dA_hidden * d_activation_hidden

            # Activation of the layer before current hidden layer
            # If i=0 (first hidden layer), prev_activation is input X_batch.T
            # Otherwise, it's a_values[i-1]
            prev_activation = X_batch.T if i == 0 else self.a_values[i-1]

            dW_hidden = (1/m) * np.dot(dZ_hidden, prev_activation.T)
            db_hidden = (1/m) * np.sum(dZ_hidden, axis=1, keepdims=True)

            # Update for next iteration (going backwards)
            dZ_next_layer = dZ_hidden
            weights_next_layer = self.hidden_layer_weights[i]

            # Clip gradients to prevent exploding gradients
            # Apply L2 regularization to gradients (do not regularize biases)
            if self.regularization_parameter is not None and self.regularization_parameter > 0:
                dW_hidden += (self.regularization_parameter / m) * self.hidden_layer_weights[i]

            if self.absolute_gradient_clipping is not None:
                dW_hidden = np.clip(dW_hidden, -self.absolute_gradient_clipping, self.absolute_gradient_clipping)
                db_hidden = np.clip(db_hidden, -self.absolute_gradient_clipping, self.absolute_gradient_clipping)
            
            # Update weights and biases for the current hidden layer
            self.hidden_layer_weights[i] -= self.learning_rate * dW_hidden
            self.hidden_layer_biases[i] -= self.learning_rate * db_hidden

        # Apply L2 regularization to output layer gradients (do not regularize biases)
        if self.regularization_parameter is not None and self.regularization_parameter > 0:
            dW_output += (self.regularization_parameter / m) * self.output_neurons_weights

        # Clip output layer gradients
        if self.absolute_gradient_clipping is not None:
            dW_output = np.clip(dW_output, -self.absolute_gradient_clipping, self.absolute_gradient_clipping)
            db_output = np.clip(db_output, -self.absolute_gradient_clipping, self.absolute_gradient_clipping)
        
        # Update Output Layer Parameters
        self.output_neurons_weights -= self.learning_rate * dW_output
        self.output_neurons_biases -= self.learning_rate * db_output

    def train(self, epochs=1000, batch_size=32, verbose=True, print_every=100):
        """
        Trains the neural network.

        Parameters:
            epochs (int): Number of epochs to train.
            batch_size (int): Batch size for mini-batch gradient descent.
            verbose (bool): Whether to print progress.
            print_every (int): Print cost every this many epochs.
            use_lr_decay (bool): Whether to use learning rate decay.
            decay_rate (float): The decay rate (lambda for time-based, gamma for step-based).
            decay_type (str): "time_based" or "step_based".
            min_lr (float): Minimum learning rate allowed.
        """
        if not self.partitioned_data:
            raise ValueError("Data not prepared. Call PrepareData() first.")

        X_train, _, Y_train, _ = self.partitioned_data # Y_train shape: (num_train_samples, num_labels)
        num_samples = X_train.shape[0]
        costs = []
        pure_costs = []
        test_costs = []

        for epoch in range(epochs):
            epoch_cost = 0
            pure_epoch_costs = 0
            num_batches = 0

            indices = np.random.permutation(num_samples)
            X_shuffled = X_train[indices]
            Y_shuffled = Y_train[indices]

            for i in range(0, num_samples, batch_size):
                end_idx = min(i + batch_size, num_samples)
                X_batch = X_shuffled[i:end_idx]
                Y_batch_labels = Y_shuffled[i:end_idx] # Shape (batch_s, num_labels)

                # Forward propagation
                # Y_pred shape: (num_output_neurons, current_batch_size)
                Y_pred_batch = self.forward_propagation(X_batch)

                # Y_batch_true must be (num_output_neurons, current_batch_size)
                Y_batch_true = Y_batch_labels.T

                batch_cost, pure_batch_cost = self._calculate_cost(Y_pred_batch, Y_batch_true)
                epoch_cost += batch_cost
                pure_epoch_costs += pure_batch_cost
                num_batches += 1

                self.back_propagation(X_batch, Y_batch_true)

            avg_epoch_cost = epoch_cost / num_batches
            avg_pure_epoch_cost = pure_epoch_costs / num_batches
            costs.append(avg_epoch_cost)
            pure_costs.append(avg_pure_epoch_cost)

            metrics = self.test(verbose=False)

            test_costs.append(metrics[f"test_{self.cost_function_type}"]) 

            if verbose and (epoch % print_every == 0 or epoch == epochs - 1):
                print(f"Epoch {epoch + 1}/{epochs}, Cost: {avg_epoch_cost:.6f}, Learning Rate: {self.learning_rate:.8f}")

        return (costs,test_costs,pure_costs,
                    [self.neuron_map, 
                    self.hidden_activation_type, 
                    self.output_activation_type, 
                        [self.hidden_layer_weights,
                        self.hidden_layer_biases,
                        self.output_neurons_weights,
                        self.output_neurons_biases]
                        ]
                        )

    def test(self, verbose=True):
        if not self.partitioned_data:
            raise ValueError("Data not prepared. Call PrepareData() first.")

        _, X_test, _, Y_test_labels = self.partitioned_data # Y_test_labels shape (num_test_samples, num_labels)

        # Y_pred_test shape: (num_output_neurons, num_test_samples)
        Y_pred_test = self.forward_propagation(X_test)
        Y_true_test = Y_test_labels.T # Shape (num_output_neurons, num_test_samples)

        # Calculate cost based on cost function type
        if self.cost_function_type == "mse":
            _ , test_cost = self._calculate_cost(Y_pred_test, Y_true_test) # MSE
            cost_name = "MSE"
            accuracy = None  # MSE doesn't have a percentage accuracy
        elif self.cost_function_type == "categorical_cross_entropy":
            _ , test_cost = self._calculate_cost(Y_pred_test, Y_true_test) # Cross Entropy
            cost_name = "Cross Entropy"
            # Calculate accuracy for classification
            predicted_classes = np.argmax(Y_pred_test, axis=0)
            true_classes = np.argmax(Y_true_test, axis=0)
            accuracy = np.mean(predicted_classes == true_classes) * 100  # Convert to percentage
        else:
            raise ValueError(f"Unknown cost function type: {self.cost_function_type}")

        # Calculate MAE
        mae = np.mean(np.abs(Y_pred_test - Y_true_test))

        metrics = {
            f'test_{self.cost_function_type}': test_cost,
            'test_mae': mae
        }
        if accuracy is not None:
            metrics['test_accuracy'] = accuracy

        if verbose:
            print(f"\nTest Results:")
            print(f"  Test {cost_name}: {test_cost:.6f}")
            print(f"  Test MAE: {mae:.6f}")
            if accuracy is not None:
                print(f"  Test Accuracy: {accuracy:.2f}%")

        return metrics

    def predict(self, input_data, feature_columns_ordered=None):
        # input_data: np.ndarray or pd.DataFrame
        # feature_columns_ordered: list of feature column names in the order expected by the model
        # (must match order used during training if standardization was applied)

        if isinstance(input_data, pd.DataFrame):
            if feature_columns_ordered is None:
                raise ValueError("feature_columns_ordered must be provided if input_data is a DataFrame.")
            try:
                X_predict = input_data[feature_columns_ordered].values
            except KeyError as e:
                raise ValueError(f"One or more feature columns not found in input_data: {e}")
        elif isinstance(input_data, np.ndarray):
            X_predict = input_data.copy()
        else:
            raise TypeError("input_data must be a NumPy array or Pandas DataFrame.")

        if X_predict.ndim == 1: # Single sample
            X_predict = X_predict.reshape(1, -1)

        # Apply feature standardization if parameters exist
        if self.standardization_params:
            if feature_columns_ordered is None and isinstance(input_data, np.ndarray):
                 # If numpy array and no feature_columns, we assume columns are already in correct order
                 # and number of columns matches the number of features standardized.
                 if X_predict.shape[1] != len(self.standardization_params):
                     print("Warning: Number of features in input data doesn't match number of standardized features. Skipping standardization.")
                 else: # This case is tricky without explicit column names. Best to provide them.
                      # For simplicity, assume that if self.standardization_params has N entries,
                      # they correspond to the first N columns of X_predict. This is fragile.
                      # A more robust approach would be to store ordered feature names during standardization too.
                      # For now, we proceed if feature_columns_ordered is given (handled below) or assume order.
                      pass


            X_standardized = X_predict.astype(float) # Ensure float for division
            for i, feature_name in enumerate(feature_columns_ordered if feature_columns_ordered else self.standardization_params.keys()):
                if feature_name in self.standardization_params:
                    mean, std_dev = self.standardization_params[feature_name]
                    if std_dev > 1e-8:
                        X_standardized[:, i] = (X_predict[:, i] - mean) / std_dev
                    else:
                        X_standardized[:, i] = 0.0 # (X_predict[:, i] - mean) would be 0
                # else: feature not standardized during training, use as is or raise error
            X_to_prop = X_standardized
        else:
            X_to_prop = X_predict

        # Forward propagation
        raw_predictions = self.forward_propagation(X_to_prop) # Shape (num_output_neurons, num_samples)
        predictions_transposed = raw_predictions.T # Shape (num_samples, num_output_neurons)

        # Unstandardize predictions if target variables were standardized
        if self.target_standardization_params and self.ordered_target_label_names:
            unstandardized_predictions = predictions_transposed.copy()
            for i, label_name in enumerate(self.ordered_target_label_names):
                if label_name in self.target_standardization_params:
                    mean, std_dev = self.target_standardization_params[label_name]
                    if std_dev > 1e-8:
                        unstandardized_predictions[:, i] = (predictions_transposed[:, i] * std_dev) + mean
                    else: # If std_dev was 0, original values were all 'mean'. Standardized was 0.
                          # So unstandardized is 'mean'.
                        unstandardized_predictions[:, i] = mean
            return unstandardized_predictions
        else:
            return predictions_transposed
    
