import numpy as np
import nnfs
from nnfs.datasets import sine_data, spiral_data
from zipfile import ZipFile
import matplotlib.pyplot as plt
import os
import urllib
import urllib.request
import ssl
import cv2

nnfs.init()

        
class Layer_Input:
    
    #Forward pass
    def forward(self, inputs, training):
        self.output = inputs
        

class Accuracy:
    
    #Calculate the accuracy given the predictions and targets
    def calculate(self, predictions, y):
        
        #Get comparison results
        comparisons = self.compare(predictions, y)
        
        #calculate the accuraryc
        accuracy = np.mean(comparisons)
        
        #Add accumulated sum of matching balues and sample count for each batch
        self.accumulated_sum += np.sum(comparisons)
        self.accumulated_count += len(comparisons)
        
        #Return accuracy
        return accuracy
    
    
    #Calculates accumulated accuracy
    def calculate_accumulated(self):
        
        #Calculate an accuracy for each batch
        accuracy = self.accumulated_sum / self.accumulated_count
        
        #Return accuracy
        return accuracy
    
    
    #Reset variables for accumulated accuracy
    def new_pass(self):
        
        self.accumulated_sum = 0
        self.accumulated_count = 0

class Accuracy_Regression(Accuracy):
    
    def __init__(self):
        self.precision = None
        
    
    def init(self, y, reinit = False):
        
        if self.precision is None or reinit:
            self.precision = np.std(y) / 250
            
    #Compare predictions to the ground truth values
    def compare(self, predictions, y):
        return np.absolute(predictions - y) < self.precision
    

class Accuracy_Categorical(Accuracy):
    
    #No initialization is needed
    def init(self, y):
        pass
    
    #Compare predictions to the ground truth values
    def compare(self, predictions, y):
        
        #Convert one-hot vectors into sparse vectors
        if len(y.shape) == 2:
            y = np.argmax(y, axis = 1)
            
        return predictions == y
    
    
#------------------------------------------- Layers ------------------------------------#


class Layer_Dense:
    
    #Layer initialization
    def __init__(self, n_inputs, n_neurons, weight_regularizer_l1 = 0, weight_regularizer_l2 = 0,
                 bias_regularizer_l1 = 0, bias_regularizer_l2 = 0):
        
        #Creates a n_input X n_neurons matrix of weight
        #Dimensions of the weights are already transposed
        self.weights = 0.1 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        
        #Self regularization strength (lambda)
        self.weight_regularizer_l1 = weight_regularizer_l1
        self.weight_regularizer_l2 = weight_regularizer_l2
        self.bias_regularizer_l1 = bias_regularizer_l1
        self.bias_regularizer_l2 = bias_regularizer_l2
        
        
    #Forward pass of neural network
    def forward(self, inputs, training):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases
        
        
    #Backwward pass of neural network
    #dvalues is the gradient of the  next layer
    def backward(self, dvalues):
        #Gradient on parameters
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis = 0, keepdims = True)
        
        #Gradient on input values - How to change the inputs to minimize cost function
        self.dinputs  = np.dot(dvalues, self.weights.T)
        
        #Gradients on regularization
        #L1 regularization - Weights
        if self.weight_regularizer_l1 > 0:
            dL1 = np.ones_like(self.weights)
            dL1[self.weights < 0] = -1
            self.dweights += self.weight_regularizer_l1 * dL1
            
        #L2 regularization - Weights
        if self.weight_regularizer_l2 > 0:
            self.dweights += 2 * self.weight_regularizer_l2 * self.weights
            
        #L1 regularization - Biases
        if self.bias_regularizer_l1 > 0:
            dL1 = np.ones_like(self.biases)
            dL1[self.biases < 0] = -1
            self.dbiases += self.regularizer_l1 * dL1
            
        #L2 regularization - Biases
        if self.bias_regularizer_l2 > 0:
            self.dbiases += 2 * self.bias_regularizer_l2 * self.dbiases
        
        
class Layer_Dropout:
    
    def __init__(self, rate):
        
        #Store rate and invert it (i.e for a dropout rate of 20%, 
        #80% of the neurons will remain active)
        self.rate = 1 - rate
        
    #Forward pass
    def forward(self, inputs, training):
        
        #Save the input values
        self.inputs = inputs
        
        
        if not training:
            self.output = inputs.copy()
            return 
            
        #Generate and save scaled mask
        self.binary_mask = np.random.binomial(1, self.rate, size = inputs.shape) / self.rate
        
        #Apply mask to output values
        self.output = inputs * self.binary_mask
        
    #Backward pass
    def backward(self, dvalues):
        
        #Gradien on the output values
        self.dinputs = dvalues * self.binary_mask
        
    
#------------------------------- Activation Functions ----------------------------------------#


class Activation_ReLU:
    
    def forward(self, inputs, training):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)
    
    
        # Backward pass
    def backward(self, dvalues):
        
        # Since we need to modify the original variable,
        # let's make a copy of the values first
        self.dinputs = dvalues.copy()
        
        # Zero gradient where input values were negative
        self.dinputs[self.inputs <= 0] = 0
        
        
    def predictions(self, outputs):
        return outputs

class Activation_Softmax:
    
    def forward(self, inputs, training):
        exp_values = np.exp(inputs - np.max(inputs, axis = 1, keepdims = True))
        probabilities = exp_values / np.sum(exp_values, axis = 1, keepdims = True)
        self.output = probabilities
        
        
    # Backward pass
    def backward(self, dvalues):
        # Create uninitialized array
        self.dinputs = np.empty_like(dvalues)
        
        # Enumerate outputs and gradients
        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
            
            # Flatten output array
            single_output = single_output.reshape(-1, 1)
            
            # Calculate Jacobian matrix of the output
            #Jacobian matrix is an array of partial derivatives in all of the
            #combinations of both input vectors
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
            
            # Calculate sample-wise gradient
            # and add it to the array of sample gradients
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)


    #Calculate predictions for outputs
    def predictions(self, outputs):
        return np.argmax(outputs, axis = 1)


#Sigmoid activation function for the 
#Binary Logistic Regression
class Activation_Sigmoid:
    
    
    #Forward pass
    def forward(self, inputs, training):
        
        #Save the inputs and calculate output
        self.inputs = inputs
        self.output = 1 / (1 + np.exp(-inputs))
        
        
    #Backward pass
    def backward(self, dvalues):
        
        #Derivatives - calculates from output of the sigmoid function
        self.dinputs = dvalues * (1 - self.output) * self.output
        
        
    #Calculate predictions for outputs
    def predictions(self, outputs):
        return (outputs > 0.5) * 1
    
    
#Linear activation - y = x
class Activation_Linear:
    
    
    def forward(self, inputs):
        
        #Just save the values
        self.inputs = inputs
        self.output = inputs
        
        
    def backward(self, dvalues):
        
        #Derivative is 1
        self.dinputs = dvalues.copy()
        
        
    def predictions(self, outputs):
        return outputs
        
#------------------------------------------- Loss Functions ---------------------------------------#


class Loss:
    
    def remember_trainable_layers(self, trainable_layers):
        self.trainable_layers = trainable_layers
        
        
    def calculate(self, output, y, include_regularization = False):
        
        #Calculate sample losses
        sample_losses = self.forward(output, y)
        
        #Calculate mean loss
        data_loss = np.mean(sample_losses)
        
        #Add accumulated sum of losses and sample count for each batch during training
        self.accumulated_sum += np.sum(sample_losses)
        self.accumulated_count += len(sample_losses)
        
        #If just data loss - return it
        if not include_regularization:
            return data_loss
        
        #Return the data and regularization_loss
        return data_loss, self.regularization_loss()
    
    
    def calculate_accumulated(self, *, include_regularization = False):
        
        #Calculate mean loss
        data_loss = self.accumulated_sum / self.accumulated_count
        
        #if just data loss = return it
        if not include_regularization:
            return data_loss
        
        #Return data and regularization loss
        return data_loss, self.regularization_loss()
    
    
    #Reset cariables for accumulated loss
    def new_pass(self):
        
        self.accumulated_sum = 0
        self.accumulated_count = 0
    
    
    #Regularization loss calculation
    def regularization_loss(self):
         
        regularization_loss = 0;
        
        for layer in self.trainable_layers:
            
            #L1 regularization - weights
            if layer.weight_regularizer_l1 > 0:
                regularization_loss += layer.weight_regularizer * np.sum(np.abs(layer.weights))
                
            #L2 regularization - weights
            if layer.weight_regularizer_l2 > 0:
                regularization_loss += layer.weight_regularizer_l2 * np.sum(layer.weights * layer.weights)
                
            #L1 regularization - biases
            if layer.bias_regularizer_l1 > 0:
                regularization_loss += layer.bias_regularizer_l1 * np.sum(np.abs(layer.biases))
                
            #L2 regularization - biases
            if layer.bias_regularizer_l2 > 0:
                regularization_loss += layer.bias_regularizer_l2 * np.sum(layer.biases * layer.biases)
    
        return regularization_loss


class Loss_CategoricalCrossEntropy(Loss):
    
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)
        
        #Sparse data
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]
        
        #One-hot encoded data
        elif len(y_true.shape) == 2:
            correct_confidences == np.sum(y_pred_clipped * y_true, axis = 1)   
    
        negative_log_likelyhoods = -np.log(correct_confidences)
        
        return negative_log_likelyhoods
    
    def backward(self, dvalues, y_true):
        #Number of samples
        samples = len(dvalues)
        
        #Number of labels in each samples
        labels = len(dvalues[0])
        
        #If labels are sparse [1, 2, 3], turn them into one-hot vector 
        #that matches y_true
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]
            
        #Calculate gradient
        self.dinputs = -y_true / dvalues
        
        #Nomalize gradient
        #Sum computed by optimizer becomes invariant to sample size
        self.dinputs = self.dinputs / samples
        

#Calculate the gradient of the loss function with respect to the
#Softmax activation inputs
#7 times faster than calculating the gradients separately
class Activation_Softmax_Loss_CategoricalCrossEntropy():
    
    #Backward pass
    def backward(self, dvalues, y_true):
        
        #Number of samples
        samples = len(dvalues)
            
        #Copy so we can safely manipulate the data
        self.dinputs = dvalues.copy()
        
        #Calculate gradient
        #Substract 1 since y_true is a one-hot vector in the
        #Common Categorical Cross-Entropy loss and Softmax activation
        #derivative
        self.dinputs[range(samples), y_true] -= 1
        
        #Normalize the gradient
        self.dinputs = self.dinputs / samples
        
        
#Binary cross-entropy loss for the Binary Logistic Regression
class Loss_BinaryCrossEntropy(Loss):
    
    #Forward pass
    def forward(self, y_pred, y_true):
        
        #Clip data to prevent division by 0
        #Clip both sides to no drag mean towards any value
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)
        
        #Calculate sample-wise loss
        #Calculates a mean of all these losses from a single sample
        sample_losses = -(y_true * np.log(y_pred_clipped) + (1 - y_true) * np.log(1 - y_pred_clipped))
        sample_losses = np.mean(sample_losses, axis = -1)
        
        #Return losses
        return sample_losses
    
    
    #Backward pass
    def backward(self, dvalues, y_true):
        
        #Number of samples
        samples = len(dvalues)
        
        #Number of outputs in every sample
        outputs = len(dvalues[0])
        
        #CLip data to prevent a division by 0
        clipped_dvalues = np.clip(dvalues, 1e-7, 1 - 1e-7)
        
        #Calculate gradient
        self.dinputs = -(y_true / clipped_dvalues - (1 - y_true) / (1 - clipped_dvalues)) / outputs
        
        #Normalize the gradient
        self.dinputs = self.dinputs / samples
        
    
#Mean Squared Error Loss - L2 loss
#Punishes values that are far from the targets
class Loss_MeanSquaredError(Loss):
    
    def forward(self, y_pred, y_true):
        
        #Calculate loss
        sample_losses = np.mean((y_true - y_pred) ** 2, axis = -1)
        
        return sample_losses
    
    def backward(self, dvalues, y_true):
        
        samples = len(dvalues)
        outputs = len(dvalues[0])
        
        #Gradient on values
        self.dinputs = -2 * (y_true - dvalues) / outputs
        
        #Normalization
        self.dinputs = self.dinputs / samples


#Mean Absolute Error Loss - L1 loss
#Penalizes error linearly
class Loss_MeanAbsoluteError(Loss):
    
    def forward(self, y_pred, y_true):
        sample_losses = np.mean(np.abs(y_true - y_pred), axis = -1)
        
        return sample_losses
    
    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        outputs = len(dvalues[0])
        
        #Gradient on values
        self.dinputs = np.sign(y_true - dvalues) / outputs
        
        #Normalization
        self.dinputs = self.dinputs / samples
        

#---------------------------------------------Optimizers -------------------------------------------#


class Optimizer_SGD:
    
    #Initialize optimizer - set settings
    #Learning rate if 1. is default for this optimizer
    #Learning rate is defined as the parameter that controls how much the model
    #should be changed in response to the loss function
    #The learning rate decay allows the neural network to learn more complex patterns
    #that would not be possible with a large learning rate
    def __init__(self, learning_rate = 1.0, decay = 0., momentum = 0.):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.momentum = momentum
        
    #Call once before any parameter updates
    def pre_update_params(self):
        
        #If true
        #If self_decay != 0
        if self.decay:
            self.current_learning_rate = self.learning_rate * \
                                        (1. / (1 + self.decay * self.iterations))
        
    #Update the parameters
    def update_params(self, layer):
        
        #If we use momentum
        if self.momentum:
            
            #If layer does not contain momentum arrays, create them
            #filled with 0's
            if not hasattr(layer, "weight_momentums"):
                
                #Array will have the same shape as layer.weights
                layer.weight_momentums = np.zeros_like(layer.weights)
                
                #Same thing with the bias array
                layer.bias_momentums = np.zeros_like(layer.biases)
                
            #Build weight updates with momentum - take previous updates multiplied by retain factor and update with current gradients 
            #Update will contain a portion of the gradient from preceeding steps as our momentum and only a
            #portion of the current gradient. The bigger the momentum, the slower the update can change the direction,
            #thus avoiding getting stuck in a local minima (Ball rolling down a cliff)
            weight_updates = self.momentum * layer.weight_momentums - \
                             self.current_learning_rate * layer.dweights
            
            layer.weight_momentums = weight_updates
            
            #Build bias updates
            bias_updates = self.momentum * layer.bias_momentums - \
                           self.current_learning_rate * layer.dbiases
                           
            layer.bias_momentum = bias_updates
          
        else:
            weight_updates += -self.current_learning_rate * layer.dweights
            bias_updates += -self.current_learning_rate * layer.dbiases
            
        
        layer.weights += weight_updates
        layer.biases += bias_updates
        
        
    #Call once after the parameter updates
    def post_update_params(self):
        self.iterations += 1
        

#Institutes a per-parameter learning rate rather than a globally-shared rate
#Idea is to normalize parameter updates by keeping track of previous updates
#The bigger the sum of updates is, the smaller updates are made further in training
class Optimizer_Adagrad:
    
    #Initialize optimizer - set settings
    def __init__(self, learning_rate = 1., decay = 0., epsilon = 1e-7):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        
    #Call once before any parameter updates
    def pre_update_params(self):
        
        #If true
        #If self_decay != 0
        if self.decay:
            self.current_learning_rate = self.learning_rate * \
                                        (1. / (1 + self.decay * self.iterations))
        
    #Update the parameters
    def update_params(self, layer):
        
        #if layer does not contain cache arrays
        #Create them filled with zeros
        if not hasattr(layer, " weight_cache"):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)
    
        #The cache will hold a history of squared gradients
        #Update cahe with squared current gradients
        layer.weight_cache += layer.dweights ** 2
        layer.bias_cache += layer.dbiases ** 2
        
        #SGD parameter update with normalization and squared  rooted cache
        layer.weights += -self.current_learning_rate * layer.dweights / (np.sqrt(layer.weight_cache) + self.epsilon)
        layer.biases += -self.current_learning_rate * layer.dbiases / (np.sqrt(layer.bias_cache) + self.epsilon)
           
        
    #Call once after the parameter updates
    def post_update_params(self):
        self.iterations += 1
        
        
#RMSprop adds a mechanism similar to momentum bit also adds a per=parameter adaptive
#learning rate, so the learning rate changes are smoother - similar to AdaGrad, but cache is calculated differently
#Instead of adding squared gradients to a cache, it uses a moving average of the cache.
#Each update to the cache reatins a part of the cache and updates it with a fraction of the new
#squared gradients - Cache data move with data in time and learning does not stall.
class Optimizer_RMSprop:
    
    #Initialize optimizer - set settings
    def __init__(self, learning_rate = 0.001, decay = 0., epsilon = 1e-7, rho = 0.9):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.rho = rho
        
    #Call once before any parameter updates
    def pre_update_params(self):
        
        #If true
        #If self_decay != 0
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1. / (1 + self.decay * self.iterations))
        
    #Update the parameters
    def update_params(self, layer):
        
        #if layer does not contain cache arrays
        #Create them filled with zeros
        if not hasattr(layer, " weight_cache"):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)
    
        #The cache will hold a history of squared gradients
        #Update cahe with squared current gradients
        layer.weight_cache += self.rho * layer.weight_cache + (1 - self.rho) * layer.dweights ** 2
        layer.bias_cache += self.rho * layer.bias_cache + (1 - self.rho) * layer.dbiases ** 2
        
        #Vanilla SGD parameter update + normalization with square rotted cache
        layer.weights += -self.current_learning_rate * layer.dweights / (np.sqrt(layer.weight_cache) + self.epsilon)
        layer.biases += -self.current_learning_rate * layer.dbiases / (np.sqrt(layer.bias_cache) + self.epsilon)
           
        
    #Call once after the parameter updates
    def post_update_params(self):
        self.iterations += 1


#Apply momentum and a per-weight adaptive learning rate with the cache
#Adds a  bias correction mechanism which is appplied to the cache and momentum, compensating for the 
#initial zeroed valueds before they warm up with initial steps
class Optimizer_Adam:
    
    #Initialize optimizer - set settings
    def __init__(self, learning_rate = 0.001, decay = 0., epsilon = 1e-7, beta_1 = 0.9, beta_2 = 0.999):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        
        #Significantly speeds up training in the initial stages
        #Both betas approache 1 as the steps approache infinity
        #Return the parameter updates to their typical values for the later training steps
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        
    #Call once before any parameter updates
    def pre_update_params(self):
        
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay * self.iterations))
        
    #Update the parameters
    def update_params(self, layer):
        
        #if layer does not contain cache arrays
        #Create them filled with zeros
        if not hasattr(layer, 'weight_cache'):
            layer.weight_momentums = np.zeros_like(layer.weights)
            layer.weight_cache = np.zeros_like(layer.weights)
            
            layer.bias_momentums = np.zeros_like(layer.biases)
            layer.bias_cache = np.zeros_like(layer.biases)
    
        #Update cache with current gradients
        layer.weight_momentums = self.beta_1 * layer.weight_momentums + (1 - self.beta_1) * layer.dweights
        layer.bias_momentums = self.beta_1 * layer.bias_momentums + (1 - self.beta_1) * layer.dbiases
        
        #Get corrected momentum
        #Self.iteration is 0 at first pass and we need to start with 1 here
        #beta_1 is used for the initial step 
        weight_momentums_corrected = layer.weight_momentums / (1 - self.beta_1 ** (self.iterations + 1))
        bias_momentums_corrected = layer.bias_momentums / (1 - self.beta_1 ** (self.iterations + 1))
        
        #Update cache with squared current gradients
        layer.weight_cache = self.beta_2 * layer.weight_cache + (1 - self.beta_2) * layer.dweights ** 2
        layer.bias_cache = self.beta_2 * layer.bias_cache + (1 - self.beta_2) * layer.dbiases ** 2
        
        #Get corrected cache
        weight_cache_corrected = layer.weight_cache / (1 - self.beta_2 ** (self.iterations + 1))
        bias_cache_corrected = layer.bias_cache / (1 - self.beta_2 ** (self.iterations + 1))
        
        
        #Vanilla SGD parameter update + normalization with square rotted cache
        layer.weights += -self.current_learning_rate * weight_momentums_corrected / (np.sqrt(weight_cache_corrected) + self.epsilon)
        layer.biases += -self.current_learning_rate * bias_momentums_corrected / (np.sqrt(bias_cache_corrected) + self.epsilon)
           
        
    #Call once after the parameter updates
    def post_update_params(self):
        self.iterations += 1


#---------------------------------------------- Model Class ------------------------------------------#

class Model:
    
    def __init__(self):
        
        #Create a list of network objects
        self.layers = []
        
        #Softmax classigier's output object
        self.softmax_classifier_output = None
        
    #Add objects to the model
    def add(self, layer):
        self.layers.append(layer)
        
    #Set loss and optimizer
    #Asterisk notes that the subsequent parameters are keyword arguments
    def set(self, *, loss, optimizer, accuracy):
        self.loss = loss
        self.optimizer = optimizer
        self.accuracy = accuracy
    
    
    #Finalize the model
    def finalize(self):
        
        #Create and set the input layer
        self.input_layer = Layer_Input()
        
        #count all the objects
        layer_count = len(self.layers)
        
        #Initialize a list containing trainable layers
        self.trainable_layers = []
        
        #Iterate over the objects
        for i in range(layer_count):
            
            #If it is the first layer, the previous layer object is the input layer
            if i == 0:
                self.layers[i].prev = self.input_layer
                self.layers[i].next = self.layers[i + 1]
            
            #All layers except for the first and last
            elif i < layer_count - 1:
                self.layers[i].prev = self.layers[i - 1]
                self.layers[i].next = self.layers[i + 1]
                
            #Output layer is hooked to the loss function
            else:
                self.layers[i].prev = self.layers[i - 1]
                self.layers[i].next = self.loss
                self.output_layer_activation = self.layers[i]
              
                
             #If layer contains an attribute called "weights",
             #then it is trainable
             #Add it to the list of trainable layers
            if hasattr(self.layers[i], "weights"):
                self.trainable_layers.append(self.layers[i])
        
            #Update loss object with trainable layers
            self.loss.remember_trainable_layers(self.trainable_layers)
        
        #If output activation is Softmax and loss function is Categorial-Cross Entropy,
        #then, create an object of combined activation and loss function
        #Faster gradient calculation
        if isinstance(self.layers[-1], Activation_Softmax) and isinstance(self.loss, Loss_CategoricalCrossEntropy):
            self.softmax_classifier_output = Activation_Softmax_Loss_CategoricalCrossEntropy()
            
            
    #Train the model
    def train(self, X, y, *, epochs = 1, batch_size = None, print_every = 1, validation_data = None):
        
        # Initialize accuracy object
        self.accuracy.init(y)
        
        #Default value if batch size is not being set since training will take 
        #one step as we pass the whole dataset at once
        train_steps = 1
        
        #If there is validation data passed
        #Set the default number of steps for validation as well
        if validation_data is not None:
            validation_steps = 1
            
            X_val, y_val = validation_data
            
        
        #Calculate number of steps
        if batch_size is not None:
            train_steps = len(X) // batch_size
            
            #If there are some remaining data, add 1 to the number of steps
            if train_steps * batch_size < len(X):
                train_steps += 1
                
            if validation_steps is not None:
                validation_steps = len(X_val) // batch_size
                
                if validation_steps * batch_size < len(X_val):
                    validation_steps += 1
        
        #Main training loop
        for epoch in range(1, epochs + 1):
            
            #Print epoch number
            print(f'Epoch: {epoch}')
            
            #Reset accumulated values in loss and accuracy objects
            self.loss.new_pass()
            self.accuracy.new_pass()
            
            #Iterate over training steps
            for step in range(train_steps):
                
                #If batch size is not set
                #Train using one step and full dataset
                if batch_size is None:
                    batch_X = X
                    batch_y = y
                    
                #Otherwise, slice a batch
                else:
                    batch_X = X[step * batch_size : (step + 1) * batch_size]
                    batch_y = y[step * batch_size : (step + 1) * batch_size]
                    
                    
                #Perform the forward pass
                output = self.forward(batch_X, training = True)
                
                #Calculate loss
                data_loss, regularization_loss = self.loss.calculate(output, batch_y, include_regularization = True)
                loss = data_loss + regularization_loss
                
                #Get predictions and calculate an accuracy
                predictions = self.output_layer_activation.predictions(output)
                accuracy = self.accuracy.calculate(predictions, batch_y)
            
                #Perform backward pass
                self.backward(output, y)
            
                #Optimize the parameters
                self.optimizer.pre_update_params()
                
                for layer in self.trainable_layers:
                    self.optimizer.update_params(layer)
                    
                self.optimizer.post_update_params()
            
                # Print a summary
                if not step % print_every or step == train_steps - 1:
                    print(f'step: {step}, ' +
                          f'acc: {accuracy:.3f}, ' +
                          f'loss: {loss:.3f} (' +
                          f'data_loss: {data_loss:.3f}, ' +
                          f'reg_loss: {regularization_loss:.3f}), ' +
                          f'lr: {self.optimizer.current_learning_rate}')
                
            #Get and print epoch loss and accuracy
            epoch_data_loss, epoch_regularization_loss = self.loss.calculate_accumulate(include_regularization = True)
            epoch_loss = epoch_data_loss + epoch_regularization_loss
            epoch_accuracy = self.accuracy.calculate_accumulated()
            
            print(f'training, ' +
                  f'acc: {epoch_accuracy:.3f}, ' +
                  f'loss: {epoch_loss:.3f} (' +
                  f'data_loss: {epoch_data_loss:.3f}, ' +
                  f'reg_loss: {epoch_regularization_loss:.3f}), ' +
                  f'lr: {self.optimizer.current_learning_rate}')

            #If there is validation data
            if validation_data is not None:
                
                #Reset accumulated values in loss and accuracy objects
                self.loss.new_pass()
                self.accuracy.new_pass()
                
                #Iterate over validation steps
                for step in range(validation_steps):
                    
                    #If batch size is not set
                    if batch_size is None:
                        batch_X = X_val
                        batch_y = y_val
                        
                    else:
                        batch_X = X_val[step * batch_size : (step + 1) * batch_size]
                        batch_y = y_val[step * batch_size : (step + 1) * batch_size]
                        
               
                    #Perform forward pass
                    output = self.forward(batch_X, training = False)
                    
                    #Calculate the loss
                    loss = self.loss.calculate(output, batch_y)
                    
                    #Get predictions and calculate accuracy
                    predictions = self.output_layer_activation.predictions(output)
                    accuracy = self.accuracy.calculate(predictions, batch_y)
                
                #Get and print validation loss and accuracy
                validation_loss = self.loss.calculate_accumulated()
                validation_accuracy = self.accuracy.calculate_accumulated()
                
                print(f'Validation: ' +
                      f'acc: {validation_accuracy:.3f}, ' +
                      f'loss: {validation_loss:.3f}')

    
    #Forward pass
    def forward(self, X, training):
        
        #Call forward method on the input layer
        self.input_layer.forward(X, training)
        
        #Call forward method of every object in a chain
        #Pass output of the previous layer as inputs to the next
        #First layer in self.layers is the first hidden layer
        for layer in self.layers:
            layer.forward(layer.prev.output, training)
            
        #Return the output of the last layer
        return layer.output
         

    #Backward pass
    def backward(self, output, y):
        
        if self.softmax_classifier_output is not None:
            
            #First call backward method on the combined activation/loss
            #This will set dinputs property
            self.softmax_classifier_output.backward(output, y)
            
            #Since we will not call backward method of the last layer, we set the dinputs of this object
            self.layers[-1].dinputs = self.softmax_classifier_output.dinputs
            
            #Call backward method going through all the objects but the last
            #i reversed order
            for layer in reversed(self.layers[:-1]):
                layer.backward(layer.next.dinputs)
                
            return 
        
        
        #First call backward method on the loss
        #this will set dinputs property that the last layer will try to access shortly
        self.loss.backward(output, y)
        
        #Call backward method going through all the objects in reversed order
        #passing dinputs as the parameter
        for layer in reversed(self.layers):
            layer.backward(layer.next.dinputs)

    
#-------------------------------------- Implementation of Regression ----------------------------------#

'''
#Create dataset
X, y = sine_data()


#Instantiate the model
model = Model()


#Add layers
model.add(Layer_Dense(1, 64))
model.add(Activation_ReLU())
model.add(Layer_Dense(64, 64))
model.add(Activation_ReLU())
model.add(Layer_Dense(64, 1))
model.add(Activation_Linear())


#Set the loss and optimizer objects
model.set(loss = Loss_MeanSquaredError(), 
          optimizer = Optimizer_Adam(learning_rate = 0.005, decay = 1e-3),
          accuracy = Accuracy_Regression())


#Finalize the model
model.finalize()

#Train the model
model.train(X, y, epochs = 10000, print_every = 100)
'''


#-----------------Implementation of Binary Logistic Regression --------------------#


'''
#Creates a dataset of 200 points distributed into 2 classes/labels
X, y = spiral_data(samples=100, classes=2)
X_test, y_test = spiral_data(samples=100, classes=2)

# Reshape labels to be a list of lists
# Inner list contains one output (either 0 or 1)
# per each output neuron, 1 in this case
y = y.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

# Instantiate the model
model = Model()

# Add layers
model.add(Layer_Dense(2, 64, weight_regularizer_l2=5e-4, bias_regularizer_l2=5e-4))
model.add(Activation_ReLU())
model.add(Layer_Dense(64, 1))
model.add(Activation_Sigmoid())
model.set(loss=Loss_BinaryCrossEntropy(), optimizer=Optimizer_Adam(decay=5e-7), accuracy=Accuracy_Categorical())

# Finalize the model
model.finalize()

# Train the model
model.train(X, y, validation_data=(X_test, y_test), epochs=10000, print_every=100)
'''


#-----------------Implementation of Classification --------------------#


'''
# Creates a dataset of 3000 points distributed into 3 classes/labels
X, y = spiral_data(samples=1000, classes=3)
X_test, y_test = spiral_data(samples=100, classes=3)

# Instantiate the model
model = Model()

# Add layers
model.add(Layer_Dense(2, 512, weight_regularizer_l2=5e-4,
                      bias_regularizer_l2=5e-4))
model.add(Activation_ReLU())
model.add(Layer_Dropout(0.1))
model.add(Layer_Dense(512, 3))
model.add(Activation_Softmax())

# Set loss, optimizer and accuracy objects
model.set(loss=Loss_CategoricalCrossEntropy(), optimizer=Optimizer_Adam(learning_rate=0.05, decay=5e-5),
          accuracy=Accuracy_Categorical())

# Finalize the model
model.finalize()

# Train the model
model.train(X, y, validation_data=(X_test, y_test), epochs=10000, print_every=100)
'''


#----------------------- Implementation with a Real Dataset ------------------------#

#Loads a MNIST dataset
def load_mnist_dataset(dataset, path):
    
    #Scan all the directorues and create a list of labels
    labels = os.listdir(os.path.join(path, dataset))

    #Create list for samples and labels
    X = []
    y = []

    #For each label folder
    for label in labels:
        
        #And for each image in a given folder
        for file in os.listdir(os.path.join(path, dataset, label)):
            
            #Read the image
            image = cv2.imread(os.path.join(path, dataset, label, file), cv2.IMREAD_UNCHANGED)
            
            #And append it and a label to the list
            X.append(image)
            y.append(label)
    
    return np.array(X), np.array(y).astype('uint8')


#MNIST dataset (train + test)
def create_data_mnist(path):
    
    #Load both sets separately
    X, y = load_mnist_dataset('train', path)
    X_test, y_test = load_mnist_dataset('test', path)
    
    #Return all the sata
    return X, y, X_test, y_test


'''
#Create dataset
X, y, X_test, y_test = create_data_mnist('fashion_mnist_images')

#Scale features
X = (X.astype(np.float32) - 127.5) / 127.5
X_test = (X_test.astype(np.float32) - 127.5) / 127.5

#Reshape to vectors of shape (60000, 784)
X = X.reshape(X.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)

#Data shuffling
keys = np.array(range(X.shape[0]))

np.random.shuffle(keys)

#Shuffle X and y according to keys, so that they match 
X = X[keys]
y = y[keys]

'''


X, y = spiral_data(samples = 100, classes = 3)

EPOCH = 10
BATCH_SIZE = 128

#Calculate the number of steps per epoch
steps = X.shape[0] // BATCH_SIZE

#If there are some remaining data, add 1 to the number of steps
if steps * BATCH_SIZE < X.shape[0]:
    steps += 1
    
#During each step in each epoch, we are selecting a slice of the training data
#Each batch contains 128 samples
for epoch in range(EPOCH):
    for step in range(steps):
        batch_X = X[step * BATCH_SIZE: (step + 1) * BATCH_SIZE]
        batch_y = y[step * BATCH_SIZE: (step + 1) * BATCH_SIZE]



















