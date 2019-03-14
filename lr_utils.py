import numpy as np
import h5py
import tensorflow as tf
import math
    
def load_dataset():
    train_dataset = h5py.File('datasets/train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels

    test_dataset = h5py.File('datasets/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels

    classes = np.array(test_dataset["list_classes"][:]) # the list of classes
    
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes

def predict(X, parameters):

	W1 = tf.convert_to_tensor(parameters["W1"])
	b1 = tf.convert_to_tensor(parameters["b1"])
	W2 = tf.convert_to_tensor(parameters["W2"])
	b2 = tf.convert_to_tensor(parameters["b2"])
	W3 = tf.convert_to_tensor(parameters["W3"])
	b3 = tf.convert_to_tensor(parameters["b3"])

	params = {"W1": W1,
			  "b1": b1,
			  "W2": W2,
			  "b2": b2,
			  "W3": W3,
			  "b3": b3}
	x = tf.placeholder("float", [12288, 1])
	z3 = forward_propagation_for_predict(x, params)
	#print(z3)
	p = tf.sigmoid(z3)
	sess = tf.Session()
	prediction = sess.run(p, feed_dict = {x: X})
	print(prediction)
	if prediction > 0.5:
		prediction = 1
	else:
		prediction = 0
	#print("prediction is"+str(prediction))
	return prediction

def forward_propagation_for_predict(X, parameters):
    """
    Implements the forward propagation for the model: LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SOFTMAX
    
    Arguments:
    X -- input dataset placeholder, of shape (input size, number of examples)
    parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3"
                  the shapes are given in initialize_parameters

    Returns:
    Z3 -- the output of the last LINEAR unit
    """
    
    # Retrieve the parameters from the dictionary "parameters" 
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3'] 
                                                           # Numpy Equivalents:
    Z1 = tf.add(tf.matmul(W1, X), b1)                      # Z1 = np.dot(W1, X) + b1
    A1 = tf.nn.relu(Z1)                                    # A1 = relu(Z1)
    Z2 = tf.add(tf.matmul(W2, A1), b2)                     # Z2 = np.dot(W2, a1) + b2
    A2 = tf.nn.relu(Z2)                                    # A2 = relu(Z2)
    Z3 = tf.add(tf.matmul(W3, A2), b3)                     # Z3 = np.dot(W3,Z2) + b3
    
    return Z3