import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops

def one_hot_matrix(labels, C):
    """
    Creates a matrix where the i-th row corresponds to the ith class number and the jth column
                     corresponds to the jth training example. So if example j had a label i. Then entry (i,j) 
                     will be 1. 
                     
    Arguments:
    labels -- vector containing the labels 
    C -- number of classes, the depth of the one hot dimension
    
    Returns: 
    one_hot -- one hot matrix
    """
    
    ### START CODE HERE ###
    
    # Create a tf.constant equal to C (depth), name it 'C'. (approx. 1 line)
    C = tf.constant(C, name = "C")
    
    # Use tf.one_hot, be careful with the axis (approx. 1 line)
    one_hot_matrix = tf.one_hot(labels, C, axis=0)
    
    # Create the session (approx. 1 line)
    sess = tf.Session()
    
    # Run the session (approx. 1 line)
    one_hot = sess.run(one_hot_matrix)
    
    # Close the session (approx. 1 line). See method 1 above.
    sess.close()
    
    ### END CODE HERE ###
    
    return one_hot

def initParam():
    W1 = tf.get_variable("W1", shape=(25, 57600), initializer=tf.contrib.layers.xavier_initializer(seed=1), dtype=tf.float32)
    b1 = tf.get_variable("b1", shape=(25, 1), initializer=tf.zeros_initializer(), dtype=tf.float32)
    W2 = tf.get_variable("W2", shape=(10, 25), initializer=tf.contrib.layers.xavier_initializer(seed=1), dtype=tf.float32)
    b2 = tf.get_variable("b2", shape=(10, 1), initializer=tf.zeros_initializer(), dtype=tf.float32)
    W3 = tf.get_variable("W3", shape=(2, 10), initializer=tf.contrib.layers.xavier_initializer(seed=1), dtype=tf.float32)
    b3 = tf.get_variable("b3", shape=(2, 1), initializer=tf.zeros_initializer(), dtype=tf.float32)

    return {"W1": W1,
            "b1": b1,
            "W2": W2,
            "b2": b2,
            "W3": W3,
            "b3": b3}


def initPlaceholders(xsize, ysize):
    X = tf.placeholder(tf.float32, [xsize, None])
    Y = tf.placeholder(tf.float32, [ysize, None])

    return X, Y

def forwardProp(X, parameters):
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']


    Z1 = tf.add(tf.matmul(W1, X), b1)
    A1 = tf.nn.relu(Z1)
    Z2 = tf.add(tf.matmul(W2, A1), b2)
    A2 = tf.nn.relu(Z2)
    Z3 = tf.add(tf.matmul(W3, A2), b3)

    return Z3

def computeCost(Z3, Y):
    logits = tf.transpose(Z3)
    labels = tf.transpose(Y)

    loss = tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = labels)
    cost = tf.reduce_mean(loss)

    return cost


def model(X_train, Y_train, lr = 0.0001, num_epochs=2000, minibatch_size = 50, print_cost = True):
    tf.set_random_seed(1)
    seed = 3
    xsize, m = X_train.shape
    ysize = Y_train.shape[0]
    costs = []

    X, Y = initPlaceholders(xsize, ysize)

    parameters = initParam()

    Z3 = forwardProp(X, parameters)

    cost = computeCost(Z3, Y)

    optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(cost)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)

        for epoch in range( num_epochs ):

            _, c = sess.run([optimizer, cost], feed_dict={
                X: X_train, 
                Y: Y_train
            })

            print(epoch, ":", c)
            costs.append(c)

        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('epoch')
        plt.title('Learning rate')
        plt.show()
    
        parameters = sess.run(parameters)

        tf.train.Saver().save(sess, "sess.ckpt")
        
        return parameters


def get_predictor(size):
    W1 = tf.get_variable("W1", shape=(25, 57600), dtype=tf.float32)
    b1 = tf.get_variable("b1", shape=(25, 1), dtype=tf.float32)
    W2 = tf.get_variable("W2", shape=(10, 25), dtype=tf.float32)
    b2 = tf.get_variable("b2", shape=(10, 1), dtype=tf.float32)
    W3 = tf.get_variable("W3", shape=(2, 10), dtype=tf.float32)
    b3 = tf.get_variable("b3", shape=(2, 1), dtype=tf.float32)
    
    X = tf.placeholder(tf.float32,shape=size)

    Z1 = tf.add(tf.matmul(W1, X), b1)
    A1 = tf.nn.relu(Z1)
    Z2 = tf.add(tf.matmul(W2, A1), b2)
    A2 = tf.nn.relu(Z2)
    Z3 = tf.add(tf.matmul(W3, A2), b3)

    predictor = tf.argmax(Z3)

    saver = tf.train.Saver()
    sess = tf.Session()

    saver.restore(sess, "sess.ckpt")
    
    return predictor, sess, X

X_train = np.load("dataset_x.npy")
Y_train = np.load("dataset_y.npy")


# params = model(X_train, Y_train, num_epochs=100)


predictor, sess, X = get_predictor(X_train[:,0:1].shape)

for i in range(1500):
    prediction = sess.run(predictor, feed_dict={X: X_train[:,i:i+1]})

    # print(prediction, np.argmax(Y_train[:, i:i+1]), sep=": ")

    if not prediction[0] == np.argmax(Y_train[:, i:i+1]):
        print(i)
