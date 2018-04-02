import tensorflow as tf
import numpy as np

with np.load("notMNIST.npz") as data:
    Data, Target = data ["images"], data["labels"]
    np.random.seed(521)
    randIndx = np.arange(len(Data))
    np.random.shuffle(randIndx)
    Data = Data[randIndx]/255.
    Target = Target[randIndx]
    trainData, trainTarget = Data[:15000], Target[:15000]
    validData, validTarget = Data[15000:16000], Target[15000:16000]
    testData, testTarget = Data[16000:], Target[16000:]

def weighted_sum(data, unit_num):

   
    W = tf.get_variable(shape=[data.shape, unit_num],initializer=tf.contrib.layers.xavier_initializer())
    b = tf.Variable(tf.zeros(unit_num), name='biases')

    hidden_weights = tf.Variable(initializer([x_dimension, hidden_units]), name='weights')
    
    
    return tf.add(tf.matmul(X, hidden_weights), hidden_biases)








