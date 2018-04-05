import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from math import e
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

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



def weighted_sum(X, unit_num):
    initializer = tf.contrib.layers.xavier_initializer()
    W = tf.Variable(initializer([X.shape[1].value, unit_num]), name='W')
    b = tf.Variable(tf.zeros(unit_num), name='b')

    return tf.add(tf.matmul(X, W), b)


def buildGraph(learning_rate, num_layers, hidden_units, dropout):

	#inputs
	X = tf.placeholder(tf.float32, [None, 28, 28], name='input_x')
	y_target = tf.placeholder(tf.float32, name='target_y')

	#parse inputs
	X_flatten = tf.reshape(X, [-1, 28*28])
	y_onehot = tf.one_hot(tf.to_int32(y_target), 10, 1.0, 0.0, axis = -1)


	#init the input
	sums = X_flatten

	for i in range(0, num_layers):

		#get sums
		sums = tf.nn.relu(weighted_sum(sums, hidden_units))

		if dropout:
			#apply drop out
			sums = tf.nn.dropout(sums, 0.5)

	#output layer
	y_predicted = weighted_sum(sums, 10)

	#get cross entropy error
	crossEntropyLoss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y_onehot, logits = y_predicted))

	#compute accuracy
	accuracy = tf.reduce_mean(tf.to_float(tf.equal(tf.argmax(y_predicted, -1), tf.to_int64(y_target))))

	#init optimizer
	optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
	train = optimizer.minimize(loss=(crossEntropyLoss + 3 * e - 4))

	return X, y_target, y_predicted, crossEntropyLoss, train, accuracy




# contants
training_size = 15000



# init hyper parameters
learning_rate = 0.001
batch_size = 200


num_layers = 1
hidden_units = 1000

max_iter = 20000

dropout = False

# vairables for uses

trainError_list = []
validError_list = []
testError_list = []

loss_list = []
epoch_list = []

numBatches = np.floor(len(trainData)/batch_size)

X, y_target, y_predicted, crossEntropyLoss, train, accuracy = buildGraph(learning_rate, num_layers, hidden_units, dropout)

init = tf.global_variables_initializer()
sess = tf.InteractiveSession()

sess.run(init)

for k in range(0, max_iter):
	index = (batch_size * k) % training_size

	batch_Data = trainData[index: index + batch_size]
	batch_Target = trainTarget[index: index + batch_size]

    #learning
	_, loss, yhat, accu = sess.run([train, crossEntropyLoss, y_predicted, accuracy], feed_dict = {X: batch_Data, y_target: batch_Target})

	if index == 0:
		#get cross entropy loss
		loss_list.append(loss)

		#get training error
		accu = accuracy.eval(feed_dict = {X: trainData, y_target: trainTarget})
		trainError_list.append(1 - accu)

		#get validation error
		accu = accuracy.eval(feed_dict = {X: validData, y_target: validTarget})
		validError_list.append(1 - accu)

		#get test error
		accu = accuracy.eval(feed_dict = {X: testData, y_target: testTarget})
		testError_list.append(1 - accu)

		#get index
		i = int((batch_size * k) / training_size)

		#set index
		epoch_list.append(i)

	if not (k % (max_iter / 4)):
		print(" In progress " + str(100 * k / max_iter)  + "%")

		currentW = tf.get_default_graph().get_tensor_by_name("W:0")

		img = tf.reshape(currentW, shape=[-1, 28, 28, 1])

		tf.summary.image("image", img)


print("Final loss is  " + str(loss_list[-1]))


print("Training error is " + str(trainError_list[-1]))
print("Validation error is " + str(validError_list[-1]))
print("Test error is " + str(testError_list[-1]))


plt.figure(1)
plt.plot(epoch_list, trainError_list,'-', label = "training set")
plt.plot(epoch_list, validError_list,'-', label = "validation set")
plt.plot(epoch_list, testError_list,'-', label = "test set")

plt.xlabel('number of epoches')
plt.ylabel('error')
plt.legend()

plt.title("Nearul Network Errors with hidden units " + str(hidden_units))
plt.show()




plt.figure(2)
plt.plot(epoch_list, loss_list,'-', label = "training set")

plt.xlabel('number of epoches')
plt.ylabel('cross entropy loss')
plt.legend()

plt.title("Nearul Network loss with hidden units " + str(hidden_units))
plt.show()





















