import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from math import e
import os
import time
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

    return tf.add(tf.matmul(X, W), b), W


def buildGraph(learning_rate, num_layers, hidden_units, dropout, weight_decay):

	#inputs
	X = tf.placeholder(tf.float32, [None, 28, 28], name='input_x')
	y_target = tf.placeholder(tf.float32, name='target_y')

	#parse inputs
	X_flatten = tf.reshape(X, [-1, 28*28])
	y_onehot = tf.one_hot(tf.to_int32(y_target), 10, 1.0, 0.0, axis = -1)


	#init the input
	input_x = X_flatten

	weight_loss = 0

	for i in range(0, num_layers):

		#get sums
		sums, W = weighted_sum(input_x, hidden_units)

		input_x = tf.nn.relu(sums)

		weight_loss += tf.reduce_sum(W*W) * weight_decay * 0.5

		if dropout:
			#apply drop out
			input_x = tf.nn.dropout(input_x, 0.5)

	#output layer
	y_predicted, W_out = weighted_sum(input_x, 10)

	weight_loss += tf.reduce_sum(W_out*W_out) * weight_decay * 0.5

	#get cross entropy error
	crossEntropyLoss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y_onehot, logits = y_predicted))

	#compute accuracy
	accuracy = tf.reduce_mean(tf.to_float(tf.equal(tf.argmax(y_predicted, -1), tf.to_int64(y_target))))


	#init optimizer
	optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
	train = optimizer.minimize(loss=(crossEntropyLoss + weight_loss))

	return X, y_target, y_predicted, crossEntropyLoss, train, accuracy,W




# contants
training_size = 15000
batch_size = 200
max_iter = 20000


# hyper parameters
learning_rate = 0.001
weight_decay = 0
dropout = False
num_layers = 1
hidden_units = 1000


# other parameterys

image_w = True
plot = False
random = False


# random parameters
if random:
	np.random.seed(int(time.time()))

	learning_rate = np.power(10, np.random.uniform(-7.5, -4.5))
	num_layers = np.random.randint(1, 6)
	hidden_units = np.random.randint(100, 501)
	weight_decay = np.power(e, np.random.uniform(-9, -6))
	dropout = np.random.randint(0, 2)





numBatches = np.floor(len(trainData)/batch_size)


lrs = [learning_rate]

loss_array = []

for learning_rate in lrs:

	# vairables for uses
	trainError_list = []
	validError_list = []
	testError_list = []

	loss_list = []
	epoch_list = []

	X, y_target, y_predicted, crossEntropyLoss, train, accuracy,W = buildGraph(learning_rate, num_layers, hidden_units, dropout, weight_decay)

	init = tf.global_variables_initializer()
	sess = tf.InteractiveSession()

	sess.run(init)

	early_v_e = 10000

	for k in range(0, max_iter + 1):
		index = (batch_size * k) % training_size

		batch_Data = trainData[index: index + batch_size]
		batch_Target = trainTarget[index: index + batch_size]

	    #learning
		_, loss, yhat, accu, currentW = sess.run([train, crossEntropyLoss, y_predicted, accuracy, W], feed_dict = {X: batch_Data, y_target: batch_Target})

		if index == 0:

			#get index
			i = int((batch_size * k) / training_size)

			#get cross entropy loss
			loss_list.append(loss)

			#get training error
			accu = accuracy.eval(feed_dict = {X: trainData, y_target: trainTarget})
			trainError_list.append(1 - accu)

			#get validation error
			accu = accuracy.eval(feed_dict = {X: validData, y_target: validTarget})
			validError_list.append(1 - accu)

			v_error = 1 - accu

			#get test error
			accu = accuracy.eval(feed_dict = {X: testData, y_target: testTarget})
			testError_list.append(1 - accu)


			if (early_v_e > validError_list[-1]):
				early_i = i
				early_n_e = trainError_list[-1]
				early_v_e = validError_list[-1]
				early_t_e = testError_list[-1]




			#set index
			epoch_list.append(i)

		if not (k % (max_iter / 4)):
			print(" In progress " + str(100 * k / max_iter)  + "%")


			if image_w and (int(100 * k / max_iter) == 25 or int(100 * k / max_iter) == 100):

				print((currentW.transpose()).shape)
				currentW = currentW.transpose()

				for i in range(0, currentW.shape[0]):
					index = i % 100
					if (index == 0):
						plt.figure("progress-"+str(100 * k / max_iter) + "  img-"+str(int(i / 100)))


					plt.subplot(10,10, index + 1)
					img = np.reshape(currentW[i], (28,28))


					plt.imshow(img, cmap=plt.gray(),interpolation='nearest')
					plt.axis('off')


					if (index == 99):
						plt.show()







	print("######### Hyper parameters #########")
	print("Learning rate    = " + str(learning_rate))
	print("Layers           = " + str(num_layers))
	print("Hidden units     = " + str (hidden_units))
	print("Weithed decay    = " + str(weight_decay))
	print("Dropout          = " + str(dropout))

	print("######### Result #########")

	print("Final loss       = " + str(loss_list[-1]))
	print("Training error   = " + str(trainError_list[-1]))
	print("Validation error = " + str(validError_list[-1]))
	print("Test error       = " + str(testError_list[-1]))

	print("######### Early Stop #########")
	print("number of epoch  = " + str(early_i))
	print("Training error   = " + str(early_n_e))
	print("Validation error = " + str(early_v_e))
	print("Test error       = " + str(early_t_e))

	loss_array.append(loss_list)












if plot:
	plt.figure(1)
	plt.plot(epoch_list, trainError_list,'-', label = "training set")
	plt.plot(epoch_list, validError_list,'-', label = "validation set")
	plt.plot(epoch_list, testError_list,'-', label = "test set")

	plt.xlabel('number of epochs')
	plt.ylabel('error')
	plt.legend()

	plt.title("Errors with  learning_rate = " + str(learning_rate) +" layers = " + str(num_layers) +" hidden units = "+ str(hidden_units))
	plt.show()


	plt.figure(2)

	for index, item in enumerate(loss_array):
		plt.plot(epoch_list, loss_array[index],'-', label = "learning_rate is " + str(lrs[index]))

	plt.xlabel('number of epochs')
	plt.ylabel('cross entropy loss')
	plt.legend()

	plt.title("Training loss with" + " layers = " + str(num_layers) +" hidden units = "+ str(hidden_units))
	plt.show()


