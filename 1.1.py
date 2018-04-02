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



def weighted_sum(X, unit_num):
    W = tf.get_variable(shape=[784, 10, unit_num],initializer=tf.contrib.layers.xavier_initializer())
    b = tf.Variable(tf.zeros(unit_num), name='biases')
    return tf.add(tf.matmul(X, W), b)


def buildGraph(lr):

	#inputs
	X = tf.placeholder(tf.float32, [None, 28, 28], name='input_x')
	y_target = tf.placeholder(tf.float32, name='target_y')

	#parse inputs
	X_flatten = tf.reshape(X, [-1, 28*28])
	y_onehot = tf.one_hot(tf.to_int32(y_target), 10, 1.0, 0.0, axis = -1)

	#get sums
	sums = weighted_sum(X_flatten, 1000)
	y_predicted = weighted_sum(tf.nn.relu(sums), 10)

	#get cross entropy error
	crossEntropyLoss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y_onehot, logits = y_predicted))

	#compute accuracy
	accuracy = tf.reduce_mean(tf.to_float(tf.equal(tf.argmax(y_predicted, -1), tf.to_int64(y_target))))

	#init optimizer
	optimizer = tf.train.AdamOptimizer(learning_rate = lr)
	train = optimizer.minimize(loss=crossEntropyLoss)

	return X, y_target, y_predicted, crossEntropyLoss, train, accuracy











learning_rate = 0.01
batch_size = 500
training_size = 15000

max_iter = 200

trainError_list = []
validError_list = []
testError_list = []

loss_list = []

epoch_list = []


X, y_target, y_predicted, crossEntropyError, train, accuracy = buildGraph(learning_rate)



for k in range(0, max_iter - 1):
	index = (batch_size * k)/training_size

	batch_Data = trainData[index: index + batchSize]
    batch_Target = trainTarget[index: index + batchSize]

   	#learning
	_, loss, y_predicted, accuracy = sess.run([train, crossEntropyLoss, y_predicted, accuracy], feed_dict = {X: batch_Data, y_target: batch_Target})
	
	#get cross entropy loss
	loss_list.append(loss)

	#get training error
	err = 1 - accuracy.eval(feed_dict = {X: trainData, y_target: trainTarget})
	trainError_list.append(err)

	#get validation error
	err = 1 - accuracy.eval(feed_dict = {X: validData, y_target: validTarget})
	validError_list.append(err)

	#get test error
	err = 1 - accuracy.eval(feed_dict = {X: trainData, y_target: trainTarget})
	testError_list.append(err)


	#set index
	epoch_list.append(k)

	if not step % (batchSize * 5):
                print("step - %d"%(step))


plt.figure(1)
plt.plot(epoch_list[0], trainError_list[0],'-', label = "training set error")
plt.plot(epoch_list[0], validError_list[1],'-', label = "validation set error")
plt.plot(epoch_list[0], testError_list[2],'-', label = "test set error")
plt.legend()

plt.title("Nearul Netword")
plt.show()





















