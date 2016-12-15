#all necessary tensorflow imports
import numpy as np
import tensorflow as tf
# from tensorflow.models.rnn import rnn, rnn_cell
# import tf.nn.rnn_*
# from tf.nn.rnn_* import rnn, rnn_cell
# import tensorflow.models.rnn.rnn as rnn
# import tensorflow.models.rnn.rnn_cell as rnn_cell

from tensorflow.python.ops.rnn_cell import BasicLSTMCell

from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import rnn



#my personal loading data imports
import sys
sys.path.append("./../Baseline")

import loadData as ld
import baseline
import csv
import os

LOSS_FILE = "./../Loss Files/LstmLoss.csv"

class runLSTM:
	def __init__(self, featureSet, classificationSet):
		self.featureSet = featureSet
		self.classSet = classificationSet
		self.Dir = os.path.dirname(__file__)



		self.classifications = self.getActualClassification()

		self.initLSTM()
		self.train_neural_network(self.x)
		self.writeLoss()

	def initLSTM(self):
		self.num_Epochs = 50 # number of training iterations
		self.n_classes = 2 #number of possible classifications
		self.batch_size = 5 #pushing one training point through at a time
		self.chunk_size = len(self.featureSet[0])
		self.n_chunks = 1
		self.rnn_size = 256
		self.learning_rate = 0.001

		self.x = tf.placeholder('float', [None, self.n_chunks, self.chunk_size])
		self.y = tf.placeholder('float', [None, self.n_classes])


	def recurrent_neural_network(self, x):
		layer = {'weights':tf.Variable(tf.random_normal([self.rnn_size,self.n_classes])),
			 'biases':tf.Variable(tf.random_normal([self.n_classes]))}

		x = tf.transpose(x, [1, 0, 2])
		x = tf.reshape(x, [-1, self.chunk_size])
		x = tf.split(0, self.n_chunks, x)


		lstm_cell = rnn_cell.BasicLSTMCell(self.rnn_size)
		outputs, states = rnn.rnn(lstm_cell, x, dtype = tf.float32)
		output = tf.matmul(outputs[-1], layer['weights']) + layer['biases']

		return output


	def train_neural_network(self, x):
		def test_neural_network():

			remissionFeatures = [featurePoint for i, featurePoint in enumerate(self.featureSet) if self.classifications[i] == [1,0]]
			remissionClassification = [classpoint for i, classpoint in enumerate(self.classifications) if self.classifications[i] == [1,0]]
			noneFeatures = [featurePoint for i, featurePoint in enumerate(self.featureSet) if self.classifications[i] != [1,0]]
			noneClassification = [featurePoint for i, featurePoint in enumerate(self.classifications) if self.classifications[i] != [1,0]]

			print len(remissionFeatures)
			print len(noneFeatures)

			print 'class1 accuracy is: ', sess.run(accuracy, feed_dict= {x: np.array(remissionFeatures).reshape(-1, self.n_chunks, self.chunk_size), self.y: remissionClassification})
			print 'class0 accuracy is: ', sess.run(accuracy, feed_dict= {x: np.array(noneFeatures).reshape(-1, self.n_chunks, self.chunk_size), self.y: noneClassification})





		self.loss = []
		prediction = self.recurrent_neural_network(x)
		print prediction
		cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction, self.y))

		correct_pred = tf.equal(tf.argmax(prediction,1), tf.argmax(self.y,1))
		accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


		optimizer = tf.train.AdamOptimizer(learning_rate = self.learning_rate).minimize(cost)



		with tf.Session() as sess:
			sess.run(tf.initialize_all_variables())

			for epoch in range(self.num_Epochs):
				epoch_loss = 0
				print 'epoch num is: ', epoch

				#basically a really bad way to go about pushing all of the datapoints through the RNN
				i = 0
				while i < len(self.featureSet)-4:
					start = i
					end = i + self.batch_size

					# print "start is: ", start, " end is: ", end
					batch_x = np.array([self.featureSet[start:end]])
					batch_y = np.array(self.classifications[start:end])
					# print batch_x.shape
					# print np.array([self.featureSet]).shape
					# print batch_y


					batch_x = batch_x.reshape((self.batch_size, self.n_chunks, self.chunk_size))


					_,c = sess.run([optimizer, cost], feed_dict = {x: batch_x, self.y: batch_y})
					epoch_loss += c
					i += 1
					acc = sess.run(accuracy, feed_dict={self.x: batch_x, self.y: batch_y})


					# print "i is: ", i
				self.loss.append(epoch_loss)

				print 'Epoch is: ', epoch, 'completed out of: ', self.num_Epochs, ' loss is: ', epoch_loss

			correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(self.classifications, 1))

			currAcc = tf.reduce_mean(tf.cast(correct, 'float'))


			test_neural_network()


	def writeLoss(self):
		filename = os.path.join(self.Dir, LOSS_FILE)
		file = open(filename, 'w')
		writer = csv.writer(file, quoting = csv.QUOTE_ALL)
		writer.writerow(self.loss)


	def getActualClassification(self):
		actualClassification = []
		counter = 0
		for i, classPoint in enumerate(self.classSet):
			classification = [1, 0] if classPoint.condition == "relapse" else [0,1]
			actualClassification.append(classification)
		print actualClassification
		return actualClassification





def main():
	print sys.version
	dataSet = ld.importDataset()
	currSet = dataSet[:]
	cleanedSet = baseline.cleanDataset(currSet)

	# print currSet

	runLSTM(cleanedSet, currSet)

if __name__ == "__main__":
	main()



