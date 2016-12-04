#all necessary tensorflow imports
import numpy as np
import tensorflow as tf
from tensorflow.models.rnn import rnn, rnn_cell


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
		self.num_Epochs = 50
		self.n_classes = 2 #number of possible classifications
		self.batch_size = 1 #pushing one training point through at a time
		self.chunk_size = 2445
		self.n_chunks = 1
		self.rnn_size = 128

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
		self.loss = []
		prediction = self.recurrent_neural_network(x)
		print prediction
		cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction, self.y))

		optimizer = tf.train.AdamOptimizer().minimize(cost)



		with tf.Session() as sess:
			sess.run(tf.initialize_all_variables())

			for epoch in range(self.num_Epochs):
				epoch_loss = 0
				print 'epoch num is: ', epoch

				#basically a really bad way to go about 
				i = 0
				while i < len(self.featureSet):
					start = i
					end = i + self.batch_size

					batch_x = np.array([self.featureSet[start:end]])
					batch_y = np.array(self.classifications[start:end])


					batch_x = batch_x.reshape((self.batch_size, self.n_chunks, self.chunk_size))


					_,c = sess.run([optimizer, cost], feed_dict = {x: batch_x, self.y: batch_y})
					epoch_loss += c
					i += 1
					# print "i is: ", i
				self.loss.append(epoch_loss)

				print 'Epoch is: ', epoch, 'completed out of: ', self.num_Epochs, ' loss is: ', epoch_loss

			correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(self.classifications, 1))

			accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

			print 'Accuracy is: ', accuracy.eval({x: np.array(self.featureSet).reshape(-1, self.n_chunks, self.chunk_size), self.y: self.classifications})


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



