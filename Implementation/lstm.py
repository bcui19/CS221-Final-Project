#all necessary tensorflow imports
import numpy as np
import tensorflow as tf

from tensorflow.python.ops.rnn_cell import BasicLSTMCell

from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import rnn


#import modules from sklearn
from sklearn.utils import shuffle 
from sklearn.model_selection import KFold, cross_val_score


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
		self.prepData()


		self.initLSTM()
		self.runClassification(10)
		self.writeLoss(10)




	def initLSTM(self):
		self.num_Epochs = 50 # number of training iterations
		self.n_classes = 2 #number of possible classifications
		self.batch_size = 3 #pushing one training point through at a time
		self.chunk_size = len(self.featureSet[0])/31
		self.n_chunks = 31
		self.rnn_size = 128
		self.learning_rate = 0.001

		self.x = tf.placeholder('float', [None, self.n_chunks, self.chunk_size])
		self.y = tf.placeholder('float', [None, self.n_classes])


	def runClassification(self, n_splits):
		k_fold = KFold(n_splits = n_splits)

		self.lossDict = {}

		count = 0
		self.classificationsList = [0] *4
		prediction = self.recurrent_neural_network(self.x) #need to initizliae out here

		for train_index, test_index in k_fold.split(self.featureSet):


			self.train = [self.featureSet[i] for i in train_index]
			self.trainGoal = [self.classifications[i] for i in train_index]
			self.test = [self.featureSet[i] for i in test_index]
			self.testGoal = [self.classifications[i] for i in test_index]

			self.train_neural_network(self.x, prediction)

			self.lossDict[count] = self.loss[:]

			count += 1




	def prepData(self):
		self.featureSet, self.classifications = shuffle(self.featureSet, self.classifications, random_state = 0)
		print self.classifications


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


	def train_neural_network(self, x, prediction):
		def test_neural_network():

			remissionFeatures = [featurePoint for i, featurePoint in enumerate(self.test) if self.testGoal[i] == [1,0]]
			remissionClassification = [classpoint for i, classpoint in enumerate(self.testGoal) if self.testGoal[i] == [1,0]]
			noneFeatures = [featurePoint for i, featurePoint in enumerate(self.test) if self.testGoal[i] != [1,0]]
			noneClassification = [featurePoint for i, featurePoint in enumerate(self.testGoal) if self.testGoal[i] != [1,0]]

			# print len(remissionFeatures)
			# print len(noneFeatures)

			self.classificationsList[0] += len(noneFeatures)
			self.classificationsList[1] += len(remissionFeatures)

			class1Acc = sess.run(accuracy, feed_dict= {x: np.array(remissionFeatures).reshape(-1, self.n_chunks, self.chunk_size), self.y: remissionClassification})
			class0Acc =  sess.run(accuracy, feed_dict= {x: np.array(noneFeatures).reshape(-1, self.n_chunks, self.chunk_size), self.y: noneClassification})

			self.classificationsList[2] += len(noneFeatures)*class0Acc
			self.classificationsList[3] += len(remissionFeatures)*class1Acc

			print "current class 1 accuracy is: ", class1Acc
			print "current class 0 accuracy is: ", class0Acc


		self.loss = []
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

					batch_x = np.array([self.featureSet[start:end]]) #gets the features
					batch_y = np.array(self.classifications[start:end]) #gets the classifications

					batch_x = batch_x.reshape((self.batch_size, self.n_chunks, self.chunk_size))
					# print "size of batch is: ", batch_x.shape


					_,c, acc = sess.run([optimizer, cost, accuracy], feed_dict = {x: batch_x, self.y: batch_y})
					epoch_loss += c
					i += self.batch_size
					# acc = sess.run(accuracy, feed_dict={self.x: batch_x, self.y: batch_y})

					if i + self.batch_size > len(self.featureSet) -4:
						print "minibatch accuracy is: ", acc, " and loss is: ", c

					# print "i is: ", i
				self.loss.append(epoch_loss)

				print 'Epoch is: ', epoch, 'completed out of: ', self.num_Epochs, ' loss is: ', epoch_loss

			correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(self.classifications, 1))

			currAcc = tf.reduce_mean(tf.cast(correct, 'float'))


			test_neural_network()



	def writeLoss(self, num_folds):
		totalLoss = [sum([self.lossDict[i][j] for i in self.lossDict])/10 for j in range(len(self.lossDict[0]))]
		print self.lossDict
		filename = os.path.join(self.Dir, LOSS_FILE)
		file = open(filename, 'w')
		writer = csv.writer(file, quoting = csv.QUOTE_ALL)
		writer.writerow(totalLoss)

		print totalLoss

		print "Class 0 classification accuracy is: ", float(self.classificationsList[2])/self.classificationsList[0], " number correct is: ", self.classificationsList[2], " number total is: ", self.classificationsList[0]
		print "Class 1 classification accuracy is: ", float(self.classificationsList[3])/self.classificationsList[1], " number correct is: ", self.classificationsList[3], " number total is: ", self.classificationsList[1]




	def getActualClassification(self):
		actualClassification = []
		counter = 0
		for i, classPoint in enumerate(self.classSet):
			classification = [1, 0] if classPoint.condition == "relapse" else [0,1]
			actualClassification.append(classification)
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



