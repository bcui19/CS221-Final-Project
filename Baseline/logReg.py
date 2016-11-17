from sklearn import linear_model

import sys 
sys.path.append("./../Implementation/")

import svm



import loadData as ld
import baseline
import numpy as np
import copy
import math
from sklearn.model_selection import KFold, cross_val_score


from multiprocessing import Pool 
from multiprocessing import Process

class runLogistic(svm.runSVM):
	def __init__(self, dataset, classSet):
		self.dataset = dataset
		self.classSet = classSet
		self.splitDataset(10)


	def runClassification(self, train, test, foldNum, train_index, test_index):
		classificationTable = self.getActualClassification(train_index, False)

		clf = linear_model.LogisticRegression()
		clf.fit(train, classificationTable)

		predicted = clf.predict(test)
		self.prediction = clf.predict(test)
		print "predicted type is: ", type(self.prediction)

		self.getActualClassification(test_index, True)


def main():
	cleanedSet, classSet = svm.loadAndClean()
	runLogistic(cleanedSet, classSet)

if __name__ == "__main__":
	main()