from sklearn import svm

import sys 
sys.path.append("./../Baseline")


import loadData as ld
import baseline
import numpy as np
import copy
import math
from sklearn.model_selection import KFold, cross_val_score


from multiprocessing import Pool 
from multiprocessing import Process

CANCER_TAG = "relapse"


class runSVM():
	def __init__(self, dataset, classSet):
		self.dataset = dataset
		self.classSet = classSet
		self.splitDataset(10)

	def splitDataset(self, n_splits):
		self.classificationList = [0]*4 #first two are prediction second two indices are actual


		k_fold = KFold(n_splits = n_splits)
		self.nWorkers = 10
		pool = Pool(processes = self.nWorkers)


		resuDict = {}
		count = 0
		for train_index, test_index in k_fold.split(self.dataset):
			print test_index
			print "n_splits is: ", n_splits
			train = [self.dataset[i] for i in train_index]
			test = [self.dataset[i] for i in test_index]

			self.runClassification(train, test, count, train_index, test_index)

			count += 1

		try:
			print "total values predicted to be 0 are: ", self.classificationList[0], " total values actually 0 are: ", self.classificationList[2]
			print "accuracy for 0 is: ", self.classificationList[0]*1.0/self.classificationList[2]
			print "total values predicted to be 1 are: ", self.classificationList[1], " total values actually 1 are: ", self.classificationList[3]
			print "accuracy for 1 is: ", self.classificationList[1]*1.0/self.classificationList[3]
		except ZeroDivisionError:
			return


	def runClassification(self, train, test, foldNum, train_index, test_index):
		classificationTable = self.getActualClassification(train_index, False)

		clf = svm.SVC()
		clf.fit(train, classificationTable)

		self.prediction = clf.predict(test)
		print self.prediction
		print "predicted type is: ", type(self.prediction)

		self.getActualClassification(test_index, True)




	def getActualClassification(self, test_index, testBool):
		actualClassification = []
		counter = 0
		for i, classPoint in enumerate(self.classSet):
			if i in test_index:
				classification = 1 if classPoint.condition == CANCER_TAG else 0
				print classPoint.condition
				actualClassification.append(classification)
				if testBool:
					print "patient name is: ", classPoint.patientName
					print "classification is: ", classification
					print "actual classification is: ", self.prediction[counter]
					self.classificationList[classification] += 1 if self.prediction[counter] == classification else 0
					self.classificationList[2+classification] += 1
					counter += 1

		return actualClassification





def loadAndClean():
	dataset = ld.importDataset()
	currSet = dataset[:]
	cleanedSet = baseline.cleanDataset(currSet)
	print currSet
	return cleanedSet, currSet

def main():
	cleanedSet, classSet = loadAndClean()
	runSVM(cleanedSet, classSet)



if __name__ == "__main__":
	main()
