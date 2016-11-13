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

class runSVM():
	def __init__(self, dataset, classSet):
		self.dataset = dataset
		self.classSet = classSet
		self.splitDataset(10)

	def splitDataset(self, n_splits):
		k_fold = KFold(n_splits = n_splits)
		self.nWorkers = 10
		pool = Pool(processes = self.nWorkers)


		resuDict = {}
		count = 0
		for train_index, test_index in k_fold.split(self.dataset):
			train = [self.dataset[i] for i in train_index]
			test = [self.dataset[i] for i in test_index]

			self.runClassification(train, test, count, train_index, test_index)

			count += 1


	def runClassification(self, train, test, foldNum, train_index, test_index):
		classificationTable = self.getActualClassification(train_index)

		clf = svm.SVC()
		clf.fit(train, classificationTable)

		print clf.predict(test)





	def getActualClassification(self, test_index):
		actualClassification = []
		for i, classPoint in enumerate(self.classSet):
			if i in test_index:
				classification = 1 if classPoint.condition == "Renal Clear Cell Carcinoma" else 0
				actualClassification.append(classification)
		return actualClassification





def loadAndClean():
	dataset = ld.importDataset()
	currSet = dataset[:]
	cleanedSet = baseline.cleanDataset(currSet)
	return cleanedSet, currSet

def main():
	cleanedSet, classSet = loadAndClean()
	runSVM(cleanedSet, classSet)



if __name__ == "__main__":
	main()
