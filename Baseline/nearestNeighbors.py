from sklearn.neighbors import NearestNeighbors as nn
import numpy as np

import loadData as ld
import baseline
import numpy as np
import copy
import math
from sklearn.model_selection import KFold, cross_val_score



import sys 
sys.path.append("./../Implementation/")

import svm

class nearestNeighbors(svm.runSVM):
	def __init__(self, trainDataset, trainClassSet):#, validationDataset, validationClass):
		self.dataset = trainDataset
		self.classSet = trainClassSet

		self.splitDataset(10) #splitting by 10 folds

	def runClassification(self, train, test, foldNum, train_index, test_index):
		def getPrediction():
			self.prediction = []
			for neighbors in indicies:
				currDict = {1: 0, 0: 0}
				neighborClass = [1 if self.classSet[train_index[neighbor]].condition == 'relapse' else 0 for neighbor in neighbors]
				for classification in neighborClass:
					currDict[classification] += 1
				self.prediction.append(1 if currDict[1] > currDict[0] else 0)



		# classificationTable = self.getActualClassification(train_index, False)

		nbrs = nn(n_neighbors=5, algorithm = 'ball_tree').fit(train)
		distances, indicies = nbrs.kneighbors(test)

		getPrediction()
		print "predicted list is: ", self.prediction
		print "predicted type is: ", type(self.prediction)

		self.getActualClassification(test_index, True)

		for patient in self.classSet:
			print patient.condition == svm.CANCER_TAG




def main():
	cleanedSet, classSet = svm.loadAndClean()

	nearestNeighbors(cleanedSet, classSet)


if __name__ == "__main__":
	main()