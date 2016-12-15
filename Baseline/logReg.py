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

NUm_GENES = 20
GENEFILE = "./../GeneList.txt"

class runLogistic(svm.runSVM):
	def __init__(self, dataset, classSet):
		self.dataset = dataset
		self.classSet = classSet

		self.geneSet = {}
		self.count = 0


		self.splitDataset(10)

		self.filterGenes()


	def filterGenes(self):
		geneSet = set()
		for key in self.geneSet:
			for gene in self.geneSet[key]:
				geneSet.add(gene[1])

		# print geneSet
		# for geneIndex in geneSet:
		geneKeys = self.classSet[0].printGenes().keys()
		with open(GENEFILE, "w") as f:
			f.write("\n".join([geneKeys[i] for i in geneSet]))






	def runClassification(self, train, test, foldNum, train_index, test_index):
		classificationTable = self.getActualClassification(train_index, False)

		clf = linear_model.LogisticRegression(class_weight = 'balanced')
		clf.fit(train, classificationTable)

		predicted = clf.predict(test)
		self.prediction = clf.predict(test)
		print "predicted type is: ", type(self.prediction)

		self.getActualClassification(test_index, True)

		self.genes = clf.coef_
		self.getImportantGenes()
		self.count += 1


	def getImportantGenes(self):
		maxList = []
		minVal = -1000
		for i,gene in enumerate(self.genes[0]):
			# print gene
			if gene > minVal:
				if len(maxList) > NUm_GENES:
					maxList.pop()

				maxList.append((abs(gene), i))
				maxList = sorted(maxList, key = lambda gene: gene[0])
				minVal = maxList[0][0]

		self.geneSet[self.count] = maxList
 




def main():
	cleanedSet, classSet = svm.loadAndClean()
	runLogistic(cleanedSet, classSet)

if __name__ == "__main__":
	main()