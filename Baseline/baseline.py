'''
Forces the train/test dataset to do a ~70:30 split


'''

import csv 
import os
import loadData as ld
import math
import random 
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import MinMaxScaler
from multiprocessing import Pool
from multiprocessing import Process


def cleanDataset(dataList):
	normalize = MinMaxScaler(feature_range=(-1.0,1.0))

	dataMatrix = []
	for currClass in dataList:
		featureList = []
		currDict = currClass.featureDict
		for key in currDict:
			featureList.append(currDict[key])
		dataMatrix.append(normalize.fit_transform(featureList)[:]) # need to normalize to make sure everything is legit
	return dataMatrix


class logisticRegression:
	def __init__(self, inputDataset, classDataset):
		self.classData = classDataset
		self.dataSet = inputDataset
		self.learningRate = 0.00001
		self.epochMax = 100
		self.splitDataset()

	def splitDataset(self):
		self.classList = [0,0]
		k_fold = KFold(n_splits = 10)

		self.nWorkers = 48
		pool = Pool(process = self.nWorkers)

		count = 0 
		for train_index,test_index in k_fold.split(self.dataSet):
			self.train = [self.dataSet[i] for i in train_index]
			# print "train size is: ", len(self.train)
			self.test = [self.dataSet[i] for i in test_index]
			# print "test size is: ", len(self.test)
			self.runClassification()
			self.classList[0] += self.totCorr0
			self.classList[1] += self.totCorr1


			

		self.totList = [sum([int(term==0) for term in self.actualClassification], sum([int(term==1) for term in self.actualClassification]))]
		
		print "probability this is correct is: ", float(sum(self.classList))/len(self.dataSet)
		print "false positive rate is: ", float(self.totList[0] - self.classList[0])/self.totList[0], " number of positive datasets is: ", self.totList[0]
		print "false negative rate is: ", float(self.totList[1] - self.classList[1])/self.totList[1], "number of negative datsets is: ", self.totList[1]

	def runClassification(self):
		self.runLogReg(self.learningRate, self.epochMax)
		self.classificationTable = []
		self.actualClassification = []
		self.classifyData(self.test)
		self.getActualClassification(self.classData)
		self.getComparison()

	def classifyData(self, dataSet):
		for dataPoint in dataSet:
			probability = self.calcLogVal(dataPoint, self.beta, 0)
			if probability > 0.5:
				self.classificationTable.append(1)
			else: 
				self.classificationTable.append(0)

	def getActualClassification(self, classDataset):
		for classPoint in classDataset:
			classification = 1 if classPoint.condition == "Renal Clear Cell Carcinoma" else 0
			self.actualClassification.append(classification)

	def runLogReg(self, learningRate, epochMax):
		def resetGradient():
			return [0] * len(self.train[0])
		beta = [0] * len(self.train[0])
		gradient = beta[:]
		for epochNum in range(epochMax):
			gradient = resetGradient()
			print "epochNum is: ", epochNum
			print "number of features is: ", len(self.train[0])
			
			for index in range(len(self.train)):

				logisticValue = self.calcLogVal(self.train[index], beta, 0)
				resultantClassification = 1 if self.classData[index].condition == "Renal Clear Cell Carcinoma" else 0

				for j in range(len(self.train[0])):
					if j == 0:
						gradient[j] += (resultantClassification - logisticValue)
						continue
					gradient[j] += self.train[index][j-1]*(resultantClassification - logisticValue)

			for update in range(len(beta)):
				beta[update] += learningRate * gradient[update]
		
		self.beta = beta[:]



	def calcLogVal(self, dataPoint, beta, z):
		z = sum(beta[i] * float(dataPoint[i]) for i in range(len(beta)))
		return 1/(1 + math.e**(-z))

	def getComparison(self):
		self.totCorr0 = 0
		self.totCorr1 = 0
		self.tot0 = 0
		self.tot1 = 0
		for i in range(len(self.test)):
			if self.classificationTable[i] == self.actualClassification[i] and self.classificationTable[i] == 0:
				self.totCorr0 += 1
			elif self.classificationTable[i] == self.actualClassification[i]:
				self.totCorr1 += 1
			if self.classificationTable[i] == 0:
				self.tot0 += 1
			else:
				self.tot1 += 1

		print "probability this is correct is: ", float(self.totCorr0 + self.totCorr1)/len(self.test)
		if self.tot0:
			print "false positive rate is: ", float(self.tot0 - self.totCorr0)/self.tot0, " number of positive datasets is: ", self.tot0
		if self.tot1:
			print "false negative rate is: ", float(self.tot1 - self.totCorr1)/self.tot1, "number of negative datsets is: ", self.tot1






if __name__ == "__main__":
	dataSet = ld.importDataset()
	currSet = dataSet[:]
	cleanedSet = cleanDataset(currSet)
	logisticRegression(cleanedSet, currSet)
