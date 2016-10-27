'''
Forces the train/test dataset to do a ~70:30 split


'''

import csv 
import os
import loadData as ld
import math
import random 


def cleanDataset(dataList):
	dataMatrix = []
	for currClass in dataList:
		featureList = []
		currDict = currClass.featureDict
		for key in currDict:
			featureList.append(currDict[key])
		dataMatrix.append(featureList[:])
	return dataMatrix


class logisticRegression:
	def __init__(self, inputDataset, classDataset):
		self.classData = classDataset
		self.dataSet = inputDataset
		self.learningRate = 0.01
		self.epochMax = 10
		self.splitDataset()
		self.runClassification()

	def splitDataset(self):
		self.train = []
		self.test = []
		self.testClass = []
		self.validation = []
		self.validationClass = []
		for i in range(len(self.dataSet)):
			randomGenerated = random.randint(0, 10)
			if randomGenerated >= 7:
				self.test.append(self.dataSet[i][:])
				self.testClass.append(self.classData[i])
			elif randomGenerated >=1:
				self.train.append(self.dataSet[i][:])
			else: 
				self.validation.append(self.dataSet[i][:])
				self.validationClass.append(self.classData[i])
		print "test length is: ", len(self.test)
		print "train length is: ", len(self.train)



	def runClassification(self):
		self.runLogReg(self.learningRate, self.epochMax)
		self.classificationTable = []
		self.actualClassification = []
		self.classifyData(self.test)
		self.getActualClassification(self.testClass)
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
		totCorr0 = 0
		totCorr1 = 0
		tot0 = 0
		tot1 = 0
		for i in range(len(self.test)):
			if self.classificationTable[i] == self.actualClassification[i] and self.classificationTable[i] == 0:
				totCorr0 += 1
			elif self.classificationTable[i] == self.actualClassification[i]:
				totCorr1 += 1
			if self.classificationTable[i] == 0:
				tot0 += 1
			else:
				tot1 += 1
		print "probability this is correct is: ", float(totCorr0 + totCorr1)/len(self.test)
		print "false positive rate is: ", float(tot0 - totCorr0)/tot0
		print "false negative rate is: ", float(tot1 - totCorr1)/tot1






if __name__ == "__main__":
	dataSet = ld.importDataset()
	currSet = dataSet[:]
	cleanedSet = cleanDataset(currSet)
	logisticRegression(cleanedSet, currSet)
