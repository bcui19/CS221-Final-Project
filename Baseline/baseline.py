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


def logRunHelper(logReg, train, test, foldNum, train_index, test_index):
	logReg.runClassification(train, test, foldNum, train_index, test_index)


class logisticRegression:
	def __init__(self, inputDataset, classDataset):
		self.classData = classDataset
		self.dataSet = inputDataset
		self.learningRate = 0.00001
		self.epochMax = 2
		self.splitDataset(10)

	def splitDataset(self, n_splits):
		self.classList = [[0,0]*n_splits]
		k_fold = KFold(n_splits = n_splits)

		self.nWorkers = 1
		pool = Pool(processes = self.nWorkers)

		count = 0 
		resuDict = {}
		for train_index,test_index in k_fold.split(self.dataSet):
			train = [self.dataSet[i] for i in train_index]
			# print "train size is: ", len(self.train)
			test = [self.dataSet[i] for i in test_index]
			# print "test size is: ", len(self.test)
			tempResult = pool.apply_async(logRunHelper, args = (self, train, test, count, train_index, test_index))

			count += 1
			resuDict[count] = tempResult

		for resu in resuDict:
			resuDict[resu].get()

		pool.close()

		self.totList = [sum([int(term==0) for term in self.actualClassification], sum([int(term==1) for term in self.actualClassification]))]
		
		print "probability this is correct is: ", float(sum(self.classList))/len(self.dataSet)
		print "false positive rate is: ", float(self.totList[0] - sum([self.classList[i][0] for i in range(len(self.classList))]))/self.totList[0], " number of positive datasets is: ", self.totList[0]
		print "false negative rate is: ", float(self.totList[1] - sum([self.classList[i][1] for i in range(len(self.classList))]))/self.totList[1], "number of negative datsets is: ", self.totList[1]

	def runClassification(self, train, test, foldNum, train_index, test_index):
		beta = self.runLogReg(self.learningRate, self.epochMax, train_index, train)
		self.actualClassification = []
		classificationTable = self.classifyData(test, beta)
		self.getActualClassification(self.classData, test)
		
		tupleResu = self.getComparison()
		self.classList[foldNum][0] += tupleResu[0]
		self.classList[foldNum][0] += tupleResu[1]
			# return (totCorr0, totCorr1, tot0, tot1)


	def classifyData(self, dataSet, beta):
		classificationTable = []
		for dataPoint in dataSet:
			probability = self.calcLogVal(dataPoint, beta, 0)
			if probability > 0.5:
				classificationTable.append(1)
			else: 
				classificationTable.append(0)
		return classificationTable

	def getActualClassification(self, classDataset, test_index):
		actualClassification = []
		for i, classPoint in enumerate(classDataset):
			if i in test_index:
				classification = 1 if classPoint.condition == "Renal Clear Cell Carcinoma" else 0
			actualClassification.append(classification)
		return actualClassification

	def runLogReg(self, learningRate, epochMax, train_index, train):
		def resetGradient():
			return [0] * len(train[0])
		beta = [0] * len(train[0])
		gradient = beta[:]
		for epochNum in range(epochMax):
			gradient = resetGradient()
			print "epochNum is: ", epochNum
			print "number of features is: ", len(train[0])
			

			for i, index in enumerate(train_index):

				logisticValue = self.calcLogVal(train[i], beta, 0)
				# print "after logisticValue"
				resultantClassification = 1 if self.classData[index].condition == "Renal Clear Cell Carcinoma" else 0
				# print "after classification"

				for j in range(len(train[0])):
					if j == 0:
						gradient[j] += (resultantClassification - logisticValue)
						continue
					# print "len of gradient is: ", len(gradient)
					# print "len of train is: ", len(train[index])
					gradient[j] += train[i][j-1]*(resultantClassification - logisticValue)
				print "after gradient update"
			print "after training"

			for update in range(len(beta)):
				beta[update] += learningRate * gradient[update]
			print "updating beta"
		
		return beta[:]



	def calcLogVal(self, dataPoint, beta, z):
		z = sum([beta[i] * float(dataPoint[i]) for i in range(len(beta))])
		return 1/(1 + math.e**(-z))

	def getComparison(self, classificationTable, actualClassification, test):
		totCorr0 = 0
		totCorr1 = 0
		tot0 = 0
		tot1 = 0
		for i in range(len(test)):
			if classificationTable[i] == actualClassification[i] and classificationTable[i] == 0:
				totCorr0 += 1
			elif classificationTable[i] == actualClassification[i]:
				totCorr1 += 1
			if classificationTable[i] == 0:
				tot0 += 1
			else:
				tot1 += 1

		print "probability this is correct is: ", float(self.totCorr0 + self.totCorr1)/len(self.test)
		if tot0:
			print "false positive rate is: ", float(self.tot0 - self.totCorr0)/self.tot0, " number of positive datasets is: ", self.tot0
		if tot1:
			print "false negative rate is: ", float(self.tot1 - self.totCorr1)/self.tot1, "number of negative datsets is: ", self.tot1
		return (totCorr0, totCorr1, tot0, tot1)






if __name__ == "__main__":
	dataSet = ld.importDataset()
	currSet = dataSet[:]
	cleanedSet = cleanDataset(currSet)
	logisticRegression(cleanedSet, currSet)
