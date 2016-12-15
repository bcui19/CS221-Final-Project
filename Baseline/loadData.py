import os
import csv

# DATAPATH = "./../Datasets/Kidney Cancer Dataset/GSE16449_series_matrix_filtered.csv"
DATAPATH = "./../Datasets/Remission Dataset/GSE13576_series_matrix_heavilyFiltered.csv"

class bioStructure:
	def __init__(self, patientName):
		self.patientName = patientName
		self.featureDict = {}

	#defines the condition for the class
	def defineCondition(self, condition):
		self.condition = condition

	#adds a key value to a class dictionary
	def addToDict(self, featureName, featureVal):
		self.featureDict[featureName] = featureVal


	def printGenes(self):
		# print self.featureDict
		return self.featureDict

def importDataset():
	#adds fetures to each class
	def addFeature(row):
		featureName = row[0]
		for i in range(1, len(row)):
			currStruct = sampleList[i-1]
			try:
				currStruct.addToDict(featureName, float(row[i]))
			except ValueError as e:
				currStruct.addToDict(featureName, 0) #Idk if this actually works


	sampleList = []
	DIR = os.path.dirname(__file__) 
	inputFile = os.path.join(DIR, DATAPATH)
	with open(inputFile, 'r') as csvFile:
		bioReader = csv.reader(csvFile, delimiter = ',')
		counter = 0
		for row in bioReader:
			#class initialization 
			if counter < 2:
				if counter == 0: 
					for i in range(1, len(row)):
						currStruct = bioStructure(row[i])
						sampleList.append(currStruct)
				if counter == 1: 
					for i in range(len(row)-1):
						currStruct = sampleList[i]
						currStruct.defineCondition(row[i+1])
				counter += 1
				continue
			
			addFeature(row)
			counter += 1
	return sampleList

def main():
	returnList = importDataset()
	print returnList[0]

# if __name__ == "__main__":
	# main()