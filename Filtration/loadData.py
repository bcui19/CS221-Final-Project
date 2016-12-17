import os
import csv
import scipy.stats as stats

DATAPATH = "GSE13576_series_matrix.csv"

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


	def printShit(self):
		print self.featureDict

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
			# if counter > 5:
				# break
			counter += 1
	return sampleList

def cleanDataset(dataList):
#	normalize = MinMaxScaler(feature_range=(-1.0,1.0))

	dataMatrix = []
	for currClass in dataList:
		featureList = []
		currDict = currClass.featureDict
		for key in currDict:
			featureList.append(currDict[key])
#		dataMatrix.append(normalize.fit_transform(featureList)[:]) # need to normalize to make sure everything is legit
		dataMatrix.append(featureList[:])
	return dataMatrix

def outputFiltered(dataset):
    print "filtering"
    cancerIndices = []
    normalIndices = []
    for i in range(len(dataset)):
        if dataset[i].condition == "relapse":
            cancerIndices.append(i)
        else:
            normalIndices.append(i)
    for k in dataset[1].featureDict.keys():
        cancerExpressions = [dataset[i].featureDict[k] for i in cancerIndices]
        normalExpressions = [dataset[i].featureDict[k] for i in normalIndices]
        t_stat, p_val = stats.ttest_ind(cancerExpressions, normalExpressions, equal_var=False)
        if p_val <= 0.001 and (sum(cancerExpressions)/sum(normalExpressions) > 5 or sum(normalExpressions)/sum(cancerExpressions) > 5):
            print k

def main():
	dataset = importDataset()
        for i in range(len(dataset)):
            row = dataset[i].featureDict
            s = sum(row.values())
            for v in row.keys():
               row[v] = row[v]/s * 10000
            dataset[i].featureDict = row
        outputFiltered(dataset)
#	cleanedSet = cleanDataset(dataset)
	#print cleanedSet




if __name__ == "__main__":
	main()
