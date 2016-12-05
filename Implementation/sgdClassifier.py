from sklearn import linear_model
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, cross_val_score
from sklearn.model_selection import GridSearchCV 


import sys
sys.path.append("./../Baseline")

import loadData as ld
import baseline

import svm

#treating sgdClassifier as LASSO essentially
class runLasso(svm.runSVM):

	def runClassification(self, train, test, foldNum, train_index, test_index):
		classificationTable = self.getActualClassification(train_index, False)

		alphaVals = {'alpha':  [10.0 **a for a in range(-6,-2)]}

		print 'alpha vals are:', alphaVals

		clf = GridSearchCV(linear_model.SGDClassifier(loss = 'perceptron', penalty = 'elasticnet', n_iter = 100, class_weight = 'balanced', learning_rate = 'optimal', verbose = False), alphaVals,
			cv = 10, scoring = 'f1_macro')

		clf.fit(train, classificationTable)

		self.prediction = clf.predict(test)
		# print "predicted type is: ", type(self.prediction)

		self.getActualClassification(test_index, True)






def main():
	dataSet = ld.importDataset()
	currSet = dataSet[:]
	cleanedSet = baseline.cleanDataset(currSet)

	runLasso(cleanedSet, currSet)


if __name__ == "__main__":
	main()
