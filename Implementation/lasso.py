from sklearn import linear_model
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, cross_val_score

import sys
sys.path.append("./../Baseline")

import loadData as ld
import baseline

import svm


class runLasso(svm.runSVM):

	def runClassification(self, train, test, foldNum, train_index, test_index):
		classificationTable = self.getActualClassification(train_index, False)
		self.alpha = 0.01
		clf = linear_model.Lasso(alpha=self.alpha, fit_intercept = True, max_iter = 100000)
		clf.fit(train, classificationTable)

		self.prediction = clf.predict(test)
		print "predicted type is: ", type(self.prediction)

		self.getActualClassification(test_index, True)






def main():
	dataSet = ld.importDataset()
	currSet = dataSet[:]
	cleanedSet = baseline.cleanDataset(currSet)

	runLasso(cleanedSet, currSet)


if __name__ == "__main__":
	main()
