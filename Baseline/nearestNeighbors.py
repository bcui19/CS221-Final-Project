from sklearn.neighbors import NearestNeighbors
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



def main():
	cleanedSet, classSet = svm.loadAndClean()



if __name__ == "__main__":
	main()