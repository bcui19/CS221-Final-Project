import os
import csv
import scipy.stats as stats

if __name__ == "__main__":
    f = open('features.txt', 'r')
    usedFeatures = []
    for line in f:
        usedFeatures.append(line.rstrip())
    #print usedFeatures
    f = open('GSE13576_series_matrix.csv', 'r')
    for line in f:
        if line.split(',')[0] in usedFeatures or line.split(',')[0] == 'ID_REF' or line.split(',')[0] == '!Sample_source_name_ch1':
            print line.rstrip()
        #else:
        #    print line.split(',')[0]

