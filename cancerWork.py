#preprocessing

import csv

# read data

import csv
from scipy import stats

with open("./data/cancer_data.txt", 'rb') as f:
    reader = csv.reader(f)
    raw_data = list(reader)

# labels: Cancer, Gender, Age, Weight, Height, Job
labels = [float(x[0]) for x in raw_data]
ages = [float(x[2]) for x in raw_data]
weights = [float(x[3]) for x in raw_data]
heights = [float(x[4]) for x in raw_data]
#box cox test
#age
_,lambdaAge = stats.boxcox(ages)
_,lambdaWeight = stats.boxcox(weights)
_,lambdaHeight = stats.boxcox(heights)

print lambdaAge,lambdaHeight,lambdaWeight
print sum(labels)