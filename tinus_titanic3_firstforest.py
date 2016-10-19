#Python 2.7

import pandas as pd
import numpy as np
from matplotlib import pyplot

#For .read_csv, always use header=0 when you know row 0 is the header row

df = pd.read_csv('train.csv', header=0)


print '------------------'
print 'df.head(3)'
print '------------------'
print df.head(3)

print '------------------'
print 'df.tail(3)'
print '------------------'
print df.tail(3)

print ""

print '------------------'
print 'df.dtypes'
print '------------------'
print df.dtypes

# run in the interpreter...

print '------------------'
print 'df.describe()'
print '------------------'
print df.describe()

print '------------------'
print 'df.info()'
print '------------------'
print df.info()

print '------------------'
print "df.['Age'][0:10]  or df.Age[0:10]"
print '------------------'
print df['Age'][0:10]

print '------------------'
print "df.Age[0:10].mean()"
print '------------------'
print df.Age[0:10].mean

print '------------------'
print "df[['Sex','Pclass','Age']]"
print '------------------'
print df[['Sex','Pclass','Age']]

print '------------------'
print "df[df['Age'] > 60][['Sex', 'Pclass','Age','Survived']]"
print '------------------'
print df[df['Age'] > 60][['Sex', 'Pclass','Age','Survived']]

print '------------------'
print "df['Age'].hist()"
print '------------------'
df['Age'].hist()
#pyplot.show()

print '------------------'
print "df['Age'].dropna().hist(bins=16, range=(0,80), alpha = .5)"
print '------------------'
df['Age'].dropna().hist(bins=16, range=(0,80), alpha = .5)
#pyplot.show()

# The Data cleaning Starts....

#create new Gender column... hard to work with string values male and female
print '------------------'
print "Create gender column with value 4 "
print '------------------'
df['Gender'] = 4
print df.Gender.describe()

#lambda... from lambda calculus... create an anonymous function at runtime. Here takes the first char
print '------------------'
print "Set gender column = to first char of Sex column (uppercase)"
print '------------------'
df['Gender'] = df['Sex'].map(lambda x: x[0].upper())
print df[['Sex','Gender']].head(3)

print '------------------'
print "Gender set to 0->female, 1->male"
print '------------------'
df['Gender'] = df['Sex'].map({'female': 0, 'male': 1}).astype(int)
print df['Gender'].describe()

#check the 64.7% males percentage
print '------------------'
print "Check the percentage males"
print '------------------'

print df[df['Sex']=='male']['Sex'].count() / (df['Sex'].count() * 1.0)


#Assign values for missing ages...assigning median per passenger class
print '------------------'
print "Median Ages to null Age values"
print '------------------'
median_ages = np.zeros((2,3))

for i in range(0,2):				#note the end point is not included
	for j in range(0,3):
		median_ages[i,j] = df[(df['Gender'] == i) & \
								(df['Pclass'] == j+1)]['Age'].dropna().median()

print median_ages

df['AgeFill'] = df['Age']

print df[df['Age'].isnull()][['Gender', 'Pclass', 'Age','AgeFill']].head(10)

for i in range(0,2):				#note the end point is not included
	for j in range(0,3):
		df.loc[(df.Age.isnull()) & (df.Gender == i) & (df.Pclass == j+1),'AgeFill'] = median_ages[i,j]
		#loc(row_indexer, column_indexer) function does label based location...actually gives you a warning.

print df[df['Age'].isnull()][['Gender', 'Pclass', 'Age','AgeFill']].head(10)

df['AgeIsNull'] = pd.isnull(df.Age).astype(int)

df['FamilySize'] = df['SibSp'] + df['Parch']

df['Age*Class'] = df.AgeFill * df.Pclass

#idea... try to pull more information from the Men

print '------------------'
print "Final datatypes"
print '------------------'
print df.dtypes	#must 1) determine non-numeric columns still left.... 2) convert to numpy.array


print '------------------'
print "Object/String datatypes (only) ... to drop"
print '------------------'
print df.dtypes[df.dtypes.map(lambda x: x=='object')]

df = df.drop(['Name','Sex','Ticket','Cabin','Embarked'], axis=1) #axis=1 means columns... 0 are rows..
df = df.drop(['Age'], axis=1)
df = df.drop(['PassengerId'], axis=1)


print '------------------'
print "Final dataframe"
print '------------------'

print df.describe()

train_data = df.values #sends back a numpy array

print '------------------'
print "Final Numpy array"
print '------------------'
print train_data


## Test Data ##

testdf = pd.read_csv('test.csv', header=0)

print '------------------'
print "Test Data"
print '------------------'

print testdf.dtypes
print testdf.describe()

#Fix testdf

testdf['Gender'] = testdf['Sex'].map({'female': 0, 'male': 1}).astype(int)

for i in range(0,2):				#note the end point is not included
	for j in range(0,3):
		testdf.loc[(testdf.Age.isnull()) & (testdf.Gender == i) & (testdf.Pclass == j+1),'AgeFill'] = /
			median_ages[i,j]

testdf['AgeIsNull'] = pd.isnull(testdf.Age).astype(int)
testdf['FamilySize'] = testdf['SibSp'] + testdf['Parch']
testdf['Age*Class'] = testdf.AgeFill * testdf.Pclass
testdf = testdf.drop(['Name','Sex','Ticket','Cabin','Embarked','Age','PassengerId'], axis=1)

print '------------------'
print "Cleaned Test Data"
print '------------------'

print testdf.describe()
test_data = testdf.values

#### Random Forest ####

print '------------------'
print "Random Forest"
print '------------------'

from sklearn.ensemble import RandomForestClassifier

#create random forest object... including param for fit
forest = RandomForestClassifier(n_estimators = 100)

# Fit the training data to the survived labels and create the decision trees
# first column is passenger id... second column is the survived column, rest is the passenger data
forest = forest.fit(train_data[0::,1::], train_data[0::,0])

# Take the same decision trees and run it on the test data
output = forest.predict(test_data)