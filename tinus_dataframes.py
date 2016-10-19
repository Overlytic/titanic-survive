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
pyplot.show()

print '------------------'
print "df['Age'].dropna().hist(bins=16, range=(0,80), alpha = .5)"
print '------------------'

df['Age'].dropna().hist(bins=16, range=(0,80), alpha = .5)
pyplot.show()

