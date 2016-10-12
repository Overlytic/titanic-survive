import numpy as np 
import csv as csv

# Import the data

data = []

with open('train.csv') as train_file:
	train_reader = csv.reader(train_file)

	header = next(train_reader)	#python2.7 train_reader.next()

	for row in train_reader:
		data.append(row)

	data = np.array(data)

print(header)
print(data[0]) #first row

print(data[-1])	#last row

print(data[0:2, 3])	#Name column entries 1-3. For all entries 0::

#print(data[0::, 4])	#Gender column all entries

num_passengers = np.size(data[0::,1].astype(np.float))
num_survived = np.sum(data[0::,1].astype(np.float))
perc_survived = num_survived / num_passengers

print('Passengers: ',num_passengers,' Survived: ', num_survived, '  Perc: ', perc_survived*100, '%')

males_list=data[0::,4] == 'male'
females_list=data[0::,4] == 'female'

data_male = data[males_list,1].astype(np.float)		#survived column
data_female = data[females_list,1].astype(np.float) #survived column

#male_onboard = np.size(data_male)
#female_onboard = np.size(data_female)

prop_male_surv = np.sum(data_male)/np.size(data_male)
prop_female_surv = np.sum(data_female)/np.size(data_female)

print('Males Survived Proportion: ', prop_male_surv)
print('Females Survived Proportion: ', prop_female_surv)


##test file - gender model##

with open('test.csv') as test_file:
	test_reader = csv.reader(test_file)
	header2 = next(test_reader)

	print('header_test_file: ', header2)	

	#gender model output
	with open('tinus_gender_model.csv','w') as predict_file:
		predict_writer = csv.writer(predict_file)
		predict_writer.writerow(['PassengerId', 'Survived'])

		for row in test_reader:
			 if row[3] == "male": #gender column
			 	predict_writer.writerow([row[0], '0'])
			 else:
			 	predict_writer.writerow([row[0], '1'])





