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

females_list=data[0::,4] == 'female'
males_list=data[0::,4] != 'female'

data_male = data[males_list,1].astype(np.float)		#survived column
data_female = data[females_list,1].astype(np.float) #survived column

#male_onboard = np.size(data_male)
#female_onboard = np.size(data_female)

prop_male_surv = np.sum(data_male)/np.size(data_male)
prop_female_surv = np.sum(data_female)/np.size(data_female)

print('Males Survived Proportion: ', prop_male_surv)
print('Females Survived Proportion: ', prop_female_surv)


#gender class fare model 

fare_ceiling = 40

data[ data[0::,9].astype(np.float) >= fare_ceiling,9] = fare_ceiling - 1.0 #limits everything to the fare_ceiling

fare_bracket_size = 10
number_of_price_brackets = round(fare_ceiling / fare_bracket_size)

number_of_classes = 3
number_of_classes = len(np.unique(data[0::,2]))		#better. len gives number of rows. 
													#normal size gives numb elements. could also do: np.size(X,0) for rows

survival_table = np.zeros((2,number_of_classes,number_of_price_brackets))

for i in range(number_of_classes):																#xrange renamed to range in python3
	for j in range(number_of_price_brackets):

		women_only_stats = data[ (data[0::,4] == 'female') & 
								 (data[0::,2].astype(np.float) == i+1) &
								 (data[0::,9].astype(np.float) >= j*fare_bracket_size) &
								 (data[0::,9].astype(np.float) < (j+1)*fare_bracket_size)
								 ,1] 															#survival column only


		men_only_stats = 	data[(data[0::,4] != 'female') & 
								 (data[0::,2].astype(np.float) == i+1) &
								 (data[0::,9].astype(np.float) >= j*fare_bracket_size) &
								 (data[0::,9].astype(np.float) < (j+1)*fare_bracket_size)
								 ,1] 															#survival column only


		survival_table[0,i,j] = np.mean(women_only_stats.astype(np.float)) #same as sum/size
		survival_table[1,i,j] = np.mean(men_only_stats.astype(np.float)) #same as sum/size

survival_table[survival_table != survival_table] = 0	#sets all nan values to 0

print(survival_table)

survival_table[survival_table < 0.5] = 0
survival_table[survival_table >= 0.5] = 1

print(survival_table)

	
##test file - gender class fare model##

with open('test.csv') as test_file:
	test_reader = csv.reader(test_file)
	header2 = next(test_reader)

	print('header_test_file: ', header2)	

	#gender model output
	with open('tinus_gender_class_model.csv','w') as predict_file:
		predict_writer = csv.writer(predict_file)
		predict_writer.writerow(['PassengerId', 'Survived'])

		for row in test_reader:
			for j in range(number_of_price_brackets):
				try:
					row[8]=float(row[8])	#some passengers have no fare data. So try to make a float. If fail... no data.
				except:
					bin_fare = 3 - float(row[1])	#if no fare info, bin according to class. first class will generally have a higher fare. 
					break
				if row[8] > fare_ceiling:
					bin_fare = number_of_price_brackets-1
					break
				if row[8] >= j * fare_bracket_size and row[8] < (j+1) * fare_bracket_size:
					bin_fare = j
					break
	
			#fare will now be binned.

			if row[3] == "female": #gender column
			 	predict_writer.writerow([row[0], '%d' % int(survival_table[0,float(row[1])-1,bin_fare])])
			else:
			 	predict_writer.writerow([row[0], '%d' % int(survival_table[1,float(row[1])-1,bin_fare])])	