import pandas as pd
import json
import numpy as np
from randomForest import randomForest as rf

def get_data_from_csv(csv_file, sep):
	#reading dataset from csv
	data = pd.read_csv(csv_file, sep)

	#reading dataset descritpion from json file.
	with open(csv_file[:-4] + "-description.json") as desc_file:
		data_desc = json.loads(desc_file.read())

	return data, data_desc


def cross_validation(data, data_desc, n_trees, kfolds, n_cross_val=1):
	total_accuracies = []

	for cross_val in range(n_cross_val):
		print("Cross validation # " + str(cross_val+1))
		print("----------------------------")
		accuracies = []

		#Reordering data randomly
		data = data.reindex(np.random.permutation(data.index))

		#Getting the class column (assuming class column in the last column) 
		classe = list(data.columns.values)[-1]

		#Getting actual class instances
		class_instances = data[classe]

		#Getting unique class instances values from class column
		unique_class_values = class_instances.unique()

		data_classes = []

		for class_value in unique_class_values:
			data_class = data[data[classe] == class_value]
			data_classes.append(data_class)

		for kfold in range(kfolds):
			print("Working on kfold " + str(kfold+1) + " of " + str(kfolds))
			test_data = pd.DataFrame.from_records([])
			training_data = pd.DataFrame.from_records([])

			for data_class in data_classes:
				#Splitting data into 'kfolds' folds
				splitted_data_class = np.array_split(data_class,kfolds)

				test_data_class = splitted_data_class[kfold]
				training_data_class = pd.DataFrame.from_records([])

				for i in range(kfolds):
					if i != kfold:
						training_data_class = pd.concat([training_data_class, splitted_data_class[i]])

				test_data = pd.concat([test_data, test_data_class]).reset_index(drop=True)
				training_data = pd.concat([training_data, training_data_class]).reset_index(drop=True)

			#Getting random forest model
			rforest = rf(n_trees,training_data, data_desc)

			#fit the model
			rforest.fit()

			#getting predictions from the random forest.
			predictions = rforest.predict(test_data)

			#### Comparing predictions with actual class values from test_data
			#### to get mean accuracy and standard deviation
			correct_predictions = 0
			for index,row in test_data.iterrows():
				if row[-1] == predictions[index]:
					correct_predictions += 1

			accuracy = round(correct_predictions / test_data.shape[0], 3)
			accuracies.append(accuracy)
			total_accuracies.append(accuracy)

		print("Accuracy: " + str(np.mean(accuracies)) + " ± " + str(np.std(accuracies)))

	print("Total Accuracy: " + str(np.mean(total_accuracies)) + " ± " + str(np.std(total_accuracies)))

	return total_accuracies

		