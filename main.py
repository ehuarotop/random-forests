import Utils
from decisionTree import decisionTree as dt
from randomForest import randomForest as rf

#datasets/phpOkU53r.csv
#datasets/dadosBenchmark_validacaoAlgoritmoAD.csv
#datasets/dataset_31_credit-g.csv
#datasets/dataset_191_wine.csv

def main():
	data, data_desc = Utils.get_data_from_csv("datasets/dadosBenchmark_validacaoAlgoritmoAD.csv", ";")
	det = dt(data, data_desc)

	#Initializing random forest
	rforest = rf(5,data, data_desc)
	#Training the model
	rforest.fit()

	#Performing some predictions
	prediction = rforest.predict(data.iloc[[0,1]])
	#print(data.iloc[[0,1]])
	print(prediction)

if __name__ == "__main__":
	main()
