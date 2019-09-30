import Utils
from decisionTree import decisionTree as dt
from randomForest import randomForest as rf
import time

#datasets/phpOkU53r.csv
#datasets/dadosBenchmark_validacaoAlgoritmoAD.csv
#datasets/dataset_31_credit-g.csv
#datasets/dataset_191_wine.csv

def main():
	#Getting time at the beginning of the execution
	start_time = time.time()

	data, data_desc = Utils.get_data_from_csv("datasets/dataset_191_wine.csv", ",")

	accuracies = Utils.cross_validation(data,data_desc, 10, 10, 2)

	#taking time at the end of the execution
	print("--- Cross validation executed in %s seconds ---" % (time.time() - start_time))

if __name__ == "__main__":
	main()
