import Utils
from decisionTree import decisionTree as dt
from randomForest import randomForest as rf
import time
import click
import os

#datasets/phpOkU53r.csv
#datasets/dadosBenchmark_validacaoAlgoritmoAD.csv
#datasets/dataset_31_credit-g.csv
#datasets/dataset_191_wine.csv

@click.command()
@click.option('--dataset', default='1',  type=click.Choice(['1', '2', '3','4'], case_sensitive=False),
	help=''' 1:dadosBenchmark_validacaoAlgoritmoAD.csv
2:phpOkU53r.csv
3:dataset_31_credit-g.csv
4:dataset_191_wine.csv''')
@click.option('--ntree', required=True, type=int, help='Number of decision Trees to generate random forest')
@click.option('--kfolds', required=True, type=int, help='Number of folds to divide the data for cross validation')
@click.option('--repeat_cv', required=True, type=int, help='Number of cross validation to be executed (repeated cross validation)')
def main(dataset, ntree, kfolds, repeat_cv):
	#Getting time at the beginning of the execution
	start_time = time.time()

	#Declaring all the dataset available to test
	datasets = ['datasets/dadosBenchmark_validacaoAlgoritmoAD.csv', 'datasets/phpOkU53r.csv',
				'datasets/dataset_31_credit-g.csv', 'datasets/dataset_191_wine.csv']

	dataset_path = datasets[int(dataset) - 1]

	print('Using dataset: ' + dataset_path)
	print('Using ' + str(ntree) + ' decision Trees in each random forest (each cross validation)')
	print('Using ' + str(kfolds) + ' folds for each cross validation')
	print('Executing '+ str(repeat_cv) + ' repeated cross validations')

	#Getting data from csv
	if dataset == 1:
		data, data_desc = Utils.get_data_from_csv(dataset_path, ";")
	else:
		data, data_desc = Utils.get_data_from_csv(dataset_path, ",")

	#getting dataset filename
	dataset_name = os.path.split(dataset_path)[1]

	#Performing repeated cross validation
	Utils.cross_validation(dataset_name, data,data_desc, ntree, kfolds, repeat_cv)

	#taking time at the end of the execution
	print("--- Cross validation executed in %s seconds ---" % (time.time() - start_time))

if __name__ == "__main__":
	main()
