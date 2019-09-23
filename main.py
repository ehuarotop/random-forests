import Utils
from decisionTree import decisionTree as dt

#datasets/phpOkU53r.csv
#datasets/dadosBenchmark_validacaoAlgoritmoAD.csv
#datasets/dataset_31_credit-g.csv
#datasets/dataset_191_wine.csv

def main():
	data, data_desc = Utils.get_data_from_csv("datasets/dataset_191_wine.csv", ",")
	det = dt(data, data_desc)
	root_node = det.generateDecisionTree()
	det.renderDecisionTree(root_node)

if __name__ == "__main__":
	main()