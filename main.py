import Utils
from decisionTree import decisionTree as dt

#datasets/phpOkU53r.csv
#datasets/dadosBenchmark_validacaoAlgoritmoAD.csv
#datasets/dataset_31_credit-g.csv
#datasets/dataset_191_wine.csv

def main():
	data, data_desc = Utils.get_data_from_csv("datasets/dadosBenchmark_validacaoAlgoritmoAD.csv", ";")
	det = dt(data, data_desc)
	root_node = det.generateDecisionTree()
	det.renderDecisionTree(root_node)
	#prediction = det.predict_instance(data.iloc[0])

if __name__ == "__main__":
	main()