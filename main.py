import Utils
from decisionTree import decisionTree as dt

def main():
	data = Utils.get_data_from_csv("dadosBenchmark_validacaoAlgoritmoAD.csv")
	det = dt(data)
	root_node = det.generateDecisionTree()
	det.renderDecisionTree(root_node)
	#det.exportDecisionTreeToPNG(root_node,'decisionTree.png')
	#det.getInformationGain('Joga', 'Tempo')
	#det.getInformationGain('Joga', 'Temperatura')
	#det.getInformationGain('Joga', 'Umidade')
	#det.getInformationGain('Joga', 'Ventoso')

if __name__ == "__main__":
	main()