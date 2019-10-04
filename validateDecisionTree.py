import Utils
from decisionTree import decisionTree as dt

#Getting validation dataset and data description
data, desc_data = Utils.get_data_from_csv('datasets/dadosBenchmark_validacaoAlgoritmoAD.csv', ';')
#Creating decision Tree in verbose mode to print information gains and data divisions
det = dt(data,desc_data, 4, 'verbose')
#Generating the decision Tree
root_node = det.generateDecisionTree()
#Printing the resulting tree to the terminal
det.renderDecisionTree(root_node)
#Optional, exporting decision Tree to a PNG image.
det.exportDecisionTreeToPNG(root_node, 'decisionTree.png')