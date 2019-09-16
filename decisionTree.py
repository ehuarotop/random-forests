import Node
import math

class decisionTree:
	
	def __init__(self, data):
		self.data = data
		#self.root_node = rootNode

	############## Information Gain ##############
	def getInformationGain(self, classe, attribute=None):
		#target will be all of the values in classe 
		class_instances = self.data[classe]
		#Getting the number of instances
		total = class_instances.count()
		#Getting unique values in pandas series (column)
		unique_values = class_instances.unique()

		#information for all instances of class.
		entropy = 0.0

		for value in unique_values:
			count_instances = class_instances.where(class_instances == value).count()
			entropy += -(count_instances/total)*math.log(count_instances/total, 2)

		############### Getting information gain for the specific attribute ###############
		attr_instances = self.data[attribute]
		total = attr_instances.count()
		attr_unique_values = attr_instances.unique()

		information_gain = 0.0

		for attr_value in attr_unique_values:
			class_instances = self.data[self.data[attribute]==attr_value][classe]
			class_unique_values = class_instances.unique()
			class_attr_total = class_instances.count()

			attr_information_gain = 0.0
			
			#attr value information gain
			for class_value in class_unique_values:
				count_instances = class_instances.where(class_instances == class_value).count()
				attr_information_gain += -(count_instances/class_attr_total)*math.log(count_instances/class_attr_total,2)

			information_gain += (class_attr_total/total)*attr_information_gain

		#Getting information gain substracting from entropy initially calculated
		information_gain = entropy - information_gain
		
		return information_gain



