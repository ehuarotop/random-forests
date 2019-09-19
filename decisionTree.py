#from Node import Node
import math
import time
from anytree import Node, RenderTree
#from anytree.exporter import DotExporter
#from graphviz import Source
#from graphviz import render

class decisionTree:
	
	def __init__(self, data):
		self.data = data
		self.root_node = None
		self.intern_nodes = []
		self.terminal_nodes = []

	'''def addNode(self,node):
		if self.root_node == None:
			self.root_node = node
		else:
			print('Adding intern or terminal node')'''

	############## Information Gain ##############
	def getInformationGain(self, classe, attribute=None, new_data=None):
		if new_data is None:
			data = self.data
		else:
			data = new_data

		#target will be all of the values in classe 
		class_instances = data[classe]
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
		attr_instances = data[attribute]
		total = attr_instances.count()
		attr_unique_values = attr_instances.unique()

		information_gain = 0.0

		for attr_value in attr_unique_values:
			class_instances = data[data[attribute]==attr_value][classe]
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

	def getAttributeWithMaxInfoGain(self, new_data=None):
		#root node case
		if new_data is None:
			data = self.data
		else:
			data = new_data
		
		#Getting attributes (assuming it has class predictions in the last element)
		attributes = list(data.columns.values)[:-1]
		classe = list(data.columns.values)[-1]

		info_gain = {}

		for attribute in attributes:
			attr_gain = self.getInformationGain(classe, attribute, data)
			info_gain[attribute] = attr_gain

		sorted_info_gain = dict(sorted(info_gain.items(), key=lambda kv: kv[1], reverse=True))

		return list(sorted_info_gain.keys())[0]

	def generateDecisionTree(self, parent_node=None, new_data=None):
		if new_data is None:
			data = self.data
		else:
			data = new_data

		#getting column containing the classes (assuming always last column)
		classe = list(data.columns.values)[-1]

		#Verifying stop conditions
		if data.empty:
			#Partition found is empty
			print('Nodo folha empty')
			new_node = Node('vazio',parent_node)
		if len(data[classe].unique())==1:
			#Partition found is 'Pure', only one value
			print('Nodo folha pure')
			new_node = Node(data[classe].unique(),parent_node)
		else:
			#Getting attribute that maximizes information gain
			attr_max_gain = self.getAttributeWithMaxInfoGain(data)

			#Initializing root_node
			#node = Node(attr_max_gain, data)

			#adding node to decision tree
			if parent_node == None:
				self.root_node = Node(attr_max_gain)
				new_parent_node = self.root_node
			else:
				new_parent_node = Node(attr_max_gain,parent_node)
				#self.addNode(node)

			#Getting the unique values of this attribute
			unique_attr_values = self.data[attr_max_gain].unique()

			#Iterating over the branches
			for attr_value in unique_attr_values:
				print('Generating decision tree from ' + attr_max_gain + '-' + attr_value)
				new_data = data[data[attr_max_gain]==attr_value]
				print(new_data)
				self.generateDecisionTree(new_parent_node, new_data)

			return self.root_node

	def renderDecisionTree(self,root_node):
		for pre, fill, node in RenderTree(root_node):
			print("%s%s" % (pre, node.name))

	#def exportDecisionTreeToPNG(self,root_node,path):
		#DotExporter(root_node).to_dotfile('udo.dot')
		#Source.from_file('udo.dot')
		#render('dot', 'png', 'udo.dot')
		#DotExporter(root_node).to_picture(path)
		#DotExporter(root_node,nodeattrfunc=lambda node: 'label="{}"'.format(node.name)).to_picture("graph.png")







