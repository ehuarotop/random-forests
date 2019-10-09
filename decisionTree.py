import math
import time
from anytree import AnyNode, RenderTree
import random
from anytree.exporter import DotExporter

class decisionTree:
	
	def __init__(self, data, data_desc, attributes_to_consider=None,verbose=None):
		self.data = data
		self.data_desc = data_desc
		self.root_node = None
		self.count_nodes = 0
		self.verbose = verbose
		self.attributes_to_consider = attributes_to_consider

	############## Information Gain ##############
	def getInformationGain(self, classe, attribute=None,type_attr=None,new_data=None):
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
		#Getting all the examples for the specific attribute
		attr_instances = data[attribute]
		#Getting the total number of examples of the specific attribute.
		total = attr_instances.count()

		#Initializing general information gain
		information_gain = 0.0

		#Getting unique instances based on the type of attribute
		if type_attr=="nominal":
			attr_unique_values = attr_instances.unique()

			#Getting information gain for each value of the specific attribute
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
		elif type_attr=="numeric":
			#Getting mean of the example values of this attribute.
			mean_attr_instances = attr_instances.mean()

			#Splitting examples values in "less than" and "greater than" the mean
			less_than_values = data[data[attribute] < mean_attr_instances]
			greater_than_values = data[data[attribute] >= mean_attr_instances]

			####### Getting information gain value manually for each new partition #######

			################### Getting information gain for "less than" partition
			class_instances_less = less_than_values[classe]
			class_unique_values_less = class_instances_less.unique()
			class_attr_total_less = class_instances_less.count()

			attr_information_gain_less = 0.0

			#attr value information gain
			for class_value_less in class_unique_values_less:
				count_instances_less = class_instances_less.where(class_instances_less == class_value_less).count()
				attr_information_gain_less += -(count_instances_less/class_attr_total_less)*math.log(count_instances_less/class_attr_total_less,2)

			#Updating information gain for the "less than" partition
			information_gain += (class_attr_total_less/total)*attr_information_gain_less

			################### Getting information gain for "greater than" partition
			class_instances_greater = greater_than_values[classe]
			class_unique_values_greater = class_instances_greater.unique()
			class_attr_total_greater = class_instances_greater.count()

			attr_information_gain_greater = 0.0

			#attr value information gain
			for class_value_greater in class_unique_values_greater:
				count_instances_greater = class_instances_greater.where(class_instances_greater == class_value_greater).count()
				attr_information_gain_greater += -(count_instances_greater/class_attr_total_greater)*math.log(count_instances_greater/class_attr_total_greater,2)

			#Updating information gain for the "less than" partition
			information_gain += (class_attr_total_greater/total)*attr_information_gain_greater


		#Getting information gain substracting from entropy initially calculated
		information_gain = entropy - information_gain
		
		return information_gain

	def getRandomAttributes(self, data, n_attr=None):
		'''Getting "n_attr" attributes randomly from data ----> "Amostragem de atributos"'''
		data_attributes = list(data.columns.values)[:-1]

		#Considering only the square root of the number of attributes.
		if n_attr is None:
			n_attr = round(math.sqrt(len(data_attributes)))

		random_attributes = []

		for i in range(n_attr):
			random_attribute = random.choice(data_attributes)
			if random_attribute not in random_attributes:
				random_attributes.append(random_attribute)
			else:
				while random_attribute in random_attributes:
					random_attribute = random.choice(data_attributes)

				random_attributes.append(random_attribute)

		return random_attributes

	def getAttributeWithMaxInfoGain(self, list_attr, new_data=None):
		#root node case
		if new_data is None:
			data = self.data
		else:
			data = new_data
		
		#Getting attributes (assuming it has class predictions in the last element)
		#print('Using only: ', list_attr)
		#attributes = list(data.columns.values)[:-1]
		classe = list(data.columns.values)[-1]

		info_gain = {}

		for attribute in list_attr:
			#Getting information gain for the current attribute indicating his attribute type desc.
			attr_gain = self.getInformationGain(classe, attribute, self.data_desc[attribute],data)
			info_gain[attribute] = attr_gain

		sorted_info_gain = dict(sorted(info_gain.items(), key=lambda kv: kv[1], reverse=True))

		if self.verbose is not None:
			print('Information Gains: ', sorted_info_gain)

		return list(sorted_info_gain.keys())[0], sorted_info_gain[list(sorted_info_gain.keys())[0]]

	def generateDecisionTree(self, parent_node=None, new_data=None, branch=None, mean_attr=None):
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
			self.count_nodes += 1
			new_node = AnyNode(name=str(self.count_nodes), label='vazio',parent=parent_node, branch=branch)
		if len(data[classe].unique())==1:
			#Partition found is 'Pure', only one value
			#Case: the bootstrap generated have elements only from one class
			#only one node will be generated for this decisionTree
			if self.root_node is None:
				#print('Nodo folha pure')
				self.count_nodes += 1
				self.root_node = AnyNode(name=str(self.count_nodes), label=data[classe].unique()[0])
				new_node = self.root_node
			else:
				#normal leaf node.
				self.count_nodes += 1
				new_node = AnyNode(name=str(self.count_nodes), label=data[classe].unique()[0],parent=parent_node, branch=branch)
		else:
			#Getting random attributes
			attr_list = self.getRandomAttributes(data, self.attributes_to_consider)

			#Getting attribute that maximizes information gain
			attr_max_gain, max_gain = self.getAttributeWithMaxInfoGain(attr_list, data)

			if self.verbose is not None:
				print('Attribute with max information gain: ' + attr_max_gain + '. Information gain: ' + str(max_gain))

			#if not have information gain
			if max_gain == 0.0:
				if self.root_node is None:
					majority_class = data[classe].value_counts().idxmax()
					self.count_nodes += 1
					self.root_node = AnyNode(name=str(self.count_nodes), label=majority_class, information_gain=max_gain)
					new_node = self.root_node
				else:
					#normal leaf node.
					self.count_nodes += 1
					new_node = AnyNode(name=str(self.count_nodes), label=parent_node.majority_class,parent=parent_node, branch=branch, information_gain=max_gain )

				#If not have information gain, return root_node.
				return self.root_node

			#getting majority class for the attribute with max info gain
			majority_class = data[classe].value_counts().idxmax()

			if self.data_desc[attr_max_gain] == "nominal":
				#adding node to decision tree
				if parent_node == None:
					self.count_nodes += 1
					self.root_node = AnyNode(name=str(self.count_nodes), label=attr_max_gain, majority_class=majority_class, information_gain=max_gain)
					new_parent_node = self.root_node
				else:
					self.count_nodes += 1
					new_parent_node = AnyNode(name=str(self.count_nodes), label=attr_max_gain,parent=parent_node,branch=branch, majority_class=majority_class, information_gain=max_gain)

				#Getting the unique values of this attribute
				unique_attr_values = data[attr_max_gain].unique()

				#Iterating over the branches
				for attr_value in unique_attr_values:
					if self.verbose is not None:
						print('Generating decision tree from ' + str(attr_max_gain) + '-' + str(attr_value))
					new_data = data[data[attr_max_gain]==attr_value]
					if self.verbose is not None:
						print(new_data)
						print('\n')
					self.generateDecisionTree(new_parent_node, new_data,attr_value)
			
			elif self.data_desc[attr_max_gain] == "numeric":
				#Getting mean of the example values of attr_max_gain.
				mean_attr = data[attr_max_gain].mean()

				#adding node to decision tree
				if parent_node == None:
					self.count_nodes += 1
					self.root_node = AnyNode(name=str(self.count_nodes), label=attr_max_gain, mean_attr=mean_attr, majority_class=majority_class, information_gain=max_gain)
					new_parent_node = self.root_node
				else:
					self.count_nodes += 1
					new_parent_node = AnyNode(name=str(self.count_nodes), label=attr_max_gain,parent=parent_node, branch=branch, mean_attr=mean_attr, majority_class=majority_class, information_gain=max_gain)

				#Splitting examples values in "less than" and "greater than" the mean
				less_than_values = data[data[attr_max_gain] < mean_attr]
				greater_than_values = data[data[attr_max_gain] >= mean_attr]

				#Generate decision tree for less than values data
				self.generateDecisionTree(new_parent_node, less_than_values,"left-branch",mean_attr)
				if self.verbose is not None:
					print('Generating decision tree from left branch (less values) - ' + str(mean_attr))
					print(less_than_values)

				#Generate decision tree for less than values data
				self.generateDecisionTree(new_parent_node, greater_than_values, "right-branch",mean_attr)
				if self.verbose is not None:
					print('Generating decision tree from right branch (greater values) - ' + str(mean_attr))
					print(greater_than_values)

			return self.root_node

	def renderDecisionTree(self,root_node):
		print(RenderTree(root_node))
		#for pre, fill, node in RenderTree(root_node):
			#print("%s%s" % (pre, node.name))

	def exportDecisionTreeToPNG(self, root_node, filename):

		def node_attr_func(node):
			#print('gaaaaa')
			try:
				node.information_gain
			except AttributeError as error:
				return 'label="%s"' % (node.label)
			else:
				return 'label="%s"' % (node.label + ' - ganho: ' + "{0:.3f}".format(node.information_gain))

		def edge_attr_func(node, child):
			return 'label="%s"' % (str(child.branch))

		def edge_type_func(node, child):
			return '--'

		DotExporter(root_node, graph="graph",
                       nodeattrfunc=node_attr_func,
                       edgeattrfunc=edge_attr_func,
                       edgetypefunc=edge_type_func).to_picture(filename)

	def predict_instance(self,instance, root_node=None):
		#Initializing prediction to None
		prediction = None

		#Verifying root_node (default self.root_node)
		if root_node is None:
			current_root_node = self.root_node
		else:
			current_root_node = root_node

		#Stop condition (leaf node)
		if current_root_node.is_leaf:
			prediction = current_root_node.label
		else:
			#Iterate recursively between children of the current root node
			if self.data_desc[current_root_node.label] == "nominal":
				#nominal case
				for children in current_root_node.children:
					if children.branch == instance[current_root_node.label]:
						current_root_node = children
						prediction = self.predict_instance(instance, current_root_node)
						return prediction
			elif self.data_desc[current_root_node.label] == "numeric":
				#Numeric case
				if instance[current_root_node.label] < current_root_node.mean_attr:
					for children in current_root_node.children:
						if children.branch == "left-branch":
							current_root_node = children
							prediction = self.predict_instance(instance,current_root_node)
							return prediction
				elif instance[current_root_node.label] >= current_root_node.mean_attr:
					for children in current_root_node.children:
						if children.branch == "right-branch":
							current_root_node = children
							prediction = self.predict_instance(instance,current_root_node)
							return prediction

			if prediction == None:
				#returning the majority class from the parent node (in case the decisionTree don't manage to get a prediction)
				return current_root_node.majority_class

		return prediction
