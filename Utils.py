import pandas as pd
import json

def get_data_from_csv(csv_file, sep):
	#reading dataset from csv
	data = pd.read_csv(csv_file, sep)

	#reading dataset descritpion from json file.
	with open(csv_file[:-4] + "-description.json") as desc_file:
		data_desc = json.loads(desc_file.read())

	return data, data_desc