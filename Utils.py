import pandas as pd

def get_data_from_csv(csv_file):
	data = pd.read_csv(csv_file, sep=";")
	return data