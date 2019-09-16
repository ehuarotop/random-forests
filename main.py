import Utils
from decisionTree import decisionTree as dt

def main():
	data = Utils.get_data_from_csv("dadosBenchmark_validacaoAlgoritmoAD.csv")
	print(data)
	det = dt(data)
	det.getInformationGain('Joga', 'Tempo')
	det.getInformationGain('Joga', 'Temperatura')
	det.getInformationGain('Joga', 'Umidade')
	det.getInformationGain('Joga', 'Ventoso')

if __name__ == "__main__":
	main()