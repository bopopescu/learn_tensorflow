import numpy as np

def read_data_from_file(file_name):
	text = open(file_name, 'r').readlines()[1:]

	data = [line[:-1].split('\t') for line in text]
	births = [float(line[1]) for line in data]
	lifes = [float(line[2]) for line in data]
	data = list(zip(births, lifes))
	n_samples = len(data)
	data = np.asarray(data, dtype=np.float32)
	return data, n_samples
	print(data)


