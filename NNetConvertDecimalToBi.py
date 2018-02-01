from numpy import dot, array, exp, random, eye, zeros

class DecimalToBinaryNNet():
	def __init__(self):
		random.seed(1)
		self.synaptic_weight1 = 2 * random.random((15, 15)) - 1
		self.synaptic_weight2 = 2 * random.random((15, 4)) - 1
	def __sigmoid(self, x):
		return(1 / (1 + exp(-x)))

	def __sigmoid_derivative(self, x):
		return(x * (1 - x))

	def train(self, input_training_data, output_training_data, iterration_number = 10000):
		for iteration in range(iterration_number):
			l1 = input_training_data
			output = self.predict(l1)

			#error calculating
			errorl2 = output_training_data - output
			deltal2 = errorl2 * self.__sigmoid_derivative(output)
			errorl1 = deltal2.dot(self.synaptic_weight2.T)
			deltal1 = errorl1 * self.__sigmoid_derivative(self.l2)
			#calculating adjustments
			adjustment2 = self.l2.T.dot(deltal2)
			adjustment1 = l1.T.dot(deltal1)

			self.synaptic_weight2 += adjustment2
			self.synaptic_weight1 += adjustment1


	def predict(self, inputs):
		self.l2 = self.__sigmoid(dot(inputs, self.synaptic_weight1))
		return(self.__sigmoid(dot(self.l2, self.synaptic_weight2)))

	def think(self, inputs):
		return(self.predict(inputs))


if __name__ == '__main__':
	neural_network = DecimalToBinaryNNet()

	print ('Random starting synaptic weights: ')
	print (neural_network.synaptic_weight1)
	print (neural_network.synaptic_weight2)

	training_set_inputs = eye(15)
	training_set_inputs[14] = 0
	#training_set_inputs[10] = 0
	#training_set_inputs[8] = 0
	#training_set_inputs[4] = 0
	training_set_inputs[2] = 0
	print(training_set_inputs)
	
	training_set_outputs = array([[0,0,0,0],[1,0,0,0], [0,0,0,0], [1,1,0,0], [0,0,1,0],
									[1,0,1,0],[0,1,1,0], [1,1,1,0], [0,0,0,1], [1,0,0,1],
									[0,1,0,0], [1,1,0,1], [0,0,1,1], [1,0,1,1], [0,0,0,0]])
	#print(training_set_outputs)

	neural_network.train(training_set_inputs, training_set_outputs)
	print(neural_network.synaptic_weight2)
	print(neural_network.think(array([1,1,0,0,0,0,0,0,0,0,0,0,1	,0,0])))
