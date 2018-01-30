from numpy import exp, dot, array, random

class NeuralNetwork():
	def __init__(self, num_of_input_neurons, num_of_hidden_neurons = 8):
		#seed the random nunmber generator with 1 to have the same results
		random.seed(1)

		#We model a single neuron with n inputs, hidden layer and 1 output 
		#we asign random weights to a n*m and m*1 matrix with values from -1 to 1
		#and mean 0?
		self.synaptic_weights1 = 2 * random.random((num_of_input_neurons, num_of_hidden_neurons)) - 1
		self.synaptic_weights2 = 2 * random.random((num_of_hidden_neurons, 1)) - 1

	#sigmod function
	def __sigmoid(self, x):
		return(1 / (1 + exp(-x)))

	def __sigmoid_derivative(self, x):
		return x * (1 - x)

	def train(self, training_set_inputs, training_set_outputs, number_of_itterations):
		for itteration in range(number_of_itterations):
			#pass the training set through our net
			output = self.predict(training_set_inputs)

			#calculate an error and delta
			l1_error = training_set_outputs - output
			l1_delta = l1_error * self.__sigmoid_derivative(output) 
			l0_error = l1_delta.dot(self.synaptic_weights2.T)
			l0_delta = l0_error * self.__sigmoid_derivative(self.l1)

			#calculate the adjustment
			adjustment2 = self.l1.T.dot(l1_delta)
			adjustment1 = training_set_inputs.T.dot(l0_delta)
			#adjust the weight
			self.synaptic_weights1 += adjustment1
			self.synaptic_weights2 += adjustment2

	def predict(self, inputs):
		self.l1 = self.__sigmoid(dot(inputs, self.synaptic_weights1))
		return (self.__sigmoid(dot(self.l1, self.synaptic_weights2)))

	def think(self, inputs):
		#pass inputs through our network
		return self.predict(inputs)  


if __name__ == '__main__':

	#initialize a single neuron neural network
	neural_network = NeuralNetwork(6)

	print ('Random starting synaptic weights: ')
	print (neural_network.synaptic_weights1)
	print (neural_network.synaptic_weights2)

	#the training set. it has 4 examples, each containing  of 3 input values 
	# and 1 output layer
	training_set_inputs = array([[0,0,0,0,0,0], [0,1,0,0,0,0], [1,0,0,0,1,0], [1,1,1,0,1,1], [1,0,1,1,0,1], [1,1,1,1,0,1], [0,1,0,1,0,1]
		, [1,1,1,1,1,0], [0,0,0,0,1,1], [1,1,0,0,1,0]]) 
	training_set_outputs = array([[0, 0.1, 0.2, 0.5, 0.4, 0.5, 0.3,0.5,0.2,0.3]]).T


	#train the neural net using data
	#do it 10000 times, making small adjustments all the time
	neural_network.train(training_set_inputs, training_set_outputs, 100000)

	print('New synaptic weights after training: ')
	print (neural_network.synaptic_weights1)
	print (neural_network.synaptic_weights2)

	print (training_set_inputs)
	print (neural_network.think(training_set_inputs))
	#testing the network
	while True:
		a = [int(x) for x in input('Enter numbers: ').split()]
		try:
			print(neural_network.think(array(a)))
		except Exception as e:
			raise e
		
		
		if  input('Enter q to exit') == "q":
			break
