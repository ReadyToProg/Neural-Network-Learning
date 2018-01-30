from numpy import exp, dot, array, random

class NeuralNetwork():
	def __init__(self):
		#seed the random nunmber generator with 1 to have the same results
		random.seed(1)

		#We model a single neuron with 3 input and 1 output 
		#we asign random weights to a 3*1 matrix with values from -1 to 1
		#and mean 0?
		self.synaptic_weights = 2 * random.random((3,1)) - 1

	#sigmod function
	def __sigmoid(self, x):
		return(1 / (1 + exp(-x)))

	def __sigmoid_derivative(self, x):
		return x * (1 - x)

	def train(self, training_set_inputs, training_set_outputs, number_of_itterations):
		for itteration in range(number_of_itterations):
			#pass the training set through our net
			output = self.predict(training_set_inputs)

			#calculate an error
			error = training_set_outputs - output

			#calculate the adjustment
			adjustment = dot(training_set_inputs.T, error * self.__sigmoid_derivative(output))

			#adjust the weight
			self.synaptic_weights += adjustment

	def predict(self, inputs):
		return (self.__sigmoid(dot(inputs, self.synaptic_weights)))

	def think(self, inputs):
		#pass inputs through our network
		return self.predict(inputs)  


if __name__ == '__main__':

	#initialize a single neuron neural network
	neural_network = NeuralNetwork()

	print ('Random starting synaptic weights: ')
	print (neural_network.synaptic_weights)

	#the training set. it has 4 examples, each containing  of 3 input values 
	# and 1 output layer
	training_set_inputs = array([[0,0,1], [1,1,1], [1,0,1], [0,1,0]]) 
	training_set_outputs = array([[0,1,1,0]]).T


	#train the neural net using data
	#do it 10000 times, making small adjustments all the time
	neural_network.train(training_set_inputs, training_set_outputs, 10000)

	print('New synaptic weights after training: ')
	print (neural_network.synaptic_weights)

	#test the network
	print('Considering [1,0,0]')
	print(neural_network.think(array([[1,0,0]])))
