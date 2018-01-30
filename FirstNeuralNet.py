import numpy as np

def nonlin(x, deriv = False):
	if(deriv == True):
		return (x * (1 - x))
	return 1 / (1 + np.exp(-x))

X = np.array([ [0,0],
				[0,1],
				[1,1]])

Y = np.array([[0],[0],[1]])

np.random.seed(1)

syn0 = 2*np.random.random((2,10)) - 1
syn1 = 2*np.random.random((10,1)) - 1

#print(syn0)
#print(syn1)

for j in range(100):
	l0 = X
	l1 = nonlin(np.dot(l0, syn0))
	l2 = nonlin(np.dot(l1, syn1))

	l2_error = Y - l2

	#if(j%10000)==0 :
	#	print("Error:") 
	#	print (str(np.mean(np.abs(l2_error))))
		#print("l0: ")
		#print(l0)
		#print("l1: ")
		#print(l1)
		#print("l2: ")
		#print(l2)

	l2_delta = l2_error * nonlin(l2, deriv=True)

	l1_error = l2_delta.dot(syn1.T)

	l1_delta = l1_error * nonlin(l1, deriv=True)

	#update weights

	syn1 += l1.T.dot(l2_delta)
	syn0 += l0.T.dot(l1_delta)

def neuralNet(x):
	l0 = X
	l1 = nonlin(np.dot(l0, syn0))
	l2 = nonlin(np.dot(l1, syn1))
	return l2

	
print ("Output after training: ")
np.round(l2,decimals=2)
print (l2)

while True:
	x = input("Enter 1 num: ")
	x = float(x)
	y = input("Enter 2 num: ")
	y = float(y)
	if y == -1:
		break
	l2 = neuralNet(np.array([x,y]))
	print(l2)
