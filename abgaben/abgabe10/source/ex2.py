import random
import numpy as np
import tensorflow as tf

##### Ex10.2 a ######
#read data
with open('data.txt', 'r') as myfile:
    data=myfile.read().replace('\n', '').replace(" ", "")

##### Ex10.2 b ######
#extract random substring of length chunk_size
def extract( chunk_size ):
	start = random.randrange(0, len(data)-chunk_size-1, 1)
	chunk = data[start: start+k]
	return chunk

##### Ex10.2 c ######
#create one hot vectors
def one_hot( str ):
	a = np.array([0,0,0,0])
	for x in range(0, len(str)):
		if str[x] == 'a':
			newrow = [1,0,0,0]	
		if str[x] == 'c':
			newrow = [0,1,0,0]
		if str[x] == 'g':
			newrow = [0,0,1,0]
		if str[x] == 't':
			newrow = [0,0,0,1]
		a = np.vstack([a,newrow])
	return a;

##### Ex10.2 d ######
learning_rate = 0.0001
training_epochs = 50000

#cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
#optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

##### Ex10.2 e ######
k=5
chunk = extract(k)
#print(chunk)
vec = one_hot(chunk)
#print(vec)

