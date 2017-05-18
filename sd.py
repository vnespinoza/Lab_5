import os, tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'#hides hardware usage warnings from tensorflow

''' create vector representaiton of data on the fly
each email represented as 0s and 1s determined by whats in vocab'''
def vocab():
	f =open('vocab.txt')
	out = []
	for line in f: out.append(line.strip())
	f.close()
	return out

def vec(lines, v): #create table from data in v
	xs = []; ys = []
	for line in lines:
		x = [0]*len(v)
		label, title = line.strip().split('\t')
		for word in title.split():
			if word in v: #make sure word in vocab
				word _index = v.index(word)
				x[word_index] = 1
		if label == '0' : y = [1,0] #ham not spam
		elif label == '1' : y = [0,1] #not ham, spam
		xs.append(x)
		ys.append(y)
	return xs, ys

##create neural network
if __name__ == '__main__':
	v = vocab()
##one neuron for each word in vocab file
'''create placeholder obj (use 32bit float to specify input)(matrix using unspec num of rows and 2 columns to specify input) in this case plugging in many data points (None is underspecified)'''
	input = tensorflow.placeholder(tensorflow.float32, [None,len(v)])
#input to hidden
#Wh = weights of connections(varibale values need to be retained)
	Wh = tensorflow.Variable(tensorflow.random_normal([len(v), 1000]))
#bh = bias values
	bh = tensorflow.Variable (tensorflow.random_normal([1000]))

##1,000 neurons activated by ReLU (not sigmoid)
	hidden = tensorflow.nn.relu(tensorflow.matmul(input,Wh)+bh) #apply ReLU function of matrix multiplication value(weighted sum)

#hidden to output 
#wy= weight matrix, by = bias matrix
	Wy = tensorflow.Variable(tensorflow.random_normal([1000,2]))
	by = tensorflow.Variable(tensorflow.random_normal([2]))

##logits (no activation function applied) with 2 neurons (spam vs. not-spam) softmax cross extropy applies activation function later
	output = tensorflow.matmul(hidden,Wy)+by

#Loss functions and learning alg
	target = tensorflow.placeholder(tensorflow.float32,[None,2]) #placeholder obj to store labels
	loss = tensorflow.losses.softmax_cross_entropy(target,output) #softmax cross entropy (comparing target values and network's predictions)
	learning_rate = 0.5 #amount by which neg gradient multiplied to adjust rate
	train = tensorflow.train.GradientDescentOptimizer(learning_rate).minimize(loss) #applies gradient descent learning algorithm to tweak values in  Variable objs 

#open tensorflow session to train
	sess = tensorflow.Session()
	init = tensorflow.global_variables_initializer()#initialize variables in network
	sess.run(init)#run network
	f = open('spam_assassin.train')
	lines = f.readlines()
	f.close()
	batch_size = 100
	for epoch in range(2):
		for j in range(len(lines)/batch_size):
			batch = lines[j*batch_size:(j+1)*batch_size]
			xs, ys = vec(batch, v)
			sess.run(train, feed_dict={input:xs, target:ys}) #only input and target need to feed
	# Test data
	#get prediction
	f = open ('spam_assasin.test')
	lines = f.readlines()
	f.close()
	xs, ys = vec(lines, v)
	final_prediction = sess.run(output, feed_dict={input:xs}) #output vectors, one for each data point in network
	sess.close()
	#print final_prediction 
	#Compare predictions with correct label and score 
	tp = 0.0; tn 0.0; fp = 0.0; fn = 0.0 
	for i in range(len(final_prediction)):
		if final_prediction[i][0] >= [i][1]: prediction = [1,0]
		else: prediction = [0,1]
		#predict vs. ys[i]
		if prediction == [1,0] and ys[i] == [1,0] #true neg
			tn += 1
		elif prediction == [1,0] and ys[i] == [0,1] #false neg
			fn =+ 1
		elif prediction == [0,1] and ys[i] == [1,0] #false pos
			fp += 1
		elif prediction == [0,1] and ys[i] == [0,1] #true pos
			tp += 1
		prediction = tp / (tp+fp)
		recall = tp / (tp+fn) #how many were called that should have been spam
		accuracy = (tp+tn) / (tn+fn+fp+tp) #classification accuracy 
		print prediction
		print recall
		print accuracy
