import os, tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'#hides hardware usage warnings from tensorflow


##one neuron for each word in vocab file
input = tensorflow.placeholder(tensorflow.float32, [None,2])#create placeholder obj (use 32bit float to specify input)(matrix using unspec num of rows and 2 columns to specify input)

Wh = tensorflow.Variable(tensorflow.random_normal([2,2]))#create tensor (2x2 matrix (2D array))
bh = tensorflow.Variable (tensorflow.random_normal([2]))

##1,000 neurons activated by ReLU (not sigmoid(
hidden = tensorflow.nn.sigmoid(tensorflow.matmul(input,Wh)+bh) #apply sigmoid function of matrix multiplication value
Wy = tensorflow.Variable(tensorflow.random_normal([2,1]))
by = tensorflow.Variable(tensorflow.random_normal([1]))
output = tensorflow.nn.sigmoid(tensorflow.matmul(hidden,Wy)+by)

#learning
target = tensorflow.placeholder(tensorflow.float32,[None,1]) #placeholder obj to store labels
loss = tensorflow.losses.mean_squared_error(target,output) #calculates mean squared error by comparing target values and network's predictions
learning_rate = 0.5
train = tensorflow.train.GradientDescentOptimizer(learning_rate).minimize(loss) #applies gradient descent learning algorithm to tweak values in  Variable objs 

#open tensorflow session to train
sess = tensorflow.Session()
init = tensorflow.global_variables_initializer()
sess.run(init)
x = [ [0,0], [0,1], [1,0], [1,1] ]
y = [ [0], [1], [1], [0] ]
for i in range(10000):
	sess.run(train, feed_dict={input:x, target:y})
final_prediction = sess.run(output, feed_dict={input:x})
sess.close()
print final_prediction 
