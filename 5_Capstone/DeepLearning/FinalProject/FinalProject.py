# encoding: UTF-8


import mnist_data
import tensorflow as tf
import math
tf.set_random_seed(0)

# Download images and labels

# neural network structure for this sample:
#
# · · · · · · · · · ·      (input data, 1-deep)                 X [batch, 28, 28, 1]
# @ @ @ @ @ @ @ @ @ @   -- conv. layer 6x6x1=>6 stride 1        W1 [5, 5, 1, 6]        B1 [6]
# ∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶                                           Y1 [batch, 28, 28, 6]
#   @ @ @ @ @ @ @ @     -- conv. layer 5x5x6=>12 stride 2       W2 [5, 5, 6, 12]        B2 [12]
#   ∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶                                             Y2 [batch, 14, 14, 12]
#     @ @ @ @ @ @       -- conv. layer 4x4x12=>24 stride 2      W3 [4, 4, 12, 24]       B3 [24]
#     ∶∶∶∶∶∶∶∶∶∶∶                                               Y3 [batch, 7, 7, 24] => reshaped to YY [batch, 7*7*24]
#      \x/x\x\x/ ✞      -- fully connected layer (relu+dropout) W4 [7*7*24, 200]       B4 [200]
#       · · · ·                                                 Y4 [batch, 200]
#       \x/x\x/         -- fully connected layer (softmax)      W5 [200, 10]           B5 [10]
#        · · ·                                                  Y [batch, 20]
mnist = 0

def LoadDataSet():
    global mnist
    mnist  = mnist_data.read_data_sets("mydata")


# input X: 28x28 grayscale images, the first dimension (None) will index the images in the mini-batch
X = tf.placeholder(tf.float32, [None, 28, 28, 1])
# correct answers will go here
Y_ = tf.placeholder(tf.float32, [None, 10])
# variable learning rate
lr = tf.placeholder(tf.float32)
# Probability of keeping a node during dropout = 1.0 at test time (no dropout) and 0.75 at training time
pkeep = tf.placeholder(tf.float32)


def create_nn_model():
    # three convolutional layers with their channel counts, and a
    # fully connected layer (tha last layer has 10 softmax neurons)
    K = 6  # first convolutional layer output depth
    L = 12  # second convolutional layer output depth
    M = 24  # third convolutional layer
    N = 200  # fully connected layerVisualize

    W1 = tf.Variable(tf.truncated_normal([6, 6, 1, K], stddev=0.1), name="W1")  # 6x6 patch, 1 input channel, K output channels
    B1 = tf.Variable(tf.constant(0.1, tf.float32, [K]), name="B1")
    W2 = tf.Variable(tf.truncated_normal([5, 5, K, L], stddev=0.1), name="W2")
    B2 = tf.Variable(tf.constant(0.1, tf.float32, [L]), name="B2")
    W3 = tf.Variable(tf.truncated_normal([4, 4, L, M], stddev=0.1), name="W3")
    B3 = tf.Variable(tf.constant(0.1, tf.float32, [M]), name="B3")

    W4 = tf.Variable(tf.truncated_normal([7 * 7 * M, N], stddev=0.1), name="W4")
    B4 = tf.Variable(tf.constant(0.1, tf.float32, [N]), name="B4")
    W5 = tf.Variable(tf.truncated_normal([N, 10], stddev=0.1), name="W5")
    B5 = tf.Variable(tf.constant(0.1, tf.float32, [10]), name="B5")

    # The model
    stride = 1  # output is 28x28
    Y1 = tf.nn.relu(tf.nn.conv2d(X, W1, strides=[1, stride, stride, 1], padding='SAME') + B1)
    stride = 2  # output is 14x14
    Y2 = tf.nn.relu(tf.nn.conv2d(Y1, W2, strides=[1, stride, stride, 1], padding='SAME') + B2)
    stride = 2  # output is 7x7
    Y3 = tf.nn.relu(tf.nn.conv2d(Y2, W3, strides=[1, stride, stride, 1], padding='SAME') + B3)

    # reshape the output from the third convolution for the fully connected layer
    YY = tf.reshape(Y3, shape=[-1, 7 * 7 * M])

    Y4 = tf.nn.relu(tf.matmul(YY, W4) + B4)
    YY4 = tf.nn.dropout(Y4, pkeep)
    Ylogits = tf.matmul(YY4, W5) + B5

    return Ylogits


def Start(iterations = 100):
    #weights and biases to be removed
    predictions  = create_nn_model()

    #calculates the difference between the predicion we got to the known labels we have
    #VAR renamed cross_entropy -> loss (could be  cost)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(predictions, Y_))*100

    #minimize the loss (the difference between prediciton and Y_)
    #piece of the computation graph that computes the gradient, applies the gradient to the weights and biases
    #to obtain new weights and biases
    #VAR renamed train_step -> optimizer
    optimizer = tf.train.AdamOptimizer(lr).minimize(loss)


    # initialize
    #sess = tf.Session()



    Y = tf.nn.softmax(predictions)

    # accuracy of the trained model, between 0 (worst) and 1 (best)
    correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))




    # cicles of feedforward + backprop
    num_epochs = 2
    maxAcc=0
    batch_size = 100


    #iterations = 100#10001#100
    train_data_update_freq = 20
    test_data_update_freq=100
    one_test_at_start=True
    more_tests_at_start=False

    # learning rate decay
    max_learning_rate = 0.003
    min_learning_rate = 0.0001
    decay_speed = 2000.

    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        print 'Variables:', ([v.op.name for v in tf.all_variables()])

        for i in range(int(iterations // train_data_update_freq + 1)): #500
            # if (i == iterations // train_data_update_freq): #last iteration
            #     training_step(iterations, True, True)
            # else:
            for k in range(train_data_update_freq):
                n = i * train_data_update_freq + k
                request_data_update = (n % train_data_update_freq == 0)
                request_test_data_update = (n % test_data_update_freq == 0) and (n > 0 or one_test_at_start)
                if more_tests_at_start and n < test_data_update_freq: request_test_data_update = request_data_update



                batch_X, batch_Y = mnist.train.next_batch(100)
                learning_rate = min_learning_rate + (max_learning_rate - min_learning_rate) * math.exp(-n / decay_speed)

                if request_data_update:
                    a, c = sess.run([accuracy, loss], {X: batch_X, Y_: batch_Y, pkeep: 1.0})
                    print(str(n) + ": accuracy:" + str(a) + " loss: " + str(c) + " (lr:" + str(learning_rate) + ")")

                # print test
                if request_test_data_update:
                    a, c = sess.run([accuracy, loss], {X: mnist.test.images, Y_: mnist.test.labels, pkeep: 1.0})
                    print(str(n) + ": ********* epoch " + str(i * 100 // mnist.train.images.shape[0] + 1) + " ********* test accuracy:" + str(a) + " test loss: " + str(c))
                    # datavis.append_test_curves_data(i, a, c)
                    # if (a > maxAcc):
                    #    maxAcc = a

                # the backpropagation training step
                sess.run(optimizer, {X: batch_X, Y_: batch_Y, lr: learning_rate, pkeep: 0.75})
        saver.save(sess, 'myModel')


        #try to test one image?
        #a, c = sess.run([accuracy, loss], {X: mnist.test.images[0], Y_: mnist.test.labels[0], pkeep: 1.0})

LoadDataSet()
Start(20)



print '-----------------------------------'
with tf.Graph().as_default() as g:
    with tf.Session() as sess:
        saver = tf.train.Saver()
        # saver.restore(sess, 'myModel')
        # # Initializing the variables
        # print([v.op.name for v in tf.all_variables()])
        # #print(sess.run(model.b)) #100

#print("max test accuracy: " + str(datavis.get_max_test_accuracy()))


# from datetime import datetime
# print datetime.now().strftime("%Y-%m-%d %H:%M:%S")

#for i in range(10001):
#    training_step(i,20,100)

# to save the animation as a movie, add save_movie=True as an argument to datavis.animate
# to disable the visualisation use the following line instead of the datavis.animate line
# for i in range(10000+1): training_step(i, i % 100 == 0, i % 20 == 0)

