# encoding: UTF-8
# #DO NOT REMOVE LINE ABOVE

import mnist_data
import tensorflow as tf
import math

tf.set_random_seed(0)
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


class DigitRecognition:

    def __init__(self):
        print 'Class Initialized'



    def LoadDataSet(self):
        self.mnist  = mnist_data.read_data_sets("mydata")
        self.foo = "aaaaa"

    def ResetModelVariables(self):
        tf.reset_default_graph()

    def DisplayModelVariables(self):
        return [v.op.name for v in tf.all_variables()]


    def getSaver(self):
        saver = tf.train.Saver()
        with tf.Session() as sess:
            saver.restore(sess, 'checkpoints/myModel')
            #print sess.run(self.W1)
            sess.run({self.X: self.mnist.test.images, self.pkeep: 1.0})
        return saver


    def create_nn_model(self):
        self.foo2 = "aaaaa"
        # input X: 28x28 grayscale images, the first dimension (None) will index the images in the mini-batch
        self.X = tf.placeholder(tf.float32, [None, 28, 28, 1])
        # correct answers will go here
        self.Y_ = tf.placeholder(tf.float32, [None, 10])
        # variable learning rate
        self.lr = tf.placeholder(tf.float32)
        # Probability of keeping a node during dropout = 1.0 at test time (no dropout) and 0.75 at training time
        self.pkeep = tf.placeholder(tf.float32)


        # three convolutional layers with their channel counts, and a
        # fully connected layer (tha last layer has 10 softmax neurons)
        K = 6  # first convolutional layer output depth
        L = 12  # second convolutional layer output depth
        M = 24  # third convolutional layer
        N = 200  # fully connected layerVisualize

        self.W1 = tf.Variable(tf.truncated_normal([6, 6, 1, K], stddev=0.1), name="W1")  # 6x6 patch, 1 input channel, K output channels
        self.B1 = tf.Variable(tf.constant(0.1, tf.float32, [K]), name="B1")
        self.W2 = tf.Variable(tf.truncated_normal([5, 5, K, L], stddev=0.1), name="W2")
        self.B2 = tf.Variable(tf.constant(0.1, tf.float32, [L]), name="B2")
        self.W3 = tf.Variable(tf.truncated_normal([4, 4, L, M], stddev=0.1), name="W3")
        self.B3 = tf.Variable(tf.constant(0.1, tf.float32, [M]), name="B3")

        self.W4 = tf.Variable(tf.truncated_normal([7 * 7 * M, N], stddev=0.1), name="W4")
        self.B4 = tf.Variable(tf.constant(0.1, tf.float32, [N]), name="B4")
        self.W5 = tf.Variable(tf.truncated_normal([N, 10], stddev=0.1), name="W5")
        self.B5 = tf.Variable(tf.constant(0.1, tf.float32, [10]), name="B5")

        # The model
        stride = 1  # output is 28x28
        Y1 = tf.nn.relu(tf.nn.conv2d(self.X, self.W1, strides=[1, stride, stride, 1], padding='SAME') + self.B1)
        stride = 2  # output is 14x14
        Y2 = tf.nn.relu(tf.nn.conv2d(Y1, self.W2, strides=[1, stride, stride, 1], padding='SAME') + self.B2)
        stride = 2  # output is 7x7
        Y3 = tf.nn.relu(tf.nn.conv2d(Y2, self.W3, strides=[1, stride, stride, 1], padding='SAME') + self.B3)

        # reshape the output from the third convolution for the fully connected layer
        YY = tf.reshape(Y3, shape=[-1, 7 * 7 * M])

        Y4 = tf.nn.relu(tf.matmul(YY, self.W4) + self.B4)
        YY4 = tf.nn.dropout(Y4, self.pkeep)
        self.Ylogits = tf.matmul(YY4, self.W5) + self.B5

        #dont need to
        #return Ylogits



    def StartTraining(self, iterations = 100):
        #weights and biases to be removed

        predictions  =  self.Ylogits# create_nn_model()



        #calculates the difference between the predicion we got to the known labels we have
        #VAR renamed cross_entropy -> loss (could be  cost)
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(predictions, self.Y_))*100

        #minimize the loss (the difference between prediciton and Y_)
        #piece of the computation graph that computes the gradient, applies the gradient to the weights and biases
        #to obtain new weights and biases
        #VAR renamed train_step -> optimizer
        optimizer = tf.train.AdamOptimizer(self.lr).minimize(loss)


        Y = tf.nn.softmax(predictions)

        # accuracy of the trained model, between 0 (worst) and 1 (best)
        correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(self.Y_, 1))
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



                    batch_X, batch_Y = self.mnist.train.next_batch(100)
                    learning_rate = min_learning_rate + (max_learning_rate - min_learning_rate) * math.exp(-n / decay_speed)

                    if request_data_update:
                        a, c = sess.run([accuracy, loss], {self.X: batch_X, self.Y_: batch_Y, self.pkeep: 1.0})
                        print(str(n) + ": accuracy:" + str(a) + " loss: " + str(c) + " (lr:" + str(learning_rate) + ")")

                    # print test
                    if request_test_data_update:
                        a, c = sess.run([accuracy, loss], {self.X: self.mnist.test.images, self.Y_: self.mnist.test.labels, self.pkeep: 1.0})
                        print(str(n) + ": ********* epoch " + str(i * 100 // self.mnist.train.images.shape[0] + 1) + " ********* test accuracy:" + str(a) + " test loss: " + str(c))
                        #save checkpoint here if accuracy is up


                    # the backpropagation training step
                    sess.run(optimizer, {self.X: batch_X, self.Y_: batch_Y, self.lr: learning_rate, self.pkeep: 0.75})
                ##print 'Saving Checkpoint'
                saver.save(sess, 'checkpoints/myModel')


            #try to test one image?
            #a, c = sess.run([accuracy, loss], {X: mnist.test.images[0], Y_: mnist.test.labels[0], pkeep: 1.0})

    # LoadDataSet()
    # Start(20)





    # predictions = create_nn_model()
    #
    #
    # with tf.Session() as sess:
    #     sess.run(tf.initialize_all_variables())
    #     print 'Variables:', ([v.op.name for v in tf.all_variables()])
    #     saver = tf.train.Saver()
    # #     saver.restore(sess,'checkpoints/myModel')




    # print '-----------------------------------'
    # with tf.Graph().as_default() as g:
    #     with tf.Session() as sess:
    #         saver = tf.train.Saver()
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



# d = DigitRecognition()
# d.LoadDataSet()
# d.create_nn_model()
# d.StartTraining(400)



dTest = DigitRecognition()
dTest.LoadDataSet()
dTest.create_nn_model()
saver = dTest.getSaver()
