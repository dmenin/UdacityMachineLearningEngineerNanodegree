# encoding: UTF-8
# #DO NOT REMOVE LINE ABOVE

import mnist_data
import tensorflow as tf
import math
import operator
import numpy as np
import cv2
from scipy import ndimage

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

    def ResetModelVariables(self):
        tf.reset_default_graph()

    def DisplayModelVariables(self):
        return [v.op.name for v in tf.all_variables()]


    def predict(self,i):
        saver = tf.train.Saver()
        with tf.Session() as sess:
            saver.restore(sess, 'checkpoints/myModel')

            #img = self.mnist.test.images[0]
            prediction = tf.argmax(self.Ylogits, 1)
            best = sess.run([prediction], feed_dict={self.X: [self.mnist.test.images[i]], self.pkeep: 1.0})
            return best


    def predict2(self,image):
        saver = tf.train.Saver()
        with tf.Session() as sess:
            saver.restore(sess, 'checkpoints/myModel')

            #(28,28) to (28,28,1)
            image = np.expand_dims(image, axis=2)

            best = sess.run(self.Ylogits, feed_dict={self.X: [image], self.pkeep: 1.0})
            # returns probability and value
            index, value = max(enumerate(best[0]), key=operator.itemgetter(1))
            return [value,index]





    def create_nn_model(self):
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
        test_data_update_freq=50
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
                        if a > maxAcc:
                            print 'Best Accuracy - Saving Checkpoint'
                            maxAcc = a
                            saver.save(sess, 'checkpoints/myModel')
                        else:
                            print 'No Accuracy Improvement'


                    # the backpropagation training step
                    sess.run(optimizer, {self.X: batch_X, self.Y_: batch_Y, self.lr: learning_rate, self.pkeep: 0.75})

    def getBestShift(self,img):
        cy, cx = ndimage.measurements.center_of_mass(img)
        print cy, cx

        rows, cols = img.shape
        shiftx = np.round(cols / 2.0 - cx).astype(int)
        shifty = np.round(rows / 2.0 - cy).astype(int)

        return shiftx, shifty

    def shift(self, img, sx, sy):
        rows, cols = img.shape
        M = np.float32([[1, 0, sx], [0, 1, sy]])
        shifted = cv2.warpAffine(img, M, (cols, rows))
        return shifted




    def testNewImage(self,image):
        # Read image
        originalImage = cv2.imread("test_images/" + image + ".png")

        # Read Image Black and White
        originalImageBW = cv2.imread("test_images/" + image + ".png", 0)


        _, originalImageBW = cv2.threshold(255 - originalImageBW, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        digit_image = -np.ones(originalImageBW.shape)
        height, width = originalImageBW.shape


        for cropped_width in range(100, 300, 20):
            for cropped_height in range(100, 300, 20):
                for shift_x in range(0, width - cropped_width, cropped_width / 4):
                    for shift_y in range(0, height - cropped_height, cropped_height / 4):
                        gray = originalImageBW[shift_y:shift_y + cropped_height, shift_x:shift_x + cropped_width]
                        if np.count_nonzero(gray) <= 20:
                            continue

                        if (np.sum(gray[0]) != 0) or (np.sum(gray[:, 0]) != 0) or (np.sum(gray[-1]) != 0) or (np.sum(gray[:, -1]) != 0):
                            continue

                        top_left = np.array([shift_y, shift_x])
                        bottom_right = np.array([shift_y + cropped_height, shift_x + cropped_width])

                        while np.sum(gray[0]) == 0:
                            top_left[0] += 1
                            gray = gray[1:]

                        while np.sum(gray[:, 0]) == 0:
                            top_left[1] += 1
                            gray = np.delete(gray, 0, 1)

                        while np.sum(gray[-1]) == 0:
                            bottom_right[0] -= 1
                            gray = gray[:-1]

                        while np.sum(gray[:, -1]) == 0:
                            bottom_right[1] -= 1
                            gray = np.delete(gray, -1, 1)

                        actual_w_h = bottom_right - top_left
                        if (np.count_nonzero(digit_image[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]] + 1) >
                                        0.2 * actual_w_h[0] * actual_w_h[1]):
                            continue

                        print "------------------"

                        rows, cols = gray.shape
                        compl_dif = abs(rows - cols)
                        half_Sm = compl_dif / 2
                        half_Big = half_Sm if half_Sm * 2 == compl_dif else half_Sm + 1
                        if rows > cols:
                            gray = np.lib.pad(gray, ((0, 0), (half_Sm, half_Big)), 'constant')
                        else:
                            gray = np.lib.pad(gray, ((half_Sm, half_Big), (0, 0)), 'constant')

                        gray = cv2.resize(gray, (20, 20))
                        gray = np.lib.pad(gray, ((4, 4), (4, 4)), 'constant')

                        shiftx, shifty = self.getBestShift(gray)

                        shifted = self.shift(gray, shiftx, shifty)
                        gray = shifted #/ 255.0

                        cv2.imwrite("test_images/" + image + "_" + str(shift_x) + "_" + str(shift_y) + ".png", shifted)

                        print "Prediction for ", (shift_x, shift_y, cropped_width)
                        # print "Pos"
                        # print top_left
                        # print bottom_right
                        # print actual_w_h
                        # print " "
                        # print flatten

                        pred = dTest.predict2(gray)
                        print pred

                        digit_image[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]] = pred[1]

                        cv2.rectangle(originalImage, tuple(top_left[::-1]), tuple(bottom_right[::-1]), color=(0, 255, 0),thickness=1)
                        #cv2.circle (originalImage, center = tuple(top_left[::-1]), radius=100, color=(0, 255, 0),thickness=1 )

                        font = cv2.FONT_HERSHEY_SIMPLEX
                        # value
                        cv2.putText(originalImage, str(pred[1]), (top_left[1], bottom_right[0] + 40), font, fontScale=1.4,
                                    color=(0, 255, 0), thickness=1)
                        # percentage
                        #cv2.putText(originalImage,format(pred[0]*100,".1f")+"%",(top_left[1]+30,bottom_right[0]+50), font,fontScale=0.4,color=(0,255,0),thickness=1)

        cv2.imwrite("test_images/" + image + "_result.png", originalImage)



# d = DigitRecognition()
# d.LoadDataSet()
# d.create_nn_model()
# d.StartTraining(200)


#
dTest = DigitRecognition()
dTest.LoadDataSet()
dTest.create_nn_model()
dTest.testNewImage("five")
