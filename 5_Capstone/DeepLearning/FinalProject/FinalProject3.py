import tensorflow as tf
import input_data
import cv2
import numpy as np
from scipy import ndimage
import sys
import os



def getBestShift(img):
    cy,cx = ndimage.measurements.center_of_mass(img)
    print cy,cx

    rows,cols = img.shape
    shiftx = np.round(cols/2.0-cx).astype(int)
    shifty = np.round(rows/2.0-cy).astype(int)

    return shiftx,shifty


def shift(img,sx,sy):
    rows,cols = img.shape
    M = np.float32([[1,0,sx],[0,1,sy]])
    shifted = cv2.warpAffine(img,M,(cols,rows))
    return shifted


x = tf.placeholder("float", [None, 784])

W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))



y = tf.nn.softmax(tf.matmul(x,W) + b)
y_ = tf.placeholder("float", [None,10])


cross_entropy = -tf.reduce_sum(y_*tf.log(y))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)



image = sys.argv[1]
train = False if len(sys.argv) == 2 else sys.argv[2]
checkpoint_dir = "cps/"

saver = tf.train.Saver()
sess = tf.Session()

sess.run(tf.initialize_all_variables())

if train:
    print("TRAIN!!!")
    # create a MNIST_data folder with the MNIST dataset if necessary
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

    # use 1000 batches with a size of 100 each to train our net
    for i in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        # run the train_step function with the given image values (x) and the real output (y_)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
    saver.save(sess, checkpoint_dir+'model.ckpt')
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})
else:
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        print 'No checkpoint found'
        exit(1)

    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print "accuracy: ", sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})



if not os.path.exists("img/" + image + ".png"):
    print "File img/" + image + ".png doesn't exist"
    exit(1)

# read original image
color_complete = cv2.imread("img/" + image + ".png")

print("read", "img/" + image + ".png")
# read the bw image
gray_complete = cv2.imread("img/" + image + ".png", 0)

# better black and white version
_, gray_complete = cv2.threshold(255-gray_complete, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

cv2.imwrite("pro-img/compl.png", gray_complete)

digit_image = -np.ones(gray_complete.shape)

height, width = gray_complete.shape

"""
crop into several images
"""
for cropped_width in range(100, 300, 20):
    for cropped_height in range(100, 300, 20):
        for shift_x in range(0, width-cropped_width, cropped_width/4):
            for shift_y in range(0, height-cropped_height, cropped_height/4):
                gray = gray_complete[shift_y:shift_y+cropped_height,shift_x:shift_x + cropped_width]
                if np.count_nonzero(gray) <= 20:
                     continue

                if (np.sum(gray[0]) != 0) or (np.sum(gray[:,0]) != 0) or (np.sum(gray[-1]) != 0) or (np.sum(gray[:,
                                                                                                            -1]) != 0):
                    continue

                top_left = np.array([shift_y, shift_x])
                bottom_right = np.array([shift_y+cropped_height, shift_x + cropped_width])

                while np.sum(gray[0]) == 0:
                    top_left[0] += 1
                    gray = gray[1:]

                while np.sum(gray[:,0]) == 0:
                    top_left[1] += 1
                    gray = np.delete(gray,0,1)

                while np.sum(gray[-1]) == 0:
                    bottom_right[0] -= 1
                    gray = gray[:-1]

                while np.sum(gray[:,-1]) == 0:
                    bottom_right[1] -= 1
                    gray = np.delete(gray,-1,1)

                actual_w_h = bottom_right-top_left
                if (np.count_nonzero(digit_image[top_left[0]:bottom_right[0],top_left[1]:bottom_right[1]]+1) >
                            0.2*actual_w_h[0]*actual_w_h[1]):
                    continue

                print "------------------"
                print "------------------"

                rows,cols = gray.shape
                compl_dif = abs(rows-cols)
                half_Sm = compl_dif/2
                half_Big = half_Sm if half_Sm*2 == compl_dif else half_Sm+1
                if rows > cols:
                    gray = np.lib.pad(gray,((0,0),(half_Sm,half_Big)),'constant')
                else:
                    gray = np.lib.pad(gray,((half_Sm,half_Big),(0,0)),'constant')

                gray = cv2.resize(gray, (20, 20))
                gray = np.lib.pad(gray,((4,4),(4,4)),'constant')


                shiftx,shifty = getBestShift(gray)
                shifted = shift(gray,shiftx,shifty)
                gray = shifted

                cv2.imwrite("pro-img/"+image+"_"+str(shift_x)+"_"+str(shift_y)+".png", gray)

                """
                all images in the training set have an range from 0-1
                and not from 0-255 so we divide our flatten images
                (a one dimensional vector with our 784 pixels)
                to use the same 0-1 based range
                """
                flatten = gray.flatten() / 255.0


                print "Prediction for ",(shift_x, shift_y, cropped_width)
                # print "Pos"
                # print top_left
                # print bottom_right
                # print actual_w_h
                # print " "
                #print flatten
                prediction = [tf.reduce_max(y),tf.argmax(y,1)[0]]
                pred = sess.run(prediction, feed_dict={x: [flatten]})
                #print pred


                digit_image[top_left[0]:bottom_right[0],top_left[1]:bottom_right[1]] = pred[1]

                cv2.rectangle(color_complete,tuple(top_left[::-1]),tuple(bottom_right[::-1]),color=(0,255,0),thickness=1)

                font = cv2.FONT_HERSHEY_SIMPLEX
                #value
                cv2.putText(color_complete,str(pred[1]),(top_left[1],bottom_right[0]+40), font,fontScale=1.4,color=(0,255,0),thickness=1)
                #percentage
                cv2.putText(color_complete,format(pred[0]*100,".1f")+"%",(top_left[1]+30,bottom_right[0]+50), font,fontScale=0.4,color=(0,255,0),thickness=1)



cv2.imwrite("img/"+image+"_digitized_image.png", color_complete)