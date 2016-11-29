import tensorflow as tf
import cv2 # read in pixel data
import pong # our class
import numpy as np # math
import random # random
from collections import deque # queue data structure. fast appends. and pops. replay memory
import pickle # save variables
import os.path

# definine hyperparameters
ACTIONS = 3 # up, down, stay
# define our learning rate
GAMMA = 0.99
# for updating our gradient or training over time
INITIAL_EPSILON = 1.0
FINAL_EPSILON = 0.05
# how many frames to anneal epsilon
EXPLORE = 500000
OBSERVE = 50000
# store our experiences, the size of it
REPLAY_MEMORY = 500000
# batch size to train on
BATCH = 100

# create tensorflow graph
def CreateGraph():
    # first convolutional layer. bias vector
    # creates an empty tensor with all elements set to zero with a shape
    W_conv1 = tf.Variable(tf.zeros([8, 8, 4, 32]))
    B_conv1 = tf.Variable(tf.zeros([32]))
    # second layer
    W_conv2 = tf.Variable(tf.zeros([4, 4, 32, 64]))
    B_conv2 = tf.Variable(tf.zeros([64]))
    # third layer
    W_conv3 = tf.Variable(tf.zeros([3, 3, 64, 64]))
    B_conv3 = tf.Variable(tf.zeros([64]))
    # fouth layer
    W_fc4 = tf.Variable(tf.zeros([3136, 784]))
    B_fc4 = tf.Variable(tf.zeros([784]))
    # last layer
    W_fc5 = tf.Variable(tf.zeros([784, ACTIONS]))
    B_fc5 = tf.Variable(tf.zeros([ACTIONS]))

    # input for pixel data
    s = tf.placeholder("float", [None, 84, 84, 4])

    # Computes rectified linear unit activation fucntion on a 2-D convolution given 4-D input and filter tensors
    conv1 = tf.nn.relu(tf.nn.conv2d(s, W_conv1, strides=[1, 4, 4, 1], padding="VALID") + B_conv1)
    conv2 = tf.nn.relu(tf.nn.conv2d(conv1, W_conv2, strides=[1, 2, 2, 1], padding="VALID") + B_conv2)
    conv3 = tf.nn.relu(tf.nn.conv2d(conv2, W_conv3, strides=[1, 1, 1, 1], padding="VALID") + B_conv3)

    conv3_flat = tf.reshape(conv3, [-1, 3136])
    fc4 = tf.nn.relu(tf.matmul(conv3_flat, W_fc4 + B_fc4))
    fc5 = tf.matmul(fc4, W_fc5) + B_fc5

    return s, fc5

# deep q network. feed in pixel data to graph session
def TrainGraph(inp, out, sess):
    # to calculate the argmax, we multiply the predicted output with a vector with one value 1 and rest as 0
    argmax = tf.placeholder("float", [None, ACTIONS])
    gt = tf.placeholder("float", [None]) #ground truth

    # action
    action = tf.reduce_sum(tf.mul(out, argmax), reduction_indices=1)
    # cost function we will reduce through backpropagation
    cost = tf.reduce_mean(tf.square(action - gt))
    # optimization function to reduce our minimize our cost function
    train_step = tf.train.AdamOptimizer(1e-6).minimize(cost)

    # initialize our game
    game = pong.PongGame()

    # create a queue for experience replay to store policies
    D = deque()

    # initial frame
    frame = game.GetPresentFrame()
    # convert rgb to gray scale for processing
    frame = cv2.cvtColor(cv2.resize(frame, (84, 84)), cv2.COLOR_BGR2GRAY)
    # binary colors, black or white
    ret, frame = cv2.threshold(frame, 1, 255, cv2.THRESH_BINARY)
    # stack frames, that is our input tensor
    inp_t = np.stack((frame, frame, frame, frame), axis=2)

    # saver
    saver = tf.train.Saver()

    if os.path.isfile("./python_vars.pickle"):
        saver.restore(sess, "./pong_game-dqn.chk")
        with open("./python_vars.pickle", "rb") as f:
            t, epsilon, D = pickle.load(f)
        print("Session loaded.")
    else:
        sess.run(tf.initialize_all_variables())
        t = 0
        epsilon = INITIAL_EPSILON

    # training time
    while True:
        # output tensor
        out_t = out.eval(feed_dict={ inp: [inp_t] })[0]
        # argmax function
        argmax_t = np.zeros([ACTIONS])

        if random.random() <= epsilon:
            maxIndex = random.randrange(ACTIONS)
        else:
            maxIndex = np.argmax(out_t)
        argmax_t[maxIndex] = 1

        if epsilon > FINAL_EPSILON:
            epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

        # reward tensor if score is positive
        reward_t, frame = game.GetNextFrame(argmax_t)
        # get frame pixel data
        frame = cv2.cvtColor(cv2.resize(frame, (84, 84)), cv2.COLOR_BGR2GRAY)
        ret, frame = cv2.threshold(frame, 1, 255, cv2.THRESH_BINARY)
        frame = np.reshape(frame, (84, 84, 1))
        # new input tensor
        inp_t1 = np.append(frame, inp_t[:, :, 0:3], axis=2)

        # add our input tensor, argmax tensor, reward and updated tensor to stack of experiences
        D.append((inp_t, argmax_t, reward_t, inp_t1))

        # if we run out of replay memory, make room
        if len(D) > REPLAY_MEMORY:
            D.popleft()

        # training iteration
        if t > OBSERVE:
            # get values from our replay memory
            minibatch = random.sample(D, BATCH)

            inp_batch = [d[0] for d in minibatch]
            argmax_batch = [d[1] for d in minibatch]
            reward_batch = [d[2] for d in minibatch]
            inp_t1_batch = [d[3] for d in minibatch]

            gt_batch = []
            out_batch = out.eval(feed_dict={ inp: inp_t1_batch })

            # add values to our batch
            for i in range(0, len(minibatch)):
                gt_batch.append(reward_batch[i] + GAMMA * np.max(out_batch[i]))

            # train on that
            train_step.run(feed_dict={
                gt: gt_batch,
                argmax: argmax_batch,
                inp: inp_batch
            })

        # update our input tensor the next frame
        inp_t = inp_t1
        t += 1

        # print out where we are after saving where we are
        if t % 10000 == 0:
            saver.save(sess, "./pong_game-dqn.chk")
            with open("./python_vars.pickle", "wb") as f:
                pickle.dump([t, epsilon, D], f)
            print("Session saved.")

        print("TIMESTEMP {} / EPSILON {} / ACTION {} / REWARD {} / Q_MAX {:e}".format(t, epsilon, maxIndex, reward_t, np.max(out_t)))

def Main():
    #create session
    sess = tf.InteractiveSession()
    #input layer and output layer by creating graph
    inp, out = CreateGraph()
    #train our graph on input and output with session variables
    TrainGraph(inp, out, sess)

if __name__ == "__main__":
    Main()
