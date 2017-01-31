import tensorflow as tf
import cv2 # read in pixel data
import pong # our class
import numpy as np # math
import random # random
from collections import deque # queue data structure. fast appends. and pops. replay memory

# definine hyperparameters
ACTIONS = 3 # up, down, stay
GAMMA = 0.99 # define our learning rate
INITIAL_EPSILON = 1.0 # for updating our gradient or training over time
FINAL_EPSILON = 0.05 # final value of epsilon
OBSERVE = 1000 # timesteps to observe before training
EXPLORE = 1000 # frames over which to anneal epsilon
REPLAY_MEMORY = 250000 # store our experiences, the size of it (test, how much your ram can fit!)
BATCH = 32 # batch size to train on
T_MAX = 1000000 # number of training iterations
S_MAX = 100 # the score our agent shall reach

# create tensorflow graph
def CreateGraph():
    # network weights
    W_conv1 = tf.Variable(tf.truncated_normal([8, 8, 4, 32], stddev=0.01))
    b_conv1 = tf.Variable(tf.constant(0.01, shape=[32]))

    W_conv2 = tf.Variable(tf.truncated_normal([4, 4, 32, 64], stddev=0.01))
    b_conv2 = tf.Variable(tf.constant(0.01, shape=[64]))

    W_conv3 = tf.Variable(tf.truncated_normal([3, 3, 64, 64], stddev=0.01))
    b_conv3 = tf.Variable(tf.constant(0.01, shape=[64]))

    W_fc4 = tf.Variable(tf.truncated_normal([1600, 512], stddev=0.01))
    b_fc4 = tf.Variable(tf.constant(0.01, shape=[512]))

    W_fc5 = tf.Variable(tf.truncated_normal([512, ACTIONS], stddev=0.01))
    b_fc5 = tf.Variable(tf.constant(0.01, shape=[ACTIONS]))

    # input layer for pixel data
    s = tf.placeholder("float", [None, 80, 80, 4])

    # Computes rectified linear unit activation fucntion (relu) on a 2-D convolution given 4-D input and filter tensors
    conv1 = tf.nn.relu(tf.nn.conv2d(s, W_conv1, strides=[1, 4, 4, 1], padding="SAME") + b_conv1)
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    conv2 = tf.nn.relu(tf.nn.conv2d(pool1, W_conv2, strides=[1, 2, 2, 1], padding="SAME") + b_conv2)
    conv3 = tf.nn.relu(tf.nn.conv2d(conv2, W_conv3, strides=[1, 1, 1, 1], padding="SAME") + b_conv3)

    conv3_flat = tf.reshape(conv3, [-1, 1600])
    fc4 = tf.nn.relu(tf.matmul(conv3_flat, W_fc4) + b_fc4)
    fc5 = tf.matmul(fc4, W_fc5) + b_fc5

    return s, fc5

# deep q network. feed in pixel data to graph session
def TrainGraph(inp, out, sess):
    # to calculate the argmax, we multiply the predicted output with a vector with one value 1 and rest as 0
    argmax = tf.placeholder("float", [None, ACTIONS])
    gt = tf.placeholder("float", [None]) #ground truth

    # action
    action = tf.reduce_sum(tf.mul(out, argmax), reduction_indices=1)
    # cost function we will reduce through backpropagation
    cost = tf.reduce_mean(tf.square(gt - action))
    # optimization function to reduce our minimize our cost function
    train_step = tf.train.AdamOptimizer(1e-6).minimize(cost)

    # initialize our game
    game = pong.PongGame()

    # create a queue for experience replay to store policies
    D = deque()
    
    # action do nothing
    argmax_t = np.zeros([ACTIONS])
    argmax_t[0] = 1
    # initial frame
    frame = game.GetFrame(argmax_t)[1]
    # convert rgb to gray scale for processing
    frame = cv2.cvtColor(cv2.resize(frame, (80, 80)), cv2.COLOR_BGR2GRAY)
    # binary colors, black or white
    ret, frame = cv2.threshold(frame, 1, 255, cv2.THRESH_BINARY)
    # stack frames, that is our input tensor
    inp_t = np.stack((frame, frame, frame, frame), axis=2)

    # saver
    saver = tf.train.Saver()
    sess.run(tf.initialize_all_variables())
    checkpoint = tf.train.get_checkpoint_state("saved_networks")
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print("Successfully loaded:", checkpoint.model_checkpoint_path)
    else:
        print("Could not find saved networks")

    stats_log = open("logs/stats.log", "w")
    # total game score
    score = 0

    t = 0
    epsilon = INITIAL_EPSILON

    # training time
    while True:
        # output tensor
        out_t = out.eval(feed_dict={ inp: [inp_t] })[0]
        # argmax function
        argmax_t = np.zeros([ACTIONS])

        if random.random() <= epsilon or t <= OBSERVE:
            maxIndex = random.randrange(ACTIONS)
            r_dec = "True" # optional for logging, True if randomly decided
        else:
            maxIndex = np.argmax(out_t)
            r_dec = "False"
        argmax_t[maxIndex] = 1

        # scale down epsilon
        if epsilon > FINAL_EPSILON and t > OBSERVE:
            epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

        # reward tensor if score is positive
        reward_t, frame = game.GetFrame(argmax_t)
        # get frame pixel data
        frame = cv2.cvtColor(cv2.resize(frame, (80, 80)), cv2.COLOR_BGR2GRAY)
        ret, frame = cv2.threshold(frame, 1, 255, cv2.THRESH_BINARY)
        frame = np.reshape(frame, (80, 80, 1))

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

        # print out where we are
        if t <= OBSERVE:
            state = "observe"
        elif OBSERVE < t < OBSERVE + EXPLORE:
            state = "explore"
        else:
            state = "train"

        score += reward_t

        stats = "TIMESTEP {:7} | SCORE: {: 5} | STATE {:7} | EPSILON {:6.4f} | ACTION {} | R_DEC {:5} | REWARD {:2d} | Q_MAX {: e}".format(t, score, state, epsilon, maxIndex, r_dec, reward_t, np.max(out_t))
        print(stats)
        # write into file
        stats_log.write(stats + "\n")

        #save images
        #if t % 10000 <= 100:
        #    cv2.imwrite("logs/images/frame" + str(t) + ".png", frame)


        # save our session every 10000 steps
        if t % 10000 == 0:
            saver.save(sess, "saved_networks/pong_game-dqn.chk", global_step=t)
            print("Session saved.")

        if t == T_MAX:
            return
        #if score == S_MAX:
        #    return


def Main():
    try:
        #create session
        sess = tf.InteractiveSession()
        #input layer and output layer by creating graph
        inp, out = CreateGraph()
        #train our graph on input and output with session variables
        TrainGraph(inp, out, sess)
    except KeyboardInterrupt:
        print("Closing Session...")
    sess.close()
    exit()

if __name__ == "__main__":
    Main()
