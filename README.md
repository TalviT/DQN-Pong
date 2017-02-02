# DQN-pong

## Overview

This code is a forked version of Sirajology's [pong_neural_net_live](https://github.com/llSourcell/pong_neural_network_live) project. In a live session he built the game [Pong](https://en.wikipedia.org/wiki/Pong) from scratch. Then he built a [Deep Q Network](https://www.quora.com/Artificial-Intelligence-What-is-an-intuitive-explanation-of-how-deep-Q-networks-DQN-work) that gets better over time through trial and error. The DQN is a convolutional neural network that uses the pixel data and the game score as input parameters. Through reinforcement learning, it learns what moves it needs to make to become better.

Because the code from the original project did not work, I began to fix several bugs, by combining Sirajology's and [asrivat1's](https://github.com/asrivat1/DeepLearningVideoGames) code. Now everything works as it should.

## Installation

Dependencies:
* [tensorflow](https://www.tensorflow.org/)
* [cv2](http://opencv.org/)
* [numpy](http://www.numpy.org/)
* [pygame](https://www.pygame.org/)

Use [pip](https://pypi.python.org/pypi/pip/) to install the dependencies. For tensorflow and cv2 follow the instructions on the project websites.

make sure to have the following file structure:

DQN-pong  
|-- logs/  
|-- saved_networks/  
|-- RL.py  
|-- pong.py

if "logs/" or "saved_networks/" are missing, create them by yourself.

## Usage

Run it like this in terminal. It will take about 1,000,000 Timesteps until the ai plays almost perfect.

`python RL.py`
