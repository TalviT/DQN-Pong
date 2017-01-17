# pong_neural_net_live

##Overview
This is the code for the Live [Youtube](https://www.youtube.com/watch?v=Hqf__FlRlzg) session by @Sirajology. In this live session he built
the game of [Pong](https://en.wikipedia.org/wiki/Pong) from scratch. Then he built a [Deep Q Network](https://www.quora.com/Artificial-Intelligence-What-is-an-intuitive-explanation-of-how-deep-Q-networks-DQN-work) that gets better and better over time through trial and error. The DQN is a convolutional neural network that reads in pixel data from the game and the game score. Using just those 2 parameters, it learns what moves it needs to make to become better.

Because it did not work, I combined [Sirajology's](https://github.com/llSourcell/pong_neural_network_live) and [asrivat1's](https://github.com/asrivat1/DeepLearningVideoGames) code.
Now everything works as it should.

##Installation


* tensorflow (https://www.tensorflow.org)
* cv2 (http://www.pyimagesearch.com/2015/06/15/install-opencv-3-0-and-python-2-7-on-osx/)
* numpy
* random
* collections
* pygame

use [pip](https://pypi.python.org/pypi/pip) to install the dependencies. Tensorflow and cv2 are more manual. Links provided above ^

make sure to have the following file structure:

folder  
|--logs  
|--saved_networks  
|--pong.py  
|--RL.py

if the logs folder or the saved_networks folder are missing, create them by yourself.

##Usage 

Run it like this in terminal. The longer you let it run, the better it will get.

```
python RL.py
```

##Credits

This code was by [malreddysid](https://github.com/malreddysid) Siraj merely wrapped, updated, and documented it.


