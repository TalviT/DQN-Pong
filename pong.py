import pygame # helps us make GUI games in python
import random # help us define which direction the ball will start moving in
import os # for positioning of our window

#os.environ["SDL_VIDEODRIVER"] = "dummy"

#DQN. CNN reads in pixel data.
#reinforcement learning. trial and error.
#maximize action based on reward
#agent envrioment loop
#this is called Q Learning
#based on just game state. mapping of state to action is policy
#experience replay. learns from past policies

# size of our window
WINDOW_WIDTH = 400
WINDOW_HEIGHT = 400

#position of our window
WINDOW_X_POS = 300
WINDOW_Y_POS = 100

# size of our paddle
PADDLE_WIDTH = 10
PADDLE_HEIGHT = 60
# distance from the edge of the window
PADDLE_BUFFER = 10

# size of our ball
BALL_WIDTH = 10
BALL_HEIGHT = 10

# speed of our paddle and ball
PADDLE_SPEED = 4
BALL_X_SPEED = 6
BALL_Y_SPEED = 4

# RGB colors for our paddle and ball
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

os.environ['SDL_VIDEO_WINDOW_POS'] = "{},{}".format(WINDOW_X_POS, WINDOW_Y_POS)
# initialize our screen using width and height vars
pygame.init()
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
#screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.NOFRAME)
pygame.display.set_caption("Pong Deep Q-Learning")

# Paddle1 is our learning agent/us
# Paddle2 is the evil AI

# draw our ball
def DrawBall(ballXpos, ballYpos):
    # create and draw small rectangle
    ball = pygame.Rect(ballXpos, ballYpos, BALL_WIDTH, BALL_HEIGHT)
    pygame.draw.rect(screen, WHITE, ball)

# draw our paddle
def DrawPaddle1(paddle1Ypos):
    # create and draw rectangle
    paddle1 = pygame.Rect(PADDLE_BUFFER, paddle1Ypos, PADDLE_WIDTH, PADDLE_HEIGHT)
    pygame.draw.rect(screen, WHITE, paddle1)

# draw evil AI's paddle
def DrawPaddle2(paddle2Ypos):
    # create an draw rectangle
    paddle2 = pygame.Rect(WINDOW_WIDTH - PADDLE_BUFFER - PADDLE_WIDTH, paddle2Ypos, PADDLE_WIDTH, PADDLE_HEIGHT)
    pygame.draw.rect(screen, WHITE, paddle2)

# update the ball, using the paddle posistions the balls positions and the balls directions
def UpdateBall(paddle1Ypos, paddle2Ypos, ballXpos, ballYpos, ballXdirection, ballYdirection, ticker, evil_moves):
    # update the x and y position
    ballXpos += ballXdirection * BALL_X_SPEED
    ballYpos += ballYdirection * BALL_Y_SPEED
    score = 0

    # check for a collision, if the ball hits our learning agent's paddle
    if (ballXpos <= PADDLE_BUFFER + PADDLE_WIDTH) and (ballYpos + BALL_HEIGHT >= paddle1Ypos) and (ballYpos <= paddle1Ypos + PADDLE_HEIGHT):
        ballXdirection = 1 # the ball switches direction
        score = 1 # our agent will be "rewarded"
    # if it hits the left wall
    elif ballXpos <= 0:
        ballXpos = 0
        ballXdirection = 1 # the ball switches direction
        score = -1 # our agent will be "punished"

    # check if it hits the other side
    elif (ballXpos + BALL_WIDTH >= WINDOW_WIDTH - PADDLE_WIDTH - PADDLE_BUFFER) and (ballYpos + BALL_HEIGHT >= paddle2Ypos) and (ballYpos <= paddle2Ypos + PADDLE_HEIGHT):
        ballXdirection = -1
        evil_moves = 30
    elif ballXpos >= WINDOW_WIDTH - BALL_WIDTH:
        ballXpos = WINDOW_WIDTH - BALL_WIDTH
        ballXdirection = -1

    # if it hits the top move down
    if ballYpos <= 0:
        ballYpos = 0
        ballYdirection = 1
    # if it hits the bottom move up
    elif ballYpos >= WINDOW_HEIGHT - BALL_HEIGHT:
        ballYpos = WINDOW_HEIGHT - BALL_HEIGHT
        ballYdirection = -1

    # the players get multiple rewards if the ball hits the paddle from behind.
    # this is a workaround to make sure that the agent gets just one reward and not more
    # maybe not the best solution, but it works...
    if ticker > 0:
        score = 0
        ticker -= 1

    if score != 0 and ticker == 0:
        ticker = 10

    return [ score, paddle1Ypos, paddle2Ypos, ballXpos, ballYpos, ballXdirection, ballYdirection, ticker, evil_moves ]

# update the paddle position
def UpdatePaddle1(action, paddle1Ypos):
    # if move up
    if action[1] == 1:
        paddle1Ypos -= PADDLE_SPEED
    # if move down
    elif action[2] == 1:
        paddle1Ypos += PADDLE_SPEED

    # don't let it move off the screen!
    if paddle1Ypos < 0:
        paddle1Ypos = 0
    elif paddle1Ypos > WINDOW_HEIGHT - PADDLE_HEIGHT:
        paddle1Ypos = WINDOW_HEIGHT - PADDLE_HEIGHT

    return paddle1Ypos

def UpdatePaddle2(paddle2Ypos, ballYpos, evil_moves, evil_dir):
    # move if the ball is on the right side
    if evil_moves == 0:
        # move down ig ball is in upper half
        if paddle2Ypos + PADDLE_HEIGHT / 2 < ballYpos + BALL_HEIGHT / 2:
            paddle2Ypos += PADDLE_SPEED
        # move up if ball is in lower half
        elif paddle2Ypos + PADDLE_HEIGHT / 2 > ballYpos + BALL_HEIGHT / 2:
            paddle2Ypos -= PADDLE_SPEED
    # add some randomness to enemies movement
    # to prevent our agent from copying his moves
    else:
        if evil_dir != 0:
            paddle2Ypos += evil_dir * PADDLE_SPEED
        else:
            evil_dir = random.choice([-1, 1])
    
    if evil_moves > 0:
        evil_moves -= 1

    # don't let it move off the screen!
    if paddle2Ypos < 0:
        paddle2Ypos = 0
    elif paddle2Ypos > WINDOW_HEIGHT - PADDLE_HEIGHT:
        paddle2Ypos = WINDOW_HEIGHT - PADDLE_HEIGHT

    return paddle2Ypos, evil_moves, evil_dir

# game class
class PongGame:
    def __init__(self):
        self.ticker = 0 # bug workaround
        self.evil_moves = 0 # random moves for evilAI
        self.evil_dir = 0 # direction of evil's moves
        # initialize position of our paddle
        self.paddle1Ypos = WINDOW_HEIGHT / 2 - PADDLE_HEIGHT / 2
        self.paddle2Ypos = WINDOW_HEIGHT / 2 - PADDLE_HEIGHT / 2
        # ball starting point
        self.ballXpos = WINDOW_WIDTH / 2 - BALL_WIDTH / 2
        # randomly initialize ball direction
        self.ballXdirection = random.choice([-1, 1])
        self.ballYdirection = random.choice([-1, 1])
        # random number
        num = random.randint(0, 9)
        # where it will start, y part
        self.ballYpos = num * (WINDOW_HEIGHT - BALL_HEIGHT) / 9

    def GetFrame(self, action):
        pygame.event.pump()
        score = 0
        screen.fill(BLACK)
        # update our paddle
        self.paddle1Ypos = UpdatePaddle1(action, self.paddle1Ypos)
        DrawPaddle1(self.paddle1Ypos)
        # update evil AI paddle
        self.paddle2Ypos, self.evil_moves, self.evil_dir = UpdatePaddle2(self.paddle2Ypos, self.ballYpos, self.evil_moves, self.evil_dir)
        DrawPaddle2(self.paddle2Ypos)
        # update our vars by updatung ball position
        [score, self.paddle1Ypos, self.paddle2Ypos, self.ballXpos, self.ballYpos, self.ballXdirection, self.ballYdirection, self.ticker, self.evil_moves] = UpdateBall(self.paddle1Ypos, self.paddle2Ypos, self.ballXpos, self.ballYpos, self.ballXdirection, self.ballYdirection, self.ticker, self.evil_moves)
        # draw the ball
        DrawBall(self.ballXpos, self.ballYpos)
        # get the surface data
        image_data = pygame.surfarray.array3d(pygame.display.get_surface())
        # update the window
        pygame.display.flip()
        # return the score and the surface data
        return score, image_data
