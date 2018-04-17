import gym
import numpy as np
import time
import cv2
import os
import pickle
import traceback
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import RMSprop, SGD


HIDDEN_LAYER1=164
HIDDEN_LAYER2=150
GAMMA=0.99


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def debug(*message):
    print bcolors.WARNING+"[debug message]"
    for msg in message:
        print msg
    print "[done]"+bcolors.ENDC

class ReplayBuffer():
    def __init__(self,buffer_size=40):
        self._buffer = list()
        self.buffer_size=buffer_size

    def add(self,value):
        self._buffer.append(value)
        if len(self._buffer) > self.buffer_size:
            self._buffer.pop(0)
    
    def getBuffer(self):
        return self._buffer,(len(self._buffer)>sel.buffer_size-1)   

class QlearningAgent():
    def __init__(self,env,learning_rate=0.01):
        self.env = env
        self.learning_rate=learning_rate
        self.createNetwork()

    def createNetwork(self):
        self.model = Sequential()
        self.model.add(Dense(HIDDEN_LAYER1, kernel_initializer='lecun_uniform', input_shape=(self.env.observation_space.n,)))
        self.model.add(Activation('relu'))

        self.model.add(Dense(HIDDEN_LAYER2, kernel_initializer='lecun_uniform'))
        self.model.add(Activation('relu'))

        self.model.add(Dense(self.env.action_space.n, kernel_initializer='lecun_uniform'))
        self.model.add(Activation('relu'))
            
        sgd = SGD(lr=self.learning_rate, decay=1e-6, momentum=0.9, nesterov=True)
        self.model.compile(loss='mse', optimizer=sgd)

    def train(self,Replaybuffer):
        _buffer,trainFlag = Replaybuffer.getBuffer()
        if trainFlag:
            X = [buff[0] for buff in _buffer]
            y = [buff[1] for buff in _buffer]
            self.model.fit(X,y,batch_size=20,epochs=1)

    def oneHot(self,value):
        vect = np.zeros((1,self.env.observation_space.n))
        vect[:,value]=1
        return vect

    def getAction(self,observation):
        # print (np.shape(observation))
        # observation = np.array(observation).reshape(16,)
        observation = self.oneHot(observation)
        actions = self.model.predict(observation)
        debug(actions)
        return np.argmax(actions)


if __name__ == '__main__':
        
    env = gym.make('FrozenLake-v0')

    Ql = QlearningAgent(env)
    buff = ReplayBuffer()

    try:
        time.sleep(1)
        num_episodes = 2000
        # create lists to contain total rewards and steps per episode
        #jList = []
        rList = []
        for episode in range(num_episodes):
            # Reset environment and get first new observation
            oldObs = env.reset()
            rAll = 0
            done = False
            j = 0
            while j < 99:
                j += 1
                os.system('clear')
                env.render()
                # action = Ql.takeAction(oldObs,episode)
                # action = np.random.randint(0,4)
                print "oldObs",oldObs
                action = Ql.getAction(oldObs)
                debug("action",action)
                newObs, reward, done, _ = env.step(action)
                target = reward + GAMMA*action
                # Ql.train()                
                
                # Ql.updateQtable(oldObs, newObs, action, reward)

                # Ql.displayQTable()
                print "Score over time: " + str(sum(rList)/num_episodes)
                print "Game over:",done
                print "Episode: ",episode

                rAll += reward
                oldObs = newObs
                if done == True:
                    break
                time.sleep(0.1)

            rList.append(rAll)
            # print "Final Q-Table Values"
            # print Q
            time.sleep(0.1)
    except Exception as e:
        print(dir(e))
        print bcolors.FAIL
        traceback.print_exc()
        print bcolors.ENDC
        # Ql.saveQtable()
        quit()