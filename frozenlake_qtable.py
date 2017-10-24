import gym
import numpy as np
import time
import cv2
import os
import pickle
import traceback


class Qlearner():

    def __init__(self, env, learningRate, discount):
        self.learningRate = learningRate
        self.discount = discount
        self.env = env
        self.Qtable = np.zeros(
            [self.env.observation_space.n, self.env.action_space.n])
        # print type(self.Qtable)
        
        if os.path.exists("./qtable.dat"):
            data = np.fromfile('./qtable.dat', dtype=float).reshape(self.env.observation_space.n, self.env.action_space.n)
            if isinstance(data,np.ndarray) and (np.shape(data)==np.shape(self.Qtable)):
                self.Qtable=data
            else:
                print isinstance(data,np.ndarray)
                print np.shape(data)
                print np.shape(self.Qtable)
                print "qtable.dat is corrupt"
                time.sleep(2)
        else:
            print "No qtable.dat found"        


    def createQtable(self):
        return np.zeros([
            self.env.observation_space.n, self.env.action_space.n])

    def takeAction(self, observation,episode):
        return np.argmax(self.Qtable[observation, :] + np.random.randn(
            1, env.action_space.n)*(1./(episode+1)))

    def updateQtable(self, oldObservation, newObservation, action, reward):
        self.Qtable[oldObservation, action] = self.Qtable[
            oldObservation, action] + self.learningRate * (
            reward + self.discount * np.max(
                self.Qtable[newObservation, :]) - self.Qtable[
                oldObservation, action])
        # Q[oldObservation,action] = (1 - learningRate)*Q[oldObservation,action] + learningRate*(reward + gamma*Q[newObservation,np.argmax(Q[newObservation,action])])

    def displayQTable(self):
        print'-'*100
        print " "*46+"Q-Table"
        print'-'*100
        print self.Qtable
        print'-'*100

    def saveQtable(self):
        self.Qtable.tofile('./qtable.dat')
        print "Qtable saved to qtable.dat"





if __name__ == '__main__':
        
    env = gym.make('FrozenLake-v0')

    Ql = Qlearner(env, learningRate=0.85, discount=0.99)

    # Q = Ql.createQtable(env)

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
                action = Ql.takeAction(oldObs,episode)
                newObs, reward, done, _ = env.step(action)
                Ql.updateQtable(oldObs, newObs, action, reward)

                Ql.displayQTable()
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
    except:
        traceback.print_exc()
        Ql.saveQtable()
        quit()
