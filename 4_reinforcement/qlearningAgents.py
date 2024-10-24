# qlearningAgents.py
# ------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import gridworld

import random,util,math
import copy

class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent
      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update
      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)
      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """
    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)

        "*** YOUR CODE HERE ***"
        self.QVals = util.Counter()

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        "*** YOUR CODE HERE ***"
        s = state
        a = action
        
        if(s,a) not in self.QVals:
            self.QVals[s,a] = 0.0
            return self.QVals[s,a]
        
        return self.QVals[(s,a)]
        util.raiseNotDefined()

    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        "*** YOUR CODE HERE ***"
        
        LA = self.getLegalActions(state)
        if len(LA) == 0:
            return 0.0
        
        return max([self.getQValue(state, a) for a in self.getLegalActions(state)])

        util.raiseNotDefined()

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        "*** YOUR CODE HERE ***"
        
        LA = self.getLegalActions(state)
        if len(LA) == 0:
            return None
        
        CA = LA[0]
        CAV = float('-inf')
        
        for a in LA:
            v = self.getQValue(state, a)
            if CAV < v:
                CAV = v
                CA = a
            
            elif CAV == v:
                CA, CAV = random.choice([(CA, CAV),(a,v)])

        return CA
        util.raiseNotDefined()

    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.
          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        # Pick Action
        legalActions = self.getLegalActions(state)
        action = None
        "*** YOUR CODE HERE ***"
        
        if len(legalActions) == 0: return None
        
        elif util.flipCoin(self.epsilon):
            return random.choice(legalActions)
        
        else: return self.computeActionFromQValues(state)

        util.raiseNotDefined()

        return action

    def update(self, state, action, nextState, reward: float):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here
          NOTE: You should never call this function,
          it will be called on your behalf
        """
        "*** YOUR CODE HERE ***"
        
        #sample = reward + self.discount * self.computeValueFromQValues(nextState)
        
        #for i in range(5):  
        sample = reward + self.discount * self.getValue(nextState)
        #self.QVals[(state, action)] = (1 - self.alpha) * self.QVals[(state, action)] + self.alpha * sample
        self.QVals[(state, action)] = (1 - self.alpha) * self.getQValue(state,action) + self.alpha * sample
            
        #print("Sample: ", sample)
            
        #print(self.QVals)
        #util.raiseNotDefined()

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1
        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self,state)
        self.doAction(state,action)
        return action

class ApproximateQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent
       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """
    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()

    def getWeights(self):
        return self.weights

  
    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        "*** YOUR CODE HERE ***"
       
        #print("\ngetQValue is ran with:", (state, action))
        #print("Episode: ", self.episodesSoFar)
        
        '''
        SF = SimpleExtractor.getFeatures(SimpleExtractor, state, action)
        bias = SF["bias"]
        GhostsAStepAway = SF["#-of-ghosts-1-step-away"]
        Eat = SF["eats-food"]
        closestFood = SF["closest-food"]

        CF = CoordinateExtractor.getFeatures(CoordinateExtractor, state, action)
        xStance = CF['x=%d' % state[0]]
        yStance = CF['y=%d' % state[0]]
        aStance = CF['action=%s' % action]

        IE = IdentityExtractor.getFeatures(IdentityExtractor, state, action)
        IEFeats = IE[(state, action)] 

        features = [bias, GhostsAStepAway, Eat, closestFood, xStance, yStance, aStance, IEFeats]
        
        
        #WD = self.getWeights()
        
        
        #WD = self.getWeights()[(state,action)]
        #print("WD Before Calculation: ", WD)
        #print(type(self.getWeights()[(state,action)]))
        #if self.getWeights()[(state,action)].all() == 0.0:
        #if (state,action) not in self.getWeights().keys():
        #if type(self.getWeights()[(state,action)]) is int:
            
        
            #n = len(features)
            #print("Length of FD: ", len(FD))
            #n = []
            for i in range(len(features)):
                n.append(1)
            WD = n
            self.weights[(state,action)] = WD
            #print("WD: ", WD)
        
        
        
        import numpy as np
        
        FDv = np.array(list(FD.values()))
        WDv = np.array(WD)
        #print("FDv:",  FDv)
        #print("WDv:",  WDv)
        #self.QVals[(state, action)] = FD[(state,action)] * WD[(state,action)]
        #self.QVals[(state, action)] = np.dot(WDv, FDv)
        self.QVals[(state, action)] = np.dot(FDv, WDv)
        #FD[(state,action)] += FD[(state,action)] * WD[(state,action)]
        #FDv = list(FD.values())
        #WDv = WD
        #print("FDv:",  FDv)
        #print("WDv:",  WDv)
        
        #self.QVals[(state, action)] = sum([FDv[i] * WDv[i] for i in range(len(FDv))])
        #self.QVals[(state, action)] = sum([FDv[i] * WDv[i] for i in range(len(FDv))])
        
        #SF = SimpleExtractor.getFeatures(self, state, action)

        #print(FE.keys())
        #QV = sum([features[i] * WD[i] for i in range(len(WD))])
        #features = self.getFeatures(state, action)
        
        #QV = 0

        #self.QVals[(state,action)] = 0.0
        
        '''


        QV = 0
        FE = self.featExtractor.getFeatures(state, action)
        W = self.getWeights()
        '''
        #print(self.QVals.keys())
        #if(state,action) not in self.QVals.keys():
        #    print("Populating QVal of: ", (state, action))    
        #    self.QVals[(state,action)] = 0.0
            #return self.QVals[(state,action)]
        

        #if action == 'exit':
            #return self.QVals[(state,action)]

        #import numpy as np
        #if len(list(self.getWeights().values())) < 1:
        #    return 0.0    
        #self.QVals[(state,action)] += np.dot(list(self.getWeights().values()), list(FE.values()))
        #self.QVals[(state,action)] = list(FE) @ list(self.getWeights())
        '''
        for k in FE.keys():
            #print("Curr QVal: ", self.QVals[(state,action)])

            #self.QVals[(state,action)] += FE[k] * self.getWeights()[k]    
            #QV += FE[k] * self.getWeights()[k]    
            QV += FE[k] * W[k]    
            #print("Curr Key: ", k)
            #print("Curr FeatureValue: ", FE[k])
            #print("Curr Weight: ", self.getWeights()[k])
            
        '''
        #QV += SF['bias'] * self.weights['bias']
        #QV += features[0] * self.weights['bias']
        #QV += SF['#-of-ghosts-1-step-away'] * self.weights['#-of-ghosts-1-step-away']
        #QV += features[1] * self.weights['#-of-ghosts-1-step-away']
        #QV += SF['eats-food'] * self.weights['eats-food']
        #QV += features[2] * self.weights['eats-food']
        #QV += SF['closest-food'] * self.weights['closest-food']
        #QV += features[3] * self.weights['closest-food']
        
        #QV +=
        #QV += features[4] * self.weights['x=%d' % state[0]]
        #QV += features[5] * self.weights['y=%d' % state[0]]
        #QV += features[6] * self.weights['action=%s' % action]

        #return QV

        #print("Putting into QVals: ", self.QVals[(state,action)])
        '''
        
        
        #return self.QVals[(state,action)]
        return QV
        #util.raiseNotDefined()

    def update(self, state, action, nextState, reward: float):
        """
           Should update your weights based on transition
        """
        "*** YOUR CODE HERE ***"
        #print("\nupdate is ran")
        '''
        #qv = self.getQValue(state, action)
        #self.get
        
        #AvailableActs = self.getLegalActions(nextState)
        
        #print("AvailableActs: ", self.getLegalActions(nextState))
        #if len(AvailableActs) > 0:

        #for (s,a) in self.getWeights().keys():
        #currentFeatures = SimpleExtractor.getFeatures(self, state, action)
        #cFV = list(currentFeatures.values())
        #for (s,a) in currentFeatures.keys():
            

            
        #cFV = list(SF.values())
        #if(len(self.getLegalActions(nextState)) > 0):     
        #if 'exit' not in self.getLegalActions(state):
        
        #print("\nQNext Calc:")        
        #QNext = max([self.getQValue(nextState,act) for act in self.getLegalActions(nextState)] + [0])
        #if action == 'exit':
        #    return
        #q = [self.QVals[(nextState,act)] for act in self.getLegalActions(nextState)]
        '''
        #q = [self.QVals[(nextState,act)] for act in self.getLegalActions(nextState)] + [0]
        #LS = self.getLegalActions(nextState)
        #QNext = max([self.getQValue(nextState,act) for act in self.getLegalActions(nextState)] + [0])
        #print(q)
        #QNext = max(q)
        #Bellman = reward + self.discount * QNext
        #print("\nDifference Calc:")
        #difference = Bellman - self.getQValue(state, action)
        #difference = Bellman - self.QVals[(state,action)]

        difference = (reward + self.discount * self.computeValueFromQValues(nextState)) - self.getQValue(state,action)

        #print("Acts of next state ", nextState, ":", self.getLegalActions(nextState))
        #print("Alpha: ", self.alpha)
        #print("Reward: ",reward)
        #print("Discount: ",self.discount)
        #print("Max of QNext:",QNext)
        #print("Bellman:",Bellman)
        #print("Difference: ", difference)
        '''
        #print("cFV: ",cFV)            
            
        #features = self.getFeatures(state, action)
        #QV = 0
        #QV += features[0] * self.weights['bias']
        #QV += features[1] * self.weights['#-of-ghosts-1-step-away']
        #QV += features[2] * self.weights['eats-food']
        #QV += features[3] * self.weights['closest-food']
        #QV += features[4] * self.weights['x=%d' % state[0]]
        #QV += features[5] * self.weights['y=%d' % state[0]]
        #QV += features[6] * self.weights['action=%s' % action]

        #features = self.getFeatures(state, action)
            
        #for k in self.getWeights().keys():
        '''
        for k in self.featExtractor.getFeatures(state,action).keys():
        #for k in self.getWeights().keys():
            #print("Adjusting Weight for key:", k)    
            #print("Weights: ", self.getWeights())
        
            #self.weights[k] = self.getWeights()[k] + (self.alpha * difference * self.featExtractor.getFeatures(state, action)[k])
                
            #this doesn't work for some reason...
            if(k in self.featExtractor.getFeatures(state,action).keys()):
                self.weights[k] = self.weights[k] + (self.alpha * difference * self.featExtractor.getFeatures(state, action)[k])
                
        '''
        #self.weights['bias'] = self.weights['bias'] + self.alpha * difference * features[0]
        #self.weights['#-of-ghosts-1-step-away'] = self.weights['#-of-ghosts-1-step-away'] + self.alpha * difference * features[1]
        #self.weights['eats-food'] = self.weights['eats-food'] + self.alpha * difference * features[2]
        #self.weights['closest-food'] = self.weights['closest-food'] + self.alpha * difference * features[3]
        #self.weights['x=%d' % state[0]] = self.weights['x=%d' % state[0]] + self.alpha * difference * features[4]
        #self.weights['y=%d' % state[0]] = self.weights['y=%d' % state[0]] + self.alpha * difference * features[5]
        #self.weights['action=%s' % action] = self.weights['action=%s' % action] + self.alpha * difference * features[6]
        
      
        w = self.weights[(state,action)]                 
        for i in range(len(self.getWeights())):
            w[i] =  w[i] + self.alpha * difference * cFV[i]
        
            
            

        #print(self.getWeights()[(s,a)])
        #print((self.alpha * difference * currentFeatures.values()))
            
            

        #self.weights[(s,a)] = self.getWeights()[(s,a)] + (self.alpha * difference * currentFeatures[(s,a)])
                
        #print("current sa: ", (s,a))
        #print("Max QV: ", Bellman)
        #print("Curr Weight: ", self.getWeights()[(state,action)])
        #self.weights[i] = self.weights[i] + (self.alpha * difference + currentFeatures[i])
        
        #print("CurrentFeatures: ", self.featExtractor.getFeatures(state,action))
        '''
        #print("Final Weights: ", self.getWeights())
        #util.raiseNotDefined()

    def final(self, state):
        """Called at the end of each game."""
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            #print("Final Weights: ", self.getWeights())
            pass
