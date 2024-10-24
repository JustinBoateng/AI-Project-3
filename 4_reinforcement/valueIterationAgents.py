# valueIterationAgents.py
# -----------------------
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


# valueIterationAgents.py
# -----------------------
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


from winreg import SetValue
import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp: mdp.MarkovDecisionProcess, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        """
          Run the value iteration algorithm. Note that in standard
          value iteration, V_k+1(...) depends on V_k(...)'s.
        """
        "*** YOUR CODE HERE ***"

        StateList = self.mdp.getStates()
        
        #print(StateList)
        for i in range(self.iterations):
            print("-----------------------------------------------------------------------------------------Iteration #", i, "-------------------------------------------------------------------------------------------------------------")    
            print("-----------------------------------------------------------------------------------------Iteration #", i, "-------------------------------------------------------------------------------------------------------------")    
            print("-----------------------------------------------------------------------------------------Iteration #", i, "-------------------------------------------------------------------------------------------------------------")    
            ValueDict = self.values.copy()
            
            for j in range(len(StateList)):
                currState = StateList[j] 
                #self.dispow = 0       
                print("\n")
                U = 0
                if(len(self.mdp.getPossibleActions(currState)) > 0):     
                    #while(True):     
                    #print("Not Terminal")
                    #print(self.mdp.getPossibleActions(currState))
                    #print("CS: ",currState)
                    #policy = self.getAction(currState)
                    #print("Correct Act: ", policy)
                    
                    #if(policy != "exit"):
                    #    currState = self.mdp.getTransitionStatesAndProbs(currState, policy)[0][0]
                        #self.dispow+=1
                    #else:
                    #    break
                    acts = self.mdp.getPossibleActions(currState)
                    U = max([self.getQValue(currState, a) for a in acts])
                    ValueDict[currState] = U
                    
            
            self.values = ValueDict
            #self.values[currState] = U
            #We want to clone the dictionary so that we update the Values ONLY when the iteration is done.
                
                #else:
                #    print("Terminal")
                #    print(self.mdp.getPossibleActions(currState))
                #    print("CS: ",currState)
                    
                    #self.setValues(self.mdp.getReward(currState, exit, t[0]))
                    #policy = self.getAction(currState)
                    

        print(self.values)
        #for i in self.iterations:    
        #self.setValues[self.getAction()]    

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]

    def setValues(self, state, v):
        self.values[state] = v

    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        curr = state
        u = 0
        T = self.mdp.getTransitionStatesAndProbs(curr, action)
       

        for t in T:
            #print(self.values)
            print("From State,", curr,  ", Taking Move: ", action, " to current t: ", t, " with value: ", self.getValue(t[0]), "and reward: ", self.mdp.getReward(curr,action, t[0]))
            u += t[1] * (self.mdp.getReward(curr,action,t[0]) + self.discount * self.getValue(t[0]))
            
        print("Returning: ", u)
        return u

        util.raiseNotDefined()

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        
        act = 'south'
        v = float("-inf")
        #v = -99999
        Qv = float("-inf")
        #v = 0
        for a in self.mdp.getPossibleActions(state):
            print("Testing Move: ",a)
            #if a = exit, then this is a terminal state
            Qv = self.getQValue(state, a)
            print("State: ", state, ", Qv: ", Qv)
            print("Comparing v: ", v, ", to Qv: ", Qv)
            if v < Qv:
                print("updating a to: ", a)
                act = a
                v = Qv
        #self.setValues(state, v)
    
        return act        
                

        #We're returning an action here.
        #The action returned should be the best action to take at state
        util.raiseNotDefined()
        

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        #print(state)
        return self.computeQValueFromValues(state, action)
