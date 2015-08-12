# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions

import random, util

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """
    def __init__(self):
    	self.recentlyVisited = []

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"
        self.recentlyVisited.insert(0, gameState.generatePacmanSuccessor(legalMoves[chosenIndex]).getPacmanPosition)
        #keeps track of last 5 states
        if len(self.recentlyVisited)>5:
        	self.recentlyVisited.pop()

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        Food = 0
        #found food
        if newPos in newFood:
        	Food = 10
        
        numLeft = successorGameState.getNumFood()
        ghostDistances = 0
        #for evaluating ghost variable
        for ghost in newGhostStates:
        	ghostDistance =util.manhattanDistance(newPos, ghost.getPosition())
        	if ghostDistance == 0:
        		ghostDistance = 1e-9
        	elif ghostDistance<=3:
        		ghostDistances -=(1./ghostDistance)*100
        recentlyVisitedDistance = 0
        #tries to prevent repetition of states
        if newPos in self.recentlyVisited:
        	recentlyVisitedDistance = -200
        distances = []

        #finds closest distance to food
        for food in newFood.asList():
            distances.append(manhattanDistance(newPos, food))
        distances.sort()
        closestDistance = 0
        if len(distances)>0:
        	closestDistance = distances[0]

        return ghostDistances + recentlyVisitedDistance - 20 * numLeft - closestDistance

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """
    #finds maximinzing action
    def max_value(self, gameState, depth, agentIndex):
    	maxEval= float("-inf")
    	if gameState.isWin() or gameState.isLose():
    		return self.evaluationFunction(gameState)
    	for action in gameState.getLegalActions(0):
    		successor = gameState.generateSuccessor(0, action)
    		tempMaxEval = self.min_value(successor, depth, 1)
    		if maxEval < tempMaxEval:
    			maxEval = tempMaxEval
    			maxAction = action

    	if(depth == 1):
    		return maxAction
    	else:
    		return maxEval

    #finds minimizing action
    def min_value(self, gameState, depth, agentIndex):
    	minEval = float("inf")
    	if gameState.isWin() or gameState.isLose():
    		return self.evaluationFunction(gameState)
    	for action in gameState.getLegalActions(agentIndex):
    		successor = gameState.generateSuccessor(agentIndex, action)
    		if agentIndex ==gameState.getNumAgents()-1:
    			if depth == self.depth:
    				tempVal = self.evaluationFunction(successor)
    			else:
    				tempVal = self.max_value(successor, depth+1, 0)
    		else:
    			tempVal = self.min_value(successor, depth, agentIndex+1)
    		if tempVal < minEval:
    			minEval = tempVal

    	return minEval

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"
        minimaxAction = self.max_value(gameState, 1, 0)
        return minimaxAction

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """
    def max_pruning(self, gameState, depth, agentIndex, alpha, beta):
    	if gameState.isWin() or gameState.isLose():
    		return self.evaluationFunction(gameState)
    	maxVal = float("-inf")
    	for action in gameState.getLegalActions(0):
    		successor = gameState.generateSuccessor(0, action)
    		tempVal = self.min_pruning(successor, depth, 1, alpha, beta)
    		if tempVal > beta:
    			return tempVal

    		if tempVal>maxVal:
    			maxVal = tempVal
    			maxAction = action
    		alpha = max(maxVal, alpha)
    	if depth == 1:
    		return maxAction
    	else:
    		return maxVal

    def min_pruning(self, gameState, depth, agentIndex, alpha, beta):
    	if gameState.isWin() or gameState.isLose():
    		return self.evaluationFunction(gameState)
    	minVal = float("inf")
    	for action in gameState.getLegalActions(agentIndex):
    		successor = gameState.generateSuccessor(agentIndex, action)
    		if agentIndex == gameState.getNumAgents()-1:
    			if depth == self.depth:
    				tempVal =  self.evaluationFunction(successor)
    			else:
    				tempVal = self.max_pruning(successor, depth+1, 0, alpha, beta)
    		else:
    			tempVal = self.min_pruning(successor, depth, agentIndex+1, alpha, beta)
    		if tempVal < alpha:
    			return tempVal
    		if tempVal < minVal:
    			minVal = tempVal
    		beta = min(beta, minVal)
    	return minVal


    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        minimaxAction = self.max_pruning(gameState, 1, 0, float("-inf"), float("inf"))
        return minimaxAction

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """
    def max_value(self, gameState, depth, agentIndex):
    	if gameState.isWin() or gameState.isLose():
    		return self.evaluationFunction(gameState)
    	maxVal = float("-inf")
    	for action in gameState.getLegalActions(0):
    		successor = gameState.generateSuccessor(0, action)
    		tempVal = self.minimize(successor, depth, 1)
    		if tempVal>maxVal:
    			maxVal = tempVal
    			maxAction = action

    	if (depth==1):
    		return maxAction
    	else:
    		return maxVal

    def minimize(self, gameState, depth, agentIndex):
    	if gameState.isWin() or gameState.isLose():
    		return self.evaluationFunction(gameState)
    	minVal= 0
    	for action in gameState.getLegalActions(agentIndex):
    		successor = gameState.generateSuccessor(agentIndex, action)
    		if gameState.getNumAgents()-1 == agentIndex:
    			if depth == self.depth:
    				tempVal =  self.evaluationFunction(successor)
    			else:
    				tempVal = self.max_value(successor, depth+1, 0)
    		else:
    			tempVal = self.minimize(successor, depth, agentIndex+1)
    		minVal+=tempVal
    	minVal = float(minVal)
    	actionNum = len(gameState.getLegalActions(agentIndex))
    	actionNum = float(actionNum)
    	minVal = minVal / actionNum
    	return minVal




    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        expectimaxAction = self.max_value(gameState, 1, 0)
        return expectimaxAction

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
    score = currentGameState.getScore()

    Food = 0
    if newPos in newFood:
      	Food = 10
        
    numLeft = currentGameState.getNumFood()

    #for evaluation of ghosts
    ghostDistances = 0
    for ghost in newGhostStates:
        ghostDistance =util.manhattanDistance(newPos, ghost.getPosition())
        if ghostDistance == 0:
        	ghostDistance = 1e-9
        elif ghostDistance<=3:
        	ghostDistances -=(1./ghostDistance)*100
        if ghost.scaredTimer>ghostDistance:
        	ghostDistances+=200 - ghostDistance

    #finds closest distance from pacman position to food
    distances = []
    for food in newFood.asList():
        distances.append(manhattanDistance(newPos, food))
    distances.sort()
    closestDistance = 0
    if len(distances)>0:
    	closestDistance = distances[0]

    return score + ghostDistances - 20 * numLeft - closestDistance

# Abbreviation
better = betterEvaluationFunction

