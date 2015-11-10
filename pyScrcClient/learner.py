import collections, math, random
from datetime import datetime
from numpy import arange

# An abstract class representing a Markov Decision Process (MDP).
class MDP:
    # Return the start state.
    def startState(self): raise NotImplementedError("Override me")

    # Return set of actions possible from |state|.
    def actions(self, state): raise NotImplementedError("Override me")

    # Return a list of (newState, prob, reward) tuples corresponding to edges
    # coming out of |state|.
    # Mapping to notation from class:
    #   state = s, action = a, newState = s', prob = T(s, a, s'), reward = Reward(s, a, s')
    # If IsEnd(state), return the empty list.
    def succAndProbReward(self, state, action): raise NotImplementedError("Override me")

    def discount(self): raise NotImplementedError("Override me")

    # Compute set of states reachable from startState.  Helper function for
    # MDPAlgorithms to know which states to compute values and policies for.
    # This function sets |self.states| to be the set of all states.
    def computeStates(self):
        self.states = set()
        queue = []
        self.states.add(self.startState())
        queue.append(self.startState())
        while len(queue) > 0:
            state = queue.pop()
            for action in self.actions(state):
                for newState, prob, reward in self.succAndProbReward(state, action):
                    if newState not in self.states:
                        self.states.add(newState)
                        queue.append(newState)
        # print "%d states" % len(self.states)
        # print self.states

############################################################

# An algorithm that solves an MDP (i.e., computes the optimal
# policy).
class MDPAlgorithm:
    # Set:
    # - self.pi: optimal policy (mapping from state to action)
    # - self.V: values (mapping from state to best values)
    def solve(self, mdp): raise NotImplementedError("Override me")

############################################################

# Abstract class: an RLAlgorithm performs reinforcement learning.  All it needs
# to know is the set of available actions to take.  The simulator (see
# simulate()) will call getAction() to get an action, perform the action, and
# then provide feedback (via incorporateFeedback()) to the RL algorithm, so it can adjust
# its parameters.
class RLAlgorithm:
    # Your algorithm will be asked to produce an action given a state.
    def getAction(self, state): raise NotImplementedError("Override me")

    # We will call this function when simulating an MDP, and you should update
    # parameters.
    # If |state| is a terminal state, this function will be called with (s, a,
    # 0, None). When this function is called, it indicates that taking action
    # |action| in state |state| resulted in reward |reward| and a transition to state
    # |newState|.
    def incorporateFeedback(self, state, action, reward, newState): raise NotImplementedError("Override me")

############################################################

# Performs Q-learning.  Read util.RLAlgorithm for more information.
# actions: a function that takes a state and returns a list of actions.
# discount: a number between 0 and 1, which determines the discount factor
# featureExtractor: a function that takes a state and action and returns a list of (feature name, feature value) pairs.
# explorationProb: the epsilon value indicating how frequently the policy
# returns a random action
class QLearningAlgorithm(RLAlgorithm):
    def __init__(self, actions, discount, featureExtractor, explorationProb=0.2):
        self.actions = actions
        self.discount = discount
        self.featureExtractor = featureExtractor
        self.explorationProb = explorationProb
        self.weights = collections.Counter()
        self.numIters = 0

    # Return the Q function associated with the weights and features
    def getQ(self, state, action):
        score = 0
        for f, v in self.featureExtractor(state, action):
            score += self.weights[f] * v
        return score

    # This algorithm will produce an action given a state.
    # Here we use the epsilon-greedy algorithm: with probability
    # |explorationProb|, take a random action.
    def getAction(self, state):
        self.numIters += 1
        if random.random() < self.explorationProb:
            return random.choice(self.actions(state))
        else:
            return max((self.getQ(state, action), action) for action in self.actions(state))[1]

    # Call this function to get the step size to update the weights.
    def getStepSize(self):
        # (TODO:) figure out how best to tackle this
        return 1.0 / math.sqrt(self.numIters)
        # return 1.0

    # We will call this function with (s, a, r, s'), which you should use to update |weights|.
    # Note that if s is a terminal state, then s' will be None.  Remember to check for this.
    # You should update the weights using self.getStepSize(); use
    # self.getQ() to compute the current estimate of the parameters.
    def incorporateFeedback(self, state, action, reward, newState):
        def getVopt(self, newState):
            max_q = 0
            for a in self.actions(newState):
                max_q = max(max_q, self.getQ(newState, a))
            return max_q

        def multFeatures(scalar, feature_list):
            retval = collections.Counter()
            for f, v in feature_list:
                retval[f] = v*scalar
            return retval

        Vopt = 0
        if newState:
            Vopt = getVopt(self, newState)

        prediction = self.getQ(state, action)
        target  = reward + self.discount * Vopt 
        self.weights -=  multFeatures(self.getStepSize() * (prediction - target), self.featureExtractor(state, action))
        
        return self.weights

class DriverLearner():
    def __init__(self):
        self.prevCarState = None
        self.prevAction = None
        self.discount = 0.99
        self.rl = QLearningAlgorithm(self.getActions, self.discount, self.carFeatureExtractor, 
                                    explorationProb=0.25)
        self.cur_weights = None
        self.cur_reward = 0

        time = str(datetime.now())
        self.logFile = open('logs/log_%s.log' % time, 'a')
        self.weightsFile = open('logs/weights/weights_%s.log' % time, 'a')
        self.rewardsFile = open('logs/rewards/rewards_%s.log' % time, 'a')

    def log(self, string):
        self.logFile.write(string)

    def carFeatureExtractor(self, carState, action):
        retval = []
        retval.append((('trackPos_feature', action), carState.trackPos))
        retval.append((('angle_feature', action), carState.angle, ))
        # retval.append((('distFromStart_feature', action), carState.distFromStart))
        # (TODO:) decide how to include damage
        # retval.append((('damage_feature', action), carState.damage))
        # (TODO:) try out adding in wheel velocities include damage
        # for i, v in enumerate(carState.wheelSpinVel):
        #     retval.append((('wheelSpinVel%d_feature' % i, action), v))

        speed_in_track_dir = carState.getSpeedX() * math.cos(carState.angle)
        retval.append((('speed_in_track_dir_feature', action), speed_in_track_dir))
        # speed forward
        retval.append((('speedX_feature', action), carState.speedX))
        # speed sideways
        retval.append((('speedY_feature', action), carState.speedY))
        
        return retval


    def getReward(self, prevCarState, curCarState):
        reward = 0

        # angle of car w.r.t track, the smaller the better
        reward += -1000 * abs(curCarState.angle)

        # speed in direction of race track
        speed_in_track_dir = curCarState.getSpeedX() * math.cos(curCarState.angle)
        reward += 100 * speed_in_track_dir

        # position on track
        # off the track, penalize heavily
        if (abs(curCarState.getTrackPos()) > 1):
            reward -= 100000 * (abs(curCarState.getTrackPos()) - 1)
        # on track, minimize the distance from the center
        # (TODO:) change it so that you are rewarded the closer to the oracle position you are
        else:  
            reward += -1000 * abs(curCarState.getTrackPos())

        # (TODO:) add change in damage. Need to see if damage kills the car so it can't race anymore
        # (TODO:) add distance from the start line
        
        # lap time
        if prevCarState:
            crossed_finish = curCarState.getCurLapTime() < prevCarState.getCurLapTime()
            if crossed_finish:
                self.log("CROSSED FINISH!!!!!")
                # give high reward for crossing the line at all
                reward += 100000
                # give higher reward for smaller lap time
                reward += -10 * curCarState.getLastLapTime()
            # not completed a lap, constant penalization
            else:
                reward -= 10
        
        return reward

    def getActions(self, curCarState):
        # steering wheel actions
        steering = arange(-1, 1, 0.1)

        # accelerator actions
        accel = arange(0, 1, 0.1)

        # (TODO:) brake actions

        # put them together
        return [(s, a) for s in steering for a in accel]        

    def logWeights(self):
        self.weightsFile.write("%s \n" % self.cur_weights)

    def logRewards(self):
        self.rewardsFile.write("%f\n" % self.cur_reward)

    def learnAndGetNextAction(self, curCarState):
        reset = False

        # only after the first iteration
        if self.prevCarState:
            reward = self.getReward(self.prevCarState, curCarState)
            
            # debugging
            self.cur_reward = reward
            self.cur_weights = str(self.rl.incorporateFeedback(self.prevCarState, 
                                               self.prevAction, 
                                               reward, 
                                               curCarState))
            
            # IF CRASHED, RESTART THE RACE
            if abs(curCarState.trackPos) > 1 and \
                math.sqrt(curCarState.speedX**2 + curCarState.speedY**2 + curCarState.speedZ**2) < 0.1:
                reset = True 
        
        self.prevCarState = curCarState

        self.logRewards()

        # Calculate the next action
        steering, accel = self.rl.getAction(curCarState)
        self.prevAction =  (steering, accel, reset)
        return self.prevAction

    def cleanup(self):
        self.weightsFile.close()
        self.rewardsFile.close()
        self.logFile.close()
