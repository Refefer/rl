import pprint
from collections import namedtuple
import random
import string

import numpy as np
np.set_printoptions(precision=3)

from Levenshtein import distance

from dqn import SpellingDQN

WORLD = "randomize start"

# Randomize start
#START = []
#for i in xrange(random.randint(len(WORLD)/2, len(WORLD) * 2)):
#    START.append(random.choice(list(string.ascii_lowercase) + [' ']))
#
#START = ''.join(START)

# Anagram
START = ''.join(sorted(list(WORLD)))

GAMMA = .9
EPSILON = .5
MIN_EPSILON = .03
DECAY = 50
ITERS = 10000000
MAX_HISTORY = 100

class Transition(object):

    def action_set(self, state):
        raise NotImplementedError()

    def action_to_index(self, state, action):
        raise NotImplementedError()

    def index_to_action(self, state, action):
        raise NotImplementedError()

    def random_action(self, state):
        return random.choice(self.action_set(state))

    def evaluate(self, state, action):
        raise NotImplementedError()

    def is_terminal(self, state):
        raise NotImplementedError()
        
class UtilityF(object):
    def __init__(self, transition):
        self.transition = transition

    def evaluate(self, state, action):
        next_state = self.transition.evaluate(state, action)
        return self._reward(state, action, next_state)

    def _reward(self, old_state, new_state):
        raise NotImplementedError()

class QPolicy(object):
    def __init__(self, Q, tf):
        self.Q = Q
        self.tf = tf

    def best_action(self, state, score=False):
        # Exploit!
        action_state = self.Q.get(state)
        if action_state is None:
            # No knowledge of this state, so random action
            action = self.tf.random_action(state)
            if score:
                return action, 0

            return action

        r_idx = np.argmax(action_state)
        #wer = np.where(action_state == action_state[a_idx])
        #r_idx = random.choice(wer[0].tolist())
        action = self.tf.index_to_action(state, r_idx)
        if score:
            return action, action_state[r_idx]

        return action

class DQPolicy(object):
    def __init__(self, dqn, tf):
        self.dqn = dqn
        self.tf = tf

    def best_action(self, state, score=False):
        action_state = self.dqn.predict(state)
        r_idx = np.argmax(action_state)
        action = self.tf.index_to_action(state, r_idx)
        if score:
            return action, action_state[r_idx]

        return action

class SpellingTransition(Transition):
    INSERTS = tuple(list(string.ascii_lowercase) + [' '])
    MOVEMENTS = tuple(['left', 'right', 'delete', 'submit'])
    ACTIONS = INSERTS + MOVEMENTS
    A_INDEX = {a: i for i, a in enumerate(ACTIONS)}

    def __init__(self, max_len=64):
        self.max_len = max_len

    def is_terminal(self, state):
        return state[0] == 'submit'

    def action_set(self, state):
        return self.ACTIONS

    def index_to_action(self, state, index):
        return self.ACTIONS[index]

    def action_to_index(self, state, action):
        if state[0] == 'submit':
            return 0

        return self.A_INDEX[action]

    def evaluate(self, state, action):
        cur_pos, world = state
        prev_world = world
        if action == 'left':
            cur_pos -= 1
        elif action == 'right':
            cur_pos += 1
        elif action == 'delete':
            if  len(world) > 1:
                world = world[:cur_pos] + world[cur_pos + 1:]
        elif action != 'submit':
            if len(world) < self.max_len:
                world = world[:cur_pos] + action + world[cur_pos:]

        cur_pos = max(0, min(cur_pos, len(world) - 1))
        if action == 'submit':
            return ('submit', world)

        return (cur_pos, world)

class SCUtility(UtilityF):

    def _reward(self, old_state, action, new_state):
        prev_world, world = old_state[1], new_state[1]
        reward = 0

        if prev_world != world:
            pd = distance(prev_world, WORLD)
            nd = distance(world, WORLD)
            if pd < nd:
                reward = -1
            elif nd < pd:
                reward = 1

        elif action == 'submit':
            nd = distance(world, WORLD)
            if nd == 0:
                return 10
            else:
                reward = -(nd ** 2)

        return reward - 0.1

class AnagramTransition(Transition):
    MOVEMENTS = ('left', 'right', 'submit')
    POP = 'pop',
    PUSH = 'push',
    EMPTY = MOVEMENTS + PUSH
    NOT_EMPTY = MOVEMENTS + PUSH + POP

    def is_terminal(self, state):
        return isinstance(state, SubmitS)

    def action_set(self, state):
        return state.ACTIONS

    def index_to_action(self, state, index):
        return state.ACTIONS[index]

    def action_to_index(self, state, action):
        return state.A_IDX[action]

    def evaluate(self, state, action):
        if action == 'submit':
            return SubmitS(state.w)

        if isinstance(state, EmptyS):
            if action == 'left':
                return EmptyS(max(0, state.cp - 1), state.w)
            elif action == 'right':
                return EmptyS(min(len(state.w) - 1, state.cp + 1), state.w)
            else:
                # Push
                if not state.w:
                    # Noop
                    return state

                nw = state.w[:state.cp] + state.w[state.cp+1:]
                ncp = min(len(nw) - 1, state.cp)
                return NotEmptyS(ncp, nw, (None, state.w[state.cp]))

        elif isinstance(state, NotEmptyS):
            if action == 'left':
                return NotEmptyS(max(0, state.cp - 1), state.w, state.c)
            elif action == 'right':
                return NotEmptyS(min(len(state.w) - 1, state.cp + 1), state.w, state.c)
            elif action == 'push':
                if not state.w:
                    return state

                nw = state.w[:state.cp] + state.w[state.cp+1:]
                ncp = min(len(nw) - 1, state.cp)
                return NotEmptyS(ncp, nw, (state.c, state.w[state.cp]))
 
            else:
                # Pop
                stack, cur_c = state.c
                nw = state.w[:state.cp] + cur_c + state.w[state.cp:]
                if stack is None:
                    return EmptyS(state.cp, nw)

                return NotEmptyS(state.cp, nw, stack)

class EmptyS(namedtuple('EmptyS', 'cp,w')):
    ACTIONS = AnagramTransition.EMPTY
    A_IDX = {a: i for i, a in enumerate(ACTIONS)}

class NotEmptyS(namedtuple('NotEmptyS', 'cp,w,c')):
    ACTIONS = AnagramTransition.NOT_EMPTY
    A_IDX = {a: i for i, a in enumerate(ACTIONS)}

class SubmitS(namedtuple('SubmitS', 'w')):
    ACTIONS = 'submit',
    A_IDX = {'submit': 0} 

class AnagramUtility(UtilityF):

    def _reward(self, old_state, action, new_state):
        prev_world, world = old_state.w, new_state.w
        reward = 0

        if action == 'submit':
            nd = distance(world, WORLD)
            if nd == 0:
                reward = 10
            else:
                reward = -(nd ** 2)

        elif prev_world != world:
            pd = distance(prev_world, WORLD)
            nd = distance(world, WORLD)
            if pd < nd:
                reward = -1
            elif nd < pd:
                reward = 1

        return reward - 0.1

def main():
    #tf = AnagramTransition()
    #uf = AnagramUtility(tf)
    tf = SpellingTransition()
    uf = SCUtility(tf)
    #Q = {}
    #qp = QPolicy(Q, tf)
    Q = SpellingDQN(len(tf.ACTIONS), GAMMA)
    qp = DQPolicy(Q, tf)
    spos = 0 #random.randint(0, len(START)-1)
    best = float('-inf')
    for i in xrange(ITERS):
        #state = EmptyS(spos, START)
        state = (0, START)
        cur_eps = max(MIN_EPSILON, EPSILON - i*DECAY / (float(ITERS)))
        #cur_eps = EPSILON
        if i and i % 100 == 0:
            print 'Iteration', i
            print 'current epsilon', cur_eps
            #print "States:", len(Q)
            print "State 0:", Q.predict(state)

            rem = 0
            cum_rew = 0
            _state = state
            h = []
            while not tf.is_terminal(_state):
                if rem < MAX_HISTORY:
                    action, score = qp.best_action(_state, score=True)
                else:
                    action, score = 'submit', -100

                #print _state, action

                cum_rew += uf.evaluate(_state, action)

                h.append((rem, _state, action, score, cum_rew))
                print h[-1]
                _state = tf.evaluate(_state, action)
                rem += 1

            print "Current Reward:", cum_rew
            print "Current History:", len(h)

            if _state[1] == WORLD and cum_rew > best:
            #if _state.w == WORLD and cum_rew > best:
                for a,b,c,d,e in h:
                    print a, b, c, d, e

                raw_input()

            best = max(best, cum_rew)

        history = []
        action = None
        # Until finished
        while not tf.is_terminal(state):
            #_cur_eps = cur_eps + (len(history) / (float(MAX_HISTORY) * 2)) ** 2
            # Get the next state
            if len(history) == MAX_HISTORY:
                action = 'submit'

            elif random.random() < cur_eps:
                # Explore!
                action = tf.random_action(state)
            else:
                action = qp.best_action(state)

            #if i % 1000 == 0:
            #    print "Cur State:", state, action

            # Change the world
            new_state = tf.evaluate(state, action)

            reward = uf.evaluate(state, action)
                
            history.append((state, action, reward))
            state = new_state

        #history.append((state, action, reward))

        # Alright, run through the background
        #history = list(reversed(history))
        #for si, (state, action, reward) in enumerate(history):

        #    if state not in Q:
        #        N = len(tf.action_set(state))
        #        Q[state] = np.zeros((N,), 'float32')

        #    if si == 0:
        #        v = GAMMA * reward
        #    else:
        #        v = reward + GAMMA * Q[history[si-1][0]].max()

        #    a_idx = tf.action_to_index(state, action)
        #    Q[state][a_idx] = v

        state_prime = None
        for si, (state, action, reward) in enumerate(reversed(history)):
            a_idx = tf.action_to_index(state, action)
            if i % 100 == 0:
                print state, action, reward, state_prime 

            Q.add(state, a_idx, reward, state_prime)
            state_prime = state

        Q.learn()

if __name__ == '__main__':
    main()
