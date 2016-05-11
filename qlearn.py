from collections import namedtuple
import random
import string

import numpy as np

from Levenshtein import distance

WORLD = "namedtuple"

# Randomize start
#START = []
#for i in xrange(random.randint(len(WORLD)/2, len(WORLD) * 2)):
#    START.append(random.choice(list(string.ascii_lowercase) + [' ']))
#
#START = ''.join(START)

# Anagram
START = ''.join(sorted(list(WORLD)))

GAMMA = 1.0
EPSILON = .8
ITERS = 1000000
MAX_HISTORY = 500

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

class SpellingTransition(Transition):
    INSERTS = tuple(list(string.ascii_lowercase) + [' '])
    MOVEMENTS = tuple(['left', 'right', 'delete', 'submit'])
    ACTIONS = INSERTS + MOVEMENTS
    I_INDEX = {a: i for i, a in enumerate(INSERTS)}
    M_INDEX = {a: i for i, a in enumerate(MOVEMENTS)}
    A_INDEX = {a: i for i, a in enumerate(INSERTS + MOVEMENTS)}

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
            world = world[:cur_pos] + world[cur_pos + 1:]
        elif action != 'submit':
            world = world[:cur_pos] + action + world[cur_pos:]

        cur_pos = max(0, min(cur_pos, len(world)))
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
                reward = 2
            else:
                reward = -(nd ** 2)

        return reward - 0.1

EmptyS = namedtuple('EmptyS', 'cp,w')
FullS = namedtuple('FullS', 'cp,w,c')
SubmitS = namedtuple('SubmitS', 'w')

class AnagramTransition(Transition):
    MOVEMENTS = ('left', 'right', 'submit')
    POP = 'pop',
    PUSH = 'push',
    EMPTY = MOVEMENTS + PUSH
    FULL  = MOVEMENTS + POP

    def is_terminal(self, state):
        return isinstance(state, SubmitS)
        return state[0] == 'submit'

    def action_set(self, state):
        if isinstance(state, EmptyS):
            return self.EMPTY

        return self.FULL

    def index_to_action(self, state, index):
        if isinstance(state, EmptyS):
            return self.EMPTY[index]

        return self.FULL[index]

    def action_to_index(self, state, action):
        if isinstance(state, EmptyS):
            return self.EMPTY.index(action)
        elif isinstance(state, FullS):
            return self.FULL.index(action)

        return 0

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
                nw = state.w[:state.cp] + state.w[state.cp+1:]
                ncp = min(len(nw) - 1, state.cp)
                return FullS(ncp, nw, state.w[state.cp])

        elif isinstance(state, FullS):
            if action == 'left':
                return FullS(max(0, state.cp - 1), state.w, state.c)
            elif action == 'right':
                return FullS(min(len(state.w), state.cp + 1), state.w, state.c)
            else:
                # Pop
                nw = state.w[:state.cp] + state.c + state.w[state.cp:]
                return EmptyS(state.cp, nw)

class AnagramUtility(UtilityF):

    def _reward(self, old_state, action, new_state):
        prev_world, world = old_state.w, new_state.w
        reward = 0

        if action == 'submit':
            nd = distance(world, WORLD)
            if nd == 0:
                reward = 2
            else:
                reward = -(nd ** 2)

        elif prev_world != world:
            pd = distance(prev_world, WORLD)
            nd = distance(world, WORLD)
            if pd < nd:
                reward = -1
            elif nd < pd:
                reward = 1

        reward -= 0.1
        return reward

def main():
    Q = {}
    tf = SpellingTransition()
    uf = SCUtility(tf)
    qp = QPolicy(Q, tf)
    spos = 0 #random.randint(0, len(START)-1)
    for i in xrange(ITERS):
        #state = EmptyS(spos, START)
        state = (0, START)
        cur_eps = max(0.05, EPSILON - i / (10*float(ITERS)))
        #cur_eps = EPSILON
        if i and i % 1000 == 0:
            print 'Iteration', i
            print 'current epsilon', cur_eps
            print "States:", len(Q)
            print "State 0:", Q[state]

            if i % 1000 == 0:
                print "Best policy:"
                rem = 0
                cum_rew = 0
                while not tf.is_terminal(state):
                    if rem < MAX_HISTORY:
                        action, score = qp.best_action(state, score=True)
                    else:
                        action, score = 'submit', float('-inf')

                    print rem, state, action, score
                    cum_rew = uf.evaluate(state, action)
                    state = tf.evaluate(state, action)
                    rem += 1

                print state, action
                print "Best reward:", cum_rew

        history = []
        action = None
        # Until finished
        while not tf.is_terminal(state):
            _cur_eps = cur_eps + (len(history) / (float(MAX_HISTORY) * 2)) ** 2
            # Get the next state
            if len(history) == MAX_HISTORY:
                action = 'submit'

            elif random.random() < _cur_eps:
                # Explore!
                action = tf.random_action(state)
            else:
                action = qp.best_action(state)

            if i %1000 == 0:
                print state, action

            # Change the world
            new_state = tf.evaluate(state, action)

            reward = uf.evaluate(state, action)
                
            history.append((state, action, reward))
            state = new_state

        history.append((state, action, reward))

        # Alright, run through the background
        history = list(reversed(history))
        if i % 1000 == 0:
            print "next"

        for si, (state, action, reward) in enumerate(history):
            if si == 0:
                Q[state] = np.array([reward])

            if state not in Q:
                N = len(tf.action_set(state))
                Q[state] = np.zeros((N,), 'float32')

            if si == 0:
                v = GAMMA * reward
            else:
                v = reward + GAMMA * Q[history[si-1][0]].max()

            if i % 1000 == 0:
                print state, action, v

            a_idx = tf.action_to_index(state, action)
            Q[state][a_idx] = v

if __name__ == '__main__':
    main()
