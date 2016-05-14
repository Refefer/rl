from __future__ import print_function

import random
import json
import os
import sys

import numpy as np

from keras.models import Sequential
from keras.preprocessing import sequence
from keras.layers import *
from keras.utils.np_utils import to_categorical

def make_net(max_features, max_len, n_classes):

    # Character level RNN
    rnn = Sequential()
    rnn.add(Embedding(
        input_dim=max_features, 
        output_dim=20, 
        input_length=max_len, 
        dropout=0.2
    ))
    rnn.add(recurrent.GRU(
        output_dim=80,
        return_sequences=False,
        init='glorot_uniform', 
        inner_init='orthogonal', 
        activation='tanh', 
        inner_activation='hard_sigmoid', 
        W_regularizer=None, 
        U_regularizer=None, 
        b_regularizer=None, 
        dropout_W=0.1, 
        dropout_U=0.1
    ))
    rnn.add(Dense(
        output_dim=100
    ))

    # Positional RNN
    pnn = Sequential()
    pnn.add(Embedding(
        input_dim=2, 
        output_dim=1, 
        input_length=max_len, 
        dropout=0.0
    ))
    pnn.add(recurrent.GRU(
        output_dim=10,
        return_sequences=False,
        init='glorot_uniform', 
        inner_init='orthogonal', 
        activation='tanh', 
        inner_activation='hard_sigmoid', 
        W_regularizer=None, 
        U_regularizer=None, 
        b_regularizer=None, 
        dropout_W=0.1, 
        dropout_U=0.1
    ))
    pnn.add(Dense(
        output_dim=50
    ))

    model = Sequential()
    model.add(Merge([rnn, pnn], mode='concat'))
    model.add(Dense(
        output_dim=80,
    ))
    model.add(Activation('relu'))
    model.add(Dense(
        output_dim=80,
    ))
    model.add(Activation('relu'))
    model.add(Dense(
        output_dim=n_classes,
    ))

    return model

class SpellingDQN(object):

    def __init__(self, n_actions, gamma, max_len=64, max_exp=30000, batch_size=32, bpl=2, seed=2016):
        self.n_actions = n_actions
        self.gamma = gamma
        self.max_len = max_len
        self.max_exp = max_exp
        self.experience = []
        self.keys = []
        self.seen_states = set()
        self.model = make_net(256, max_len, n_actions)
        self.batch_size = batch_size
        self.bpl = bpl
        self.rs = np.random.RandomState(seed=seed)
        self.model.compile(
            loss='mse',
            optimizer='adam'
        )

    def state_to_rnn(self, state):
        cp, world = state

        cnn = [0] * len(world)
        cnn[cp] = 1

        cnn = sequence.pad_sequences([cnn], maxlen=self.max_len)

        world = [ord(c) for c in world]
        world = sequence.pad_sequences([world], maxlen=self.max_len)

        return [
            world.reshape((1, -1)), 
            cnn
        ]

    def predict(self, state):
        inp = self.state_to_rnn(state)
        actions = self.model.predict(inp, batch_size=1)
        return actions[0]
    
    def add(self, old_state, action, reward, new_state):
        key = (old_state, action, reward, new_state)
        if key in self.seen_states:
            return

        payload = (
            self.state_to_rnn(old_state), 
            action, 
            reward, 
            new_state if new_state is None else self.state_to_rnn(new_state)
        )

        if len(self.experience) > self.max_exp:
            idx = random.randint(0, self.max_exp - 1)
            self.seen_states.remove(self.keys[idx])
            self.experience[idx] = payload
            self.seen_states.add(key)
        else:
            self.experience.append(payload)
            self.keys.append(key)

    def add_history(self, history):
        old_state = None # Terminal
        for state, action, reward in reversed(history):
            self.add(state, action, reward, old_state)
            old_state = state

    def eval_states(self, states):
        Xw, Xc = [], []
        for w, c in states:
            Xw.append(w)
            Xc.append(c)

        return self.model.predict([np.vstack(Xw), np.vstack(Xc)])

    def learn(self):
        # sample from experience

        for _ in xrange(self.bpl):
            #
            # Build batch
            #
            idxs = self.rs.randint(0, len(self.experience), (self.batch_size,))
            new_states, old_states, new_idx = [], [], []
            i = 0
            for idx in idxs:
                os, a, r, ns = self.experience[idx]
                old_states.append(os)
                if ns is not None:
                    new_states.append(ns)
                    new_idx.append(i)
                    i += 1
                else:
                    new_idx.append(None)

                new_states.append(os)

            old_values = self.eval_states(old_states)
            new_values = self.eval_states(new_states)

            X_chars, X_pos, y_train = [], [], []
            for i, idx in enumerate(idxs):
                os, a, r, ns = self.experience[idx]
                values = old_values[i]
                nv_idx = new_idx[i]
                if nv_idx is None:
                    value = r
                else:
                    max_reward = new_values[nv_idx].max()
                    value = r + self.gamma * max_reward

                values[a] = min(max(value, -2), 2)
                X_chars.append(os[0])
                X_pos.append(os[1])
                y_train.append(values)

            # Featurize
            X_chars = np.vstack(X_chars)
            X_pos = np.vstack(X_pos)
            y_train = np.vstack(y_train)

            # And train
            self.model.train_on_batch([X_chars, X_pos], y_train)
            print(self.model.evaluate([X_chars, X_pos], y_train, batch_size=self.batch_size, verbose=0))


