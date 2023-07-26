from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import tensorflow as tf

from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM
from keras.optimizers import RMSprop
from keras.callbacks import LambdaCallback, ModelCheckpoint, ReduceLROnPlateau
import random
import sys

import warnings

warnings.simplefilter("ignore", RuntimeWarning)

class TextDataGenerator(Sequence):
    def __init__(self, text, vocabulary, char_to_indices, indices_to_char, max_length, batch_size):
        self.text = text
        self.vocabulary = vocabulary
        self.char_to_indices = char_to_indices
        self.indices_to_char = indices_to_char
        self.max_length = max_length
        self.batch_size = batch_size
        self.steps = (len(text) - max_length) // batch_size
        
    def __len__(self):
        return self.steps
    
    def __getitem__(self, idx):
        batch_start = idx * self.batch_size
        batch_end = (idx + 1) * self.batch_size
        batches = self.text[batch_start:batch_end]
        X = np.zeros((self.batch_size, self.max_length, len(self.vocabulary)), dtype=bool)
        y = np.zeros((self.batch_size, len(self.vocabulary)), dtype=bool)
        for i, batch in enumerate(batches):
            for t, char in enumerate(batch[:-1]):
                X[i, t, self.char_to_indices[char]] = 1
            y[i, self.char_to_indices[batch[-1]]] = 1
        return X, y
    
    def on_epoch_end(self):
        random.shuffle(self.text)


with open('/kaggle/input/crptic-python/cptPY.txt', 'r') as file:
    text = file.read()


vocabulary = sorted(list(set(text)))

char_to_indices = dict((c, i) for i, c in enumerate(vocabulary))
indices_to_char = dict((i, c) for i, c in enumerate(vocabulary))

print(vocabulary)

# Dividing the text into subsequences of length max_length
# So that at each time step the next max_length characters
# are fed into the network
max_length = 100
steps = 4
sentences = []
next_chars = []
for i in range(0, len(text) - max_length, steps):
    sentences.append(text[i: i + max_length])
    next_chars.append(text[i + max_length])

# Building the LSTM network for the task
model = Sequential()
model.add(LSTM(128, input_shape=(max_length, len(vocabulary))))
model.add(Dense(len(vocabulary)))
model.add(Activation('softmax'))
optimizer = RMSprop(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)
model.summary()

# Load weights if available
try:
    model.load_weights('/kaggle/working/weights.hdf5')
    print("Weights loaded")
except:
    print("Could not load weights or training has not been started")

# Helper function to sample an index from a probability array
def sample_index(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

# Defining a helper function to save the model after each epoch
# in which the loss decreases
filepath = "/kaggle/working/weights.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss',
                             verbose=1, save_best_only=True,
                             mode='min')

# Defining a helper function to reduce the learning rate each time
# the learning plateaus
reduce_alpha = ReduceLROnPlateau(monitor='loss', factor=0.2,
                                 patience=1, min_lr=0.001)


def generate_text(length, diversity, user_input):
    # Get random starting text
    if len(user_input) < max_length:
        print("Start index is too low!")
        start_index = 0
    else:
        start_index = random.randint(0, len(user_input) - max_length - 1)

    generated = ''
    sentence = user_input[start_index: start_index + max_length]
    print(sentence)
    for i in range(length):
        x_pred = np.zeros((1, max_length, len(vocabulary)))
        for t, char in enumerate(sentence):
            x_pred[0, t, char_to_indices[char]] = 1.

        preds = model.predict(x_pred, verbose=0)[0]
        next_index = sample_index(preds, diversity)
        next_char = indices_to_char[next_index]

        generated += next_char
        sentence = sentence[1:] + next_char
    return generated

generate = '''
 if not (steps % self.save_model_every):
            state_dict = self.vae.state_dict()
            model_path = str(self.results_folder / f'vae.{steps}.pt')
            torch.save(state_dict, model_path)

            ema_state_dict = self.ema_vae.state_dict()
            model_path = str(self.results_folder / f'vae.{steps}.ema.pt')
            torch.save(ema_state_dict, model_path)

            print(f'{steps}: saving model to {str(self.results_folder)}')

'''


print(generate_text(500, 0.3, generate))
