from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Activation
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import LambdaCallback, ModelCheckpoint, ReduceLROnPlateau
import random
import sys

def text_generator(text, max_length, steps):
    # Dividing the text into subsequences of length max_length
    # So that at each time step the next max_length characters
    # are fed into the network
    sentences = []
    next_chars = []
    for i in range(0, len(text) - max_length, steps):
        sentences.append(text[i: i + max_length])
        next_chars.append(text[i + max_length])

    while True:
        for i in range(0, len(sentences), batch_size):
            batch_sentences = sentences[i:i+batch_size]
            batch_next_chars = next_chars[i:i+batch_size]
            X = np.zeros((len(batch_sentences), max_length), dtype=np.uint8)
            y = np.zeros((len(batch_sentences),), dtype=np.uint8)
            for j, sentence in enumerate(batch_sentences):
                for t, char in enumerate(sentence):
                    X[j, t] = char_to_indices[char]
                y[j] = char_to_indices[batch_next_chars[j]]
            yield X[:, :, np.newaxis], y

# Set the seed for reproducibility
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

with open('/kaggle/input/crptic-python/python.txt', 'r') as file:
    text = file.read()[:8000000]

vocabulary = sorted(list(set(text)))
char_to_indices = dict((c, i) for i, c in enumerate(vocabulary))
indices_to_char = dict((i, c) for i, c in enumerate(vocabulary))

# Dividing the text into subsequences of length max_length
# So that at each time step the next max_length characters
# are fed into the network
max_length = 100
steps = 3
batch_size = 128

# Determine the number of unique characters in the vocabulary
vocab_size = len(vocabulary)

# Building the LSTM network for the task
model = Sequential()
model.add(LSTM(128, input_shape=(max_length, 1)))
model.add(Dense(vocab_size))
model.add(Activation('softmax'))
optimizer = RMSprop(lr=0.01)
model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer)


# Helper function to sample an index from a probability array
def sample_index(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


# Helper function to generate text after the end of each epoch
def on_epoch_end(epoch, logs):
    if epoch % 2 == 0:
        print()
        print('----- Generating text after Epoch: % d' % epoch)

        start_index = random.randint(0, len(text) - max_length - 1)
        for diversity in [0.1, 0.3, 0.5, 1.0]:
            print('----- diversity:', diversity)

            generated = ''
            sentence = text[start_index: start_index + max_length]
            generated += sentence
            print('----- Generating with seed: "' + sentence + '"')
            sys.stdout.write(generated)

            for i in range(400):
                x_pred = np.zeros((1, max_length, 1))
                for t, char in enumerate(sentence):
                    x_pred[0, t, 0] = char_to_indices[char]

                preds = model.predict(x_pred, verbose=0)[0]
                next_index = sample_index(preds, diversity)
                next_char = indices_to_char[next_index]

                generated += next_char
                sentence = sentence[1:] + next_char

                sys.stdout.write(next_char)
                sys.stdout.flush()
            print()


print_callback = LambdaCallback(on_epoch_end=on_epoch_end)

# Defining a helper function to save the model after each epoch
# in which the loss decreases
filepath = "weights.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss',
                             verbose=1, save_best_only=True,
                             mode='min')

# Defining a helper function to reduce the learning rate each time
# the learning plateaus
reduce_alpha = ReduceLROnPlateau(monitor='loss', factor=0.2,
                                 patience=1, min_lr=0.001)
callbacks = [print_callback, checkpoint, reduce_alpha]

# Training the LSTM model
train_generator = text_generator(text, max_length, steps)
steps_per_epoch = (len(text) - max_length) // steps // batch_size
model.fit(train_generator, steps_per_epoch=steps_per_epoch, epochs=28, callbacks=callbacks)

def generate_text(length, diversity):
    # Get random starting text
    start_index = random.randint(0, len(text) - max_length - 1)
    generated = ''
    sentence = text[start_index: start_index + max_length]
    generated += sentence
    for i in range(length):
        x_pred = np.zeros((1, max_length, 1))
        for t, char in enumerate(sentence):
            x_pred[0, t, 0] = char_to_indices[char]

        preds = model.predict(x_pred, verbose=0)[0]
        next_index = sample_index(preds, diversity)
        next_char = indices_to_char[next_index]

        generated += next_char
        sentence = sentence[1:] + next_char
    return generated


print(generate_text(500, 0.5))
