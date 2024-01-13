import tensorflow as tf
import numpy as np
import os
import time

# Load and preprocess the text data
path_to_file = tf.keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')
text = open(path_to_file, 'rb').read().decode(encoding='utf-8')

# Unique characters in the text
vocab = sorted(set(text))

# Creating a mapping from unique characters to indices
char2idx = {u: i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)

# Convert the text to numerical representation
text_as_int = np.array([char2idx[c] for c in text])

# Create training examples and targets
seq_length = 100
examples_per_epoch = len(text)//(seq_length+1)

char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)

sequences = char_dataset.batch(seq_length+1, drop_remainder=True)

def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text

dataset = sequences.map(split_input_target)

# Create training batches
BATCH_SIZE = 64
BUFFER_SIZE = 10000

dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

# Build the model
vocab_size = len(vocab)
embedding_dim = 256
rnn_units = 1024

def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim,
                                  batch_input_shape=[batch_size, None]),
        tf.keras.layers.LSTM(rnn_units,
                             return_sequences=True,
                             stateful=True,
                             recurrent_initializer='glorot_uniform'),
        tf.keras.layers.Dense(vocab_size)
    ])
    return model

model = build_model(
    vocab_size=len(vocab),
    embedding_dim=embedding_dim,
    rnn_units=rnn_units,
    batch_size=BATCH_SIZE)

# Compile the model
def loss(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

model.compile(optimizer='adam', loss=loss)

# Train the model
EPOCHS = 10

for epoch in range(EPOCHS):
    start = time.time()

    # Train the model
EPOCHS = 10

for epoch in range(EPOCHS):
    start = time.time()

    # Reset the hidden state at the start of every epoch
    model.reset_states()

    for (batch_n, (inp, target)) in enumerate(dataset):
        with tf.GradientTape() as tape:
            predictions = model(inp)
            loss_value = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(target, predictions, from_logits=True))

        grads = tape.gradient(loss_value, model.trainable_variables)
        model.optimizer.apply_gradients(zip(grads, model.trainable_variables))

        if batch_n % 100 == 0:
            print(f'Epoch {epoch+1} Batch {batch_n} Loss {loss_value.numpy()}')

    print(f'Epoch {epoch+1} Loss {loss_value.numpy()}')
    print(f'Time taken for 1 epoch {time.time()-start} sec\n')


# Generate text
def generate_text(model, start_string):
    num_generate = 1000
    input_eval = [char2idx[s] for s in start_string]
    input_eval = tf.expand_dims(input_eval, 0)
    text_generated = []

    temperature = 1.0

    model.reset_states()
    for i in range(num_generate):
        predictions = model(input_eval)
        predictions = tf.squeeze(predictions, 0)
        predictions = predictions / temperature
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1, 0].numpy()

        input_eval = tf.expand_dims([predicted_id], 0)

        text_generated.append(idx2char[predicted_id])

    return (start_string + ''.join(text_generated))

# Test the text generation
generated_text = generate_text(model, start_string="ROMEO: ")
print(generated_text)
