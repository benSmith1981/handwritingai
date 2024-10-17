import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Reshape, LSTM, Dense, Lambda
from tensorflow.keras.models import Model
import numpy as np

# Define the CTC Loss function
def ctc_loss(args):
    y_true, y_pred, input_length, label_length = args
    return tf.keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)

# Input shape for the images (height, width, channels)
input_shape = (32, 128, 1)

# Build the model
input_data = Input(name='input_data', shape=input_shape)
x = Conv2D(32, (3, 3), padding='same', activation='relu')(input_data)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)

# Reshape for LSTM input
x = Reshape(target_shape=(-1, 64))(x)

# Add RNN layers (LSTM)
x = LSTM(128, return_sequences=True)(x)
x = Dense(64, activation='relu')(x)

# Output layer (softmax for character classification)
num_classes = 31  # Assuming 30 unique characters + 1 for the blank label
# Ensure y_pred has the correct shape and size
y_pred = Dense(num_classes, activation='softmax', name='y_pred')(x)

# Define inputs for CTC Loss
labels = Input(name='labels', shape=[None], dtype='float32')
input_length = Input(name='input_length', shape=[1], dtype='int64')
label_length = Input(name='label_length', shape=[1], dtype='int64')

# Apply CTC Loss via Lambda layer
loss_out = Lambda(ctc_loss, output_shape=(1,), name='ctc_loss')([labels, y_pred, input_length, label_length])

# Define the full model
model = Model(inputs=[input_data, labels, input_length, label_length], outputs=loss_out)

# Compile the model with CTC loss
model.compile(optimizer='adam', loss=lambda y_true, y_pred: y_pred)

# Generate dummy data (replace with actual data loading)
X_train = np.random.random((1000, 32, 128, 1))  # Dummy training images
y_train = np.random.randint(0, 30, (1000, 20))  # Dummy training labels (using integers for characters)
input_lengths = np.ones((1000, 1)) * 32  # Example input lengths
label_lengths = np.ones((1000, 1)) * 20  # Example label lengths

# Train the model
model.fit(x=[X_train, y_train, input_lengths, label_lengths], y=np.zeros(1000), batch_size=32, epochs=10)


# Save the model using the Keras native format
model.save('handwriting_model.keras')

# Or alternatively, save the model in HDF5 format
model.save('handwriting_model.h5')

