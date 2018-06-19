# Fashion-MNIST with tf.Keras
# Demonstration of TensorBoard and Debugger GUI plugin
# By Margaret Maynard-Reid, 5/28/2018

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Enable eager execution
tf.enable_eager_execution

# Load the fashion-mnist pre-shuffled train data and test data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

print("x_train shape:", x_train.shape, "y_train shape:", y_train.shape)

# Visualize the data


# Print training set shape - note there are 60,000 training data of image size of 28x28, 60,000 train labels)
print("x_train shape:", x_train.shape, "y_train shape:", y_train.shape)

# Print the number of training and test datasets
print(x_train.shape[0], 'train set')
print(x_test.shape[0], 'test set')

# Define the text labels
fashion_mnist_labels = ["T-shirt/top",  # index 0
                        "Trouser",      # index 1
                        "Pullover",     # index 2 
                        "Dress",        # index 3 
                        "Coat",         # index 4
                        "Sandal",       # index 5
                        "Shirt",        # index 6 
                        "Sneaker",      # index 7 
                        "Bag",          # index 8 
                        "Ankle boot"]   # index 9

# Image index, you can pick any number between 0 and 59,999
img_index = 5
# y_train contains the lables, ranging from 0 to 9
label_index = y_train[img_index]
# Print the label, for example 2 Pullover
print ("y = " + str(label_index) + " " +(fashion_mnist_labels[label_index]))
# Show one of the images from the training dataset
plt.imshow(x_train[img_index])


# Data normalization
# Normalize the data dimensions so that they are of approximately the same scale.

x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

print("Number of train data - " + str(len(x_train)))
print("Number of test data - " + str(len(x_test)))


# ## Split the data into train/validation/test data sets
# 
# 
# *   Training data - used for training the model
# *   Validation data - used for tuning the hyperparameters and evaluate the models
# *   Test data - used to test the model after the model has gone through initial vetting by the validation set.
# 
#

# Further break training data into train / validation sets (# put 5000 into validation set and keep remaining 55,000 for train)
(x_train, x_valid) = x_train[5000:], x_train[:5000] 
(y_train, y_valid) = y_train[5000:], y_train[:5000]

# Reshape input data from (28, 28) to (28, 28, 1)
w, h = 28, 28


x_train = x_train.reshape(x_train.shape[0], w, h, 1)
x_valid = x_valid.reshape(x_valid.shape[0], w, h, 1)
x_test = x_test.reshape(x_test.shape[0], w, h, 1)

# One-hot encode the labels
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_valid = tf.keras.utils.to_categorical(y_valid, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# Print training set shape
print("x_train shape:", x_train.shape, "y_train shape:", y_train.shape)

# Print the number of training, validation, and test datasets
print(x_train.shape[0], 'train set')
print(x_valid.shape[0], 'validation set')
print(x_test.shape[0], 'test set')


# ## Create the model architecture
# 
# There are two APIs for defining a model in Keras:
# 1. [Sequential model API](https://keras.io/models/sequential/)
# 2. [Functional API](https://keras.io/models/model/)
# 
# In this notebook we are using the Sequential model API. 
# If you are interested in a tutorial using the Functional API, checkout Sara Robinson's blog [Predicting the price of wine with the Keras Functional API and TensorFlow](https://medium.com/tensorflow/predicting-the-price-of-wine-with-the-keras-functional-api-and-tensorflow-a95d1c2c1b03).
# 
# In defining the model we will be using some of these Keras APIs:
# *   Conv2D() [link text](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv2D/) - create a convolutional layer 
# *   Pooling() [link text](https://keras.io/layers/pooling/) - create a pooling layer 
# *   Dropout() [link text](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dropout) - apply drop out 


model = tf.keras.Sequential()

# Must define the input shape in the first layer of the neural network
model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=2, padding='same', activation='relu', input_shape=(28,28,1), name = 'conv1'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=2, name='maxpool1'))
model.add(tf.keras.layers.Dropout(0.3, name='dropout1'))

model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=2, padding='same', activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
model.add(tf.keras.layers.Dropout(0.3))

model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(256, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(10, activation='softmax'))

# Take a look at the model summary
model.summary()


# ## Compile the model
# Configure the learning process with compile() API before training the model. It receives three arguments:
# 
# *   An optimizer 
# *   A loss function 
# *   A list of metrics 
#

model.compile(loss='categorical_crossentropy',
             optimizer='adam',
             metrics=['accuracy'])


# ## Train the model
# 
# Now let's train the model with fit() API.
# 
# We use the [ModelCheckpoint](https://keras.io/callbacks/#modelcheckpoint) API to save the model after every epoch. Set "save_best_only = True" to save only when the validation accuracy improves.

# Uncomment for TensorBoard Debugger GUI Plugin
# from tensorflow.python import debug as tf_debug

model.load_weights('model.weights.best.hdf5')

log_path = "logs/"
checkpointer = tf.keras.callbacks.ModelCheckpoint(filepath='model.weights.best.hdf5', verbose = 1, save_best_only=True)
tflogger = tf.keras.callbacks.TensorBoard(log_dir=log_path,
                                          write_graph=True,
                                          write_images=True,
                                          write_grads=True)

with tf.Session() as sess:
    # Enable TensorBoard debugger GUI plugin
    sess.run(tf.global_variables_initializer())

    # Uncomment for TensorBoard Debugger GUI Plugin
    # sess = tf_debug.TensorBoardDebugWrapperSession(sess, "localhost:7000")

    model.fit(x_train,
             y_train,
            batch_size=64,
             epochs=2,
             validation_data=(x_valid, y_valid),
             callbacks=[checkpointer,tflogger])

# Load Model with the best validation accuracy

# Load the weights with the best validation accuracy
model.load_weights('model.weights.best.hdf5')


# ## Test Accuracy

# In[15]:


# Evaluate the model on test set
score = model.evaluate(x_test, y_test, verbose=0)

# Print test accuracy
print('\n', 'Test accuracy:', score[1])


# ## Visualize prediction
# Now let's visualize the prediction using the model you just trained. 
# First we get the predictions with the model from the test data.
# Then we print out 15 images from the test data set, and set the titles with the prediction (and the groud truth label).
# If the prediction matches the true label, the title will be green; otherwise it's displayed in red.

# In[16]:


y_hat = model.predict(x_test)

# Plot a random sample of 10 test images, their predicted labels and ground truth
figure = plt.figure(figsize=(20, 8))
for i, index in enumerate(np.random.choice(x_test.shape[0], size=15, replace=False)):
    ax = figure.add_subplot(3, 5, i + 1, xticks=[], yticks=[])
    # Display each image
    ax.imshow(np.squeeze(x_test[index]))
    predict_index = np.argmax(y_hat[index])
    true_index = np.argmax(y_test[index])
    # Set the title for each image
    ax.set_title("{} ({})".format(fashion_mnist_labels[predict_index], 
                                  fashion_mnist_labels[true_index]),
                                  color=("green" if predict_index == true_index else "red"))

