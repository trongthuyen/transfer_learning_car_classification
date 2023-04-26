## basic libraries → data exploration
import pandas as pd
from glob import glob
import matplotlib.pyplot as plt
import os
import pickle

## transfer learning
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, BatchNormalization
from tensorflow.keras.applications.efficientnet import EfficientNetB1

train_path = '../dvm_datasets/train/'
test_path = '../dvm_datasets/validate/'

df_test_labels = pd.read_csv('../stanford_car_datasets/testing_labels.csv')
train_car = glob(f"{train_path}*/*")
test_car = glob(f"{test_path}*/*")

class_names = list(df_test_labels["Cars"].unique())

## setting up some parameters for data augmentation
img_width, img_height = 240, 240
num_channels = 3
train_samples = len(train_car)
test_samples = len(test_car)
num_of_class_names = len(class_names)
batch_size = 16
epochs = 50

## performing augmentation on the training data
train_datagen = ImageDataGenerator(
  zoom_range=0.2,
  rescale=1./255,
  rotation_range = 20,
  horizontal_flip=True
)

valid_datagen = ImageDataGenerator()

## converting data to a tf.data.Dataset object
train_generator = train_datagen.flow_from_directory(
    train_path,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

valid_generator = valid_datagen.flow_from_directory(
    test_path,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')


## load the Efficientnet B1 model
model = EfficientNetB1(
  include_top=False,
  weights='imagenet',
  input_shape=(img_width, img_height, 3)
)

## adding some extra layers
x = model.output
x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x)
x = Dense(
  units=1024,
  activation='relu',
  kernel_regularizer=regularizers.l2(0.01), 
  kernel_initializer='random_uniform',
  bias_initializer='zeros'
)(x)
x = Dense(
  units=1024,
  activation='relu',
  kernel_regularizer=regularizers.l2(0.01), 
  kernel_initializer='random_uniform',
  bias_initializer='zeros'
)(x)
output = Dense(num_of_class_names, activation='softmax', name='pred')(x)

## create the extended model
tf_model = Model(inputs=model.input, outputs=output)

## fix the feature extraction part of the model
for layer in model.layers:
    if isinstance(layer, BatchNormalization):
        layer.trainable = True
    else:
        layer.trainable = False

## compile the model, define optimizer and the loss function
opt = Adam(learning_rate=0.01)

tf_model.compile(
  loss='categorical_crossentropy',
  optimizer=opt, metrics=['accuracy']
)

model.summary()

checkpoint_path = "../model_efficientnetb1/cp_efficientnetb1_model.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights
cp_callback = ModelCheckpoint(
  filepath=checkpoint_path,
  save_weights_only=True,
  verbose=1
)

# tf_model.load_weights(checkpoint_path)

training_history = tf_model.fit(
  train_generator,
  steps_per_epoch=len(train_generator),
  validation_data=valid_generator,
  validation_steps=(len(valid_generator) / batch_size),
  epochs=epochs,
  callbacks=[cp_callback]
)

with open('./trainEfficientnetB1History', 'wb') as file_pi:
    pickle.dump(training_history.history, file_pi)

# with open('./trainEfficientnetB1History', "rb") as file_pi:
#     training_history = pickle.load(file_pi)

accuracy = training_history.history['accuracy']
val_accuracy = training_history.history['val_accuracy']

loss = training_history.history['loss']
val_loss = training_history.history['val_loss']

epochs_range = range(20)

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, accuracy, label='Training Accuracy')
plt.plot(epochs_range, val_accuracy, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()
