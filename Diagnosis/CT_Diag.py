#!/usr/bin/env python
# coding: utf-8

# # Covid diagnosis task using CT images

# In[ ]:


from google.colab import drive
drive.mount('/content/drive')


# In[ ]:


import tensorflow as tf
import os
import cv2
from tqdm.notebook import trange, tqdm
import numpy as np
import matplotlib.pyplot as plt
import copy
from tensorflow import keras
from tensorflow.keras import layers
import random
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.applications.vgg16 import VGG16
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import plot_model
from keras.utils.np_utils import to_categorical
import shutil
import copy


# ## Hyperparameters:

# In[ ]:


IMG_WIDTH = 256
IMG_HEIGHT = 256
main_path_1 = '/content/drive/MyDrive/Final Dataset/'
main_path_2 = '/content/drive/MyDrive/Final Dataset Diag/'


# ## Loading Trained Models:

# In[ ]:


ground_glass_model = tf.keras.models.load_model(main_path_2+'Trained Models/ground_glass_model.hdf5')
consolidation_model = tf.keras.models.load_model(main_path_2+'Trained Models/consolidation_model.hdf5')
# pleural_effusion_model = tf.keras.models.load_model(main_path_2+'Trained Models/pleural_effusion.hdf5')


# ## Reading Dataset Path:

# In[ ]:


def read_data_path(first_dir, second_dir):
  first_img_paths = sorted(
      [
          os.path.join(first_dir, fname)
          for fname in os.listdir(first_dir)
          if fname.endswith(".jpg") or fname.endswith(".png") or fname.endswith(".jpeg")
      ]
  )
  second_img_paths = sorted(
      [
          os.path.join(second_dir, fname)
          for fname in os.listdir(second_dir)
          if fname.endswith(".jpg") or fname.endswith(".png") or fname.endswith(".jpeg")
      ]
  )
  return first_img_paths, second_img_paths


# In[ ]:


Covid_dir = main_path_2 + 'CT_COVID'
NoCovid_dir = main_path_2 + 'CT_NonCOVID'

Covid_img_paths, NoCovid_img_paths = read_data_path(Covid_dir, NoCovid_dir)
print("Number of images with Covid: ", len(Covid_img_paths))
print("Number of images without Covid: ", len(NoCovid_img_paths))


# ## Reading Dataset:<br>
# Y:<br>
# 1 ==> Covid<br>
# 0 ==> NoCovid

# In[ ]:


def read_data(first_img_paths, second_img_paths):
  X = []
  Y = []

  for i in tqdm(range(len(first_img_paths))):
    img = cv2.imread(first_img_paths[i], cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    img = np.reshape(img, (256,256,1))
    img = img.astype("float32")
    img/=255.0
    X.append(img)
    Y.append(1)

  for i in tqdm(range(len(second_img_paths))):
    img = cv2.imread(second_img_paths[i], cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    img = np.reshape(img, (256,256,1))
    img = img.astype("float32")
    img/=255.0
    X.append(img)
    Y.append(0)
  
  X = np.array(X)
  Y = np.array(Y)

  shuffler = np.random.permutation(len(X))
  X = X[shuffler]
  Y = Y[shuffler]
  
  return X, Y

X, Y = read_data(Covid_img_paths, NoCovid_img_paths)


# ## Split Data:<br>
# train: 70%<br>
# val: 15%<br>
# test: 15%

# In[ ]:


Y_ = to_categorical(Y, num_classes = 2)

train_test_split = int(len(X)*0.85)
train_val_split = int(len(X)*0.7)

x_train = X[:train_val_split]
y_train = Y_[:train_val_split]

x_val = X[train_val_split:train_test_split]
y_val = Y_[train_val_split:train_test_split]

x_test = X[train_test_split:]
y_test = Y_[train_test_split:]


# In[ ]:


del X
del Y
del Y_


# ## Predicting with trained seg models:<br>
# Train:

# In[ ]:


ground_glass_pred = ground_glass_model.predict(x_train)
ground_glass_pred = (ground_glass_pred > ground_glass_pred.max()/2).astype(np.uint8)
consolidation_pred = consolidation_model.predict(x_train)
consolidation_pred = (consolidation_pred > consolidation_pred.max()/2).astype(np.uint8)
# pleural_effusion_pred = pleural_effusion_model.predict(x_train)
# pleural_effusion_pred = (pleural_effusion_pred > pleural_effusion_pred.max()/2).astype(np.uint8)

train_concat = np.concatenate([x_train, ground_glass_pred, consolidation_pred], axis=3)#, pleural_effusion_pred], axis=3)


# val:

# In[ ]:


ground_glass_pred = ground_glass_model.predict(x_val)
ground_glass_pred = (ground_glass_pred > ground_glass_pred.max()/2).astype(np.uint8)
consolidation_pred = consolidation_model.predict(x_val)
consolidation_pred = (consolidation_pred > consolidation_pred.max()/2).astype(np.uint8)
# pleural_effusion_pred = pleural_effusion_model.predict(x_val)
# pleural_effusion_pred = (pleural_effusion_pred > pleural_effusion_pred.max()/2).astype(np.uint8)

val_concat = np.concatenate([x_val, ground_glass_pred, consolidation_pred], axis=3)#, pleural_effusion_pred], axis=3)


# test:

# In[ ]:


ground_glass_pred = ground_glass_model.predict(x_test)
ground_glass_pred = (ground_glass_pred > ground_glass_pred.max()/2).astype(np.uint8)
consolidation_pred = consolidation_model.predict(x_test)
consolidation_pred = (consolidation_pred > consolidation_pred.max()/2).astype(np.uint8)
# pleural_effusion_pred = pleural_effusion_model.predict(x_test)
# pleural_effusion_pred = (pleural_effusion_pred > pleural_effusion_pred.max()/2).astype(np.uint8)

test_concat = np.concatenate([x_test, ground_glass_pred, consolidation_pred], axis=3)#, pleural_effusion_pred], axis=3)


# In[ ]:


del ground_glass_pred
del consolidation_pred
# del pleural_effusion_pred


# ## Visualize Prediction:

# In[ ]:


num_of_img = 6
plt.figure(figsize=(16,24))
for n ,i in enumerate(list(np.random.randint(0,len(train_concat),num_of_img))):
  plt.subplot(6,4,4*n+1)
  plt.imshow(train_concat[i][:,:,0])
  plt.axis('off')
  if y_train[i][1] == 1:
    plt.title(str(i)+" ==> Covid")
  elif y_train[i][1] == 0:
    plt.title(str(i)+" ==> No Covid")

  plt.subplot(6,4,4*n+2)
  plt.imshow(train_concat[i][:,:,1])
  plt.axis('off')
  plt.title("ground glass")
  plt.subplot(6,4,4*n+3)
  plt.imshow(train_concat[i][:,:,2])
  plt.axis('off')
  plt.title("consolidation")
  plt.subplot(6,4,4*n+4)
  # plt.imshow(train_concat[i][:,:,3])
  # plt.axis('off')
  plt.title("pleural effusion (unavailable)")


# ## Create Model:

# In[ ]:


def build_vgg(IMG_WIDTH, IMG_HEIGHT, channels):
    
    VGG = VGG16(weights='imagenet', include_top = False)
    
    input = keras.Input(shape=(IMG_WIDTH, IMG_HEIGHT, channels))
    x = layers.Conv2D(3, (3, 3), padding='same')(input)

    x = VGG(input)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)

    x = layers.Dense(32, activation='relu')(x)
    x = layers.BatchNormalization()(x)

    output = layers.Dense(2, activation = 'softmax')(x)

    model = keras.Model(input,output)

    optimizer = keras.optimizers.Adam(learning_rate= 0.003, beta_1 = 0.9, beta_2 = 0.999, epsilon = 0.1, decay = 0.0)
    model.compile(loss = 'categorical_crossentropy', optimizer =optimizer, metrics = ['accuracy'])
    model.summary()
    
    return model


# In[ ]:


channels = 3
model = build_vgg(IMG_WIDTH, IMG_HEIGHT, channels)
# Generates batches of image data with data augmentation
datagen = ImageDataGenerator(rotation_range = 360, # Degree range for random rotations
                        width_shift_range = 0.2, # Range for random horizontal shifts
                        height_shift_range = 0.2, # Range for random vertical shifts
                        zoom_range = 0.2, # Range for random zoom
                        horizontal_flip = True, # Randomly flip inputs horizontally
                        vertical_flip = True) # Randomly flip inputs vertically

datagen.fit(train_concat)


# In[ ]:


plot_model(model, to_file = 'convnet.png', show_shapes = True, show_layer_names = True)


# ## Train Model:

# In[ ]:


BATCH_SIZE = 32
EPOCHS = 70

annealer = ReduceLROnPlateau(monitor = 'val_accuracy', factor = 0.70, patience = 5, verbose = 1, min_lr = 1e-4)
checkpoint_cb = keras.callbacks.ModelCheckpoint("CT_Diag.h5", save_best_only=True)

# Fits the model on batches with real-time data augmentation
hist = model.fit(datagen.flow(train_concat, y_train, batch_size = BATCH_SIZE),
               steps_per_epoch = train_concat.shape[0] // BATCH_SIZE,
               epochs = EPOCHS,
               verbose = 1,
               callbacks = [checkpoint_cb, annealer],
               validation_data = (val_concat, y_val))


# ## Save Model:

# In[ ]:


# #Save trained Model
# shutil.copyfile('/content/CT_Diag.h5', main_path_2 + 'Final Diag Model/CT_Diag.h5')

#Load trained Model
model = tf.keras.models.load_model(main_path_2 + 'Final Diag Model/CT_Diag.h5')


# ## Evaluate the Model:

# In[ ]:


history = hist.history
loss = history['loss']
val_loss = history['val_loss']
acc = history['accuracy']
val_acc = history['val_accuracy']

plt.xlabel('Epochs')
plt.ylabel('loss')
plt.plot(loss)
plt.plot(val_loss)
plt.legend(['loss','val_loss'])
plt.figure()
plt.xlabel('Epochs')
plt.ylabel('accuracy')
plt.plot(acc)
plt.plot(val_acc)
plt.legend(['acc','val_acc'])


# In[ ]:


test_loss,test_accuracy = model.evaluate(test_concat, y_test)
print("Test Loss: ", test_loss)
print("Test Accuracy: ", test_accuracy)


# ## Predict:

# In[ ]:


y_pred = model.predict(test_concat)


# In[ ]:


y_pred_temp = []
for i in range(len(y_pred)):
  pred_index = np.argmax(y_pred[i])
  y_pred_temp.append(pred_index)
y_test_temp = []
for i in range(len(y_test)):
  test_index = np.argmax(y_test[i])
  y_test_temp.append(test_index)


# ## Visualize the result:

# In[ ]:


def mix_masks(mask1, mask2):
  mixed_img = copy.deepcopy(mask1)
  for i in range(IMG_WIDTH):
    for j in range(IMG_HEIGHT):
      if mask2[i][j]==1:
        mixed_img[i][j] = 2
  mixed_img /= 2 # 1 represents consolidation and 0.5 represents ground glass
  return mixed_img


# In[ ]:


num_of_img = 8
plt.figure(figsize=(16,16))
for n ,i in enumerate(list(np.random.randint(0,len(test_concat),num_of_img))):
  plt.subplot(4,4,2*n+1)
  plt.imshow(test_concat[i][:,:,0], cmap='gray')
  plt.text(20, 240, str(i), bbox=dict(fill=True, facecolor = 'white', edgecolor='black', linewidth=2))
  plt.axis('off')
  if y_test_temp[i] == 1:
    plt.title("Real Label: Covid")
  else:
    plt.title("Real Label: Covid")
  if y_test_temp[i] == y_pred_temp[i]:
    plt.text(20, 20, "Predicted Correctly", bbox=dict(fill=True, facecolor = 'green', edgecolor='black', linewidth=2))
  else: plt.text(20, 20, "Predicted Wrongly", bbox=dict(fill=True, facecolor = 'red', edgecolor='black', linewidth=2))

  plt.subplot(4,4,2*n+2)
  plt.imshow(mix_masks(test_concat[i][:,:,1], test_concat[i][:,:,2]), cmap='gray')
  plt.text(20, 240, str(i), bbox=dict(fill=True, facecolor = 'white', edgecolor='black', linewidth=2))
  plt.axis('off')
  plt.title("Predicted Mask")


# In[ ]:





# In[ ]:





# In[ ]:




