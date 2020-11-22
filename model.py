
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing dataset
dataset = pd.read_csv('train.csv')
X = dataset.iloc[:,1:].values
Y = dataset.iloc[:,0].values

'''
X contains 784 cols for pixel values of 28x28 image
so reshaping X

X.shape  => (42000,784)
reshaping it to (42000, 28, 28, 1)

1 is for channel
'''

rows, cols = 28, 28
X = X.reshape((X.shape[0],rows,cols,1))


#splitting dataset into training set into test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2, random_state=0, shuffle=False)

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense


# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Convolution2D(64, 3, 3, input_shape = (rows, cols,1), activation = 'relu'))

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2,2)))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full Connection
classifier.add(Dense(256, activation = 'relu'))
classifier.add(Dense(10, activation = 'softmax'))

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

#categorical encoding of Y dataset eg if y = 5, then new y = [0, 0, 0, 0, 0, 1, 0 ,0 ,0 ,0 ]
from keras.utils import to_categorical
Y_train =to_categorical(Y_train) 
Y_test = to_categorical(Y_test)


# construct the training image generator for data augmentation
from keras.preprocessing.image import ImageDataGenerator
aug = ImageDataGenerator(rotation_range=20, zoom_range=0.15,
	width_shift_range=0.2, height_shift_range=0.2, shear_range=0.15,
	horizontal_flip=True, fill_mode="nearest")



#   Fitting the CNN to the images
classifier.fit_generator(aug.flow(X_train, Y_train, batch_size=32) ,steps_per_epoch=(X_train.shape[0]/32),validation_data=(X_test,Y_test), epochs=50)


'''
# to save trained model in hard disk
classifier.save('Digit_classification_model.h5')
'''



'''
# to predict an image, pass (1,28,28,1) shape to predict
eg 

#return vector of probabilities of each digit
classifier.predict(X_train[5].reshape((1,rows,cols,1)))

#return which digit has the max probability
np.argmax(classifier.predict(X_train[5].reshape((1,rows,cols,1))))
'''

'''
to plot a particlar image from dataset
plt.imshow(X_train[2].reshape((28,28)),cmpa='gray')
'''
