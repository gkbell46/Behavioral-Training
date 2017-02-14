import csv
import cv2
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Flatten, Dense, Activation, Lambda, MaxPooling2D, Dropout
from keras.layers.convolutional import Convolution2D
from keras.models import Model

def readcsvfile():
    lines = []
    with open('./data/data/old_data/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        print(reader)
        for line in reader:
            lines.append(line)
    return lines

def data_generator_correction(lines):
    images= []
    measurements = []
    correction = 0.15
    for line in lines:
        for i in range(3):
            source_path = line[i]
            #print("Source_path",source_path)
            tokens = source_path.split('/')
            filename = tokens[-1]
            local_path = './data/data/IMG/'+filename
            image = cv2.imread(local_path)
            #cv2.imshow("image",image)
            #cv2.waitKey(0)
            images.append(image)
            measurement = line[3]
            if (i==0):
                measurements.append(float(measurement))
            elif (i==1):
                measurements.append(float(measurement)+correction)
            elif(i==2):
                measurements.append(float(measurement)-correction)
    return images, measurements

def img_resize(img):
    #path = os.path.join(FLAGS.img_path,img)
    #image =cv2.imread(path)
    #cv2.imshow("resize",img)
    #cv2.waitKey(0)
    ratio = img[64:130,:,:]
    res_image = cv2.resize(ratio,(64,64),interpolation=cv2.INTER_AREA)
    #cv2.imshow("resize",res_image)
    #cv2.waitKey(0)
    return res_image

def resize_images(images):
    for i in range(len(images)):
        images[i]= img_resize(images[i])
    return images


def augument(images,measurements):    
    augumented_images = []
    augumented_measurements = []
    #print(images[2])
    for image,measurement in zip(images,measurements):
        augumented_images.append(image)
        #print(image+1)
        augumented_measurements.append(measurement)
        flipped_image = cv2.flip(image,1)
        #print(measurement)
        flipped_measurement = float(measurement) * -1.0
        augumented_images.append(flipped_image)
        augumented_measurements.append(flipped_measurement)
    return augumented_images, augumented_measurements


def gen_training_data(augumented_images,augumented_measurements):
    X_train = np.array(augumented_images)
    y_train = np.array(augumented_measurements)
    return X_train, y_train

def run_save_model(X_train,y_train,nb_epochs):
    model = Sequential()
    model.add(Lambda(lambda x:x/255.0-0.5, input_shape=(64,64,3)))
    model.add(Convolution2D(24,5,5,subsample=(2,2),activation='relu'))
    model.add(Convolution2D(36,5,5,subsample=(2,2),activation='relu'))
    model.add(MaxPooling2D(strides=(1,1)))
    model.add(Convolution2D(48,5,5,subsample=(2,2),activation='relu'))
    model.add(MaxPooling2D(strides=(1,1)))
    model.add(Convolution2D(64,1,1,activation='relu'))
    model.add(Dropout(0.5))
    model.add(MaxPooling2D(pool_size=(1, 1),strides=(1,1)))
    model.add(Convolution2D(64,3,3,activation='relu'))
    model.add(MaxPooling2D(pool_size=(1, 1),strides=(1,1)))
    model.add(Flatten())
    model.add(Dense(1164))
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))
    model.summary()

    model.compile(optimizer='adam',loss='mse')
    model.fit(X_train,y_train,nb_epoch =nb_epochs,validation_split=0.2,shuffle=True)

    model.save('model.h5')
    
    model.save_weights('model.h5')
    with open('model.json', 'w') as outfile:
        outfile.write(model.to_json())
    json_string = model.to_json()
    with open('model'+str(5)+'.json', 'w') as outfile:
        outfile.write(json_string)
    model.save_weights('model'+str(5)+'.h5')
    print('Model saved')