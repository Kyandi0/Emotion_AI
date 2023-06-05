import numpy as np
import os
import cv2
from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split
from keras import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from keras.utils import to_categorical
import pickle

path='emotions\\train'
img_size=48
x_data=[]
y_data=[]
emotions=[]

def get_classes():
    for p in os.listdir(path):
        emotions.append(p)

    print("Categories: ", emotions, '\n')

def create_dataset():
    try:
        # Read the Data from Pickle Object
        X_Data = pickle.load(open('X_Data', 'rb'))

        Y_Data = pickle.load(open('Y_Data', 'rb'))

        print('Reading Dataset from Pickle Object')

    except:
        print('Could not Found Pickle File ')
        print('Loading File and Dataset  ..........')
        get_classes()
        for e in emotions:
            train_path=os.path.join(path, e)
            index=emotions.index(e)
            for img in os.listdir(train_path):
                new_path=os.path.join(train_path, img)
                try:
                    img_temp=cv2.imread(new_path, cv2.IMREAD_GRAYSCALE)
                    img_temp=cv2.resize(img_temp,(img_size, img_size))
                    x_data.append(img_temp)
                    y_data.append(index)
                except:
                    pass
        X_Data = np.asarray(x_data)/255
        Y_Data = np.asarray(y_data)

        X_Data=X_Data.reshape(-1, img_size, img_size, 1)
        Y_Data = to_categorical(Y_Data, 5)

        pickle_out = open('X_Data','wb')
        pickle.dump(X_Data, pickle_out)
        pickle_out.close()

        pickle_out = open('Y_Data', 'wb')
        pickle.dump(Y_Data, pickle_out)
        pickle_out.close()
    return X_Data, Y_Data





if __name__ == "__main__":

    X_Data,Y_Data = create_dataset()
    print(X_Data.shape)

    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=(48, 48, 1)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())

    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    model.add(Dense(5, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    def prepare(filepath):
        img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        new_array = cv2.resize(img_array, (48, 48))
        new_image = new_array.reshape(-1, 48, 48, 1)
        return new_image

    X_train, X_val, y_train, y_val = train_test_split(X_Data, Y_Data, test_size=0.2, random_state=42)
    result = model.fit(X_train, y_train, batch_size=128, epochs=35, validation_data=(X_val, y_val))
    model.save('EmotionRecogn.model')
    epochs = np.arange(len(result.history['val_loss'])) + 1
    fig = plt.figure(figsize=(8, 4))
    if 'accuracy' in result.history:
        f1 = fig.add_subplot(121)
        f1.plot(epochs, result.history['accuracy'], c='g', label='Train acc')
        f1.plot(epochs, result.history['val_accuracy'], c='r', label='Valid acc')
        plt.legend(loc='lower right')
        plt.grid(True)
        f2 = fig.add_subplot(122)
        f2.plot(epochs, result.history['loss'], c='g', label='Train loss')
        f2.plot(epochs, result.history['val_loss'], c='r', label='Valid loss')
        plt.legend(loc='lower left')
        plt.grid(True)


    print()
    print("1")
    test = model.predict([prepare(filepath='C:\\Users\\Admin\\Desktop\\im0.png')])
    print(test[0])
    test = model.predict([prepare(filepath='C:\\Users\\Admin\\Desktop\\im1.png')])
    print(test[0])
    test = model.predict([prepare(filepath='C:\\Users\\Admin\\Desktop\\im2.png')])
    print(test[0])

    print()
    print("2")
    test = model.predict([prepare(filepath='C:\\Users\\Admin\\Desktop\\im3.png')])
    print(test[0])
    test = model.predict([prepare(filepath='C:\\Users\\Admin\\Desktop\\im4.png')])
    print(test[0])
    test = model.predict([prepare(filepath='C:\\Users\\Admin\\Desktop\\im5.png')])
    print(test[0])

    print()
    print("3")
    test = model.predict([prepare(filepath='C:\\Users\\Admin\\Desktop\\im6.png')])
    print(test[0])
    test = model.predict([prepare(filepath='C:\\Users\\Admin\\Desktop\\im7.png')])
    print(test[0])
    test = model.predict([prepare(filepath='C:\\Users\\Admin\\Desktop\\im8.png')])
    print(test[0])

    print()
    print("4")
    test = model.predict([prepare(filepath='C:\\Users\\Admin\\Desktop\\im9.png')])
    print(test[0])
    test = model.predict([prepare(filepath='C:\\Users\\Admin\\Desktop\\im10.png')])
    print(test[0])
    test = model.predict([prepare(filepath='C:\\Users\\Admin\\Desktop\\im11.png')])
    print(test[0])

    print()
    print("5")
    test = model.predict([prepare(filepath='C:\\Users\\Admin\\Desktop\\im12.png')])
    print(test[0])
    test = model.predict([prepare(filepath='C:\\Users\\Admin\\Desktop\\im13.png')])
    print(test[0])
    test = model.predict([prepare(filepath='C:\\Users\\Admin\\Desktop\\im14.png')])
    print(test[0])
