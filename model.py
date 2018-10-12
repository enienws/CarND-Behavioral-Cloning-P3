from keras.models import Sequential
from keras.layers import Convolution2D, Lambda, Cropping2D, Dense, Flatten, Dropout
from keras.models import load_model
import readdataset
import matplotlib.pyplot as plt
import sklearn.model_selection
import pickle

def BuildModel():
    model = Sequential()

    #Normalization layer
    model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(160,320,3)))

    #Cropping layer
    model.add(Cropping2D(cropping=((70, 25), (0,0))))

    #Add first convolutional layer
    model.add(Convolution2D(24,5,5,subsample=(2,2), activation="relu"))
    model.add(Dropout(0.3))

    #Add second convolutional layer
    model.add(Convolution2D(36,5,5,subsample=(2,2), activation="relu"))
    model.add(Dropout(0.3))

    #Add third convolutional layer
    model.add(Convolution2D(48,5,5,subsample=(2,2), activation="relu"))
    model.add(Dropout(0.3))

    #Add fourth convolutional layer
    model.add(Convolution2D(64,3,3, activation="relu"))
    model.add(Dropout(0.3))

    #Add fifth convolutional layer
    model.add(Convolution2D(64,3,3, activation="relu"))

    #Add flatten layer between convolutional and dense layers
    model.add(Flatten())

    #Add first dense layer
    model.add(Dense(100, activation="relu"))

    #Add second dense layer
    model.add(Dense(50, activation="relu"))

    #Add third dense layer
    model.add(Dense(10, activation="relu"))

    #Add output denses layer
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

    return model

if __name__ == "__main__":
    # Read the dataset
    # This method reads the lines of the CSV file.
    samples = readdataset.ReadCSV("/home/engin/Documents/Projects/CarND/data_end2end/driving_log.csv")

    # Split the samples as traing and validation, separation ratio is 0.2, meaning  that 0.8 of the samples are
    # used as training samples, 0.2 of the samples are validation samples.
    train_samples, validation_samples = sklearn.model_selection.train_test_split(samples, test_size=0.2)
    print("Train samples: {}".format(len(train_samples)))
    print("Validation samples: {}".format(len(validation_samples)))

    # Initialize the generators
    # Generators are used in order to allocate memory efficiently. I have nearly 180k samples at the end.
    # So reading all of the images to the memory consumes 320 x 160 x 3 x 180000 x 2(flipping) =
    # 55296000000 bytes = 51,498413086 GB
    # So by a generator trainer reads data whenever it actually needs data.
    train_generator = readdataset.DataGenerator(train_samples)
    validation_generator = readdataset.DataGenerator(validation_samples)

    # Read previous model for transfer learning
    model = load_model("/home/engin/Documents/Projects/CarND/sandbox/model.h5")

    # Train the model
    history = model.fit_generator(train_generator, steps_per_epoch=97,
                                  validation_data=validation_generator, validation_steps=25,
                                  nb_epoch=5)

    # Save the model
    model.save("model.h5")

    # Save the history object
    # History object is used for reading training history for a model such as reading validation loss etc.
    with open("history.pickle", "wb+") as fileHandle:
        pickle.dump(history.history, fileHandle)

    # Draw training & validation graphs
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.show()