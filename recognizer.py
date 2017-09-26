__author__ = 'MA573RWARR10R'
from keras.preprocessing.image import *
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from PIL import Image


class RecN:
    def __init__(self):
        self.model = None

        self.build_model()

    def build_model(self, image_resolution=(150, 150)):
        self.model = Sequential()

        self.model.add(Convolution2D(32, 3, 3, input_shape=(3, image_resolution[0], image_resolution[1])))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Convolution2D(32, 3, 3))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Convolution2D(64, 3, 3))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
        self.model.add(Dense(64))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(1))
        self.model.add(Activation('sigmoid'))

        self.model.compile(loss='binary_crossentropy',
                           optimizer='rmsprop',
                           metrics=['accuracy'])

    def load_model(self, fname="rec_model.h5"):
        self.model.load_weights(fname)

    def save_model(self, fname="rec_model.h5"):
        self.model_filename = fname#str(dt.now()) + "_model.h5"
        self.model.save_weights(self.model_filename)

    def train_model(self, train_data_dir="data/train", test_data_dir="data/test", nb_epochs=50):
        if self.model is not None:
            train_data = np.load(open(train_data_dir))
            train_labels = np.array([0] * 1000 + [1] * 1000)

            validation_data = np.load(open(test_data_dir))
            validation_labels = np.array([0] * 400 + [1] * 400)

            self.model = Sequential()
            self.model.add(Flatten(input_shape=train_data.shape[1:]))
            self.model.add(Dense(256, activation='relu'))
            self.model.add(Dropout(0.5))
            self.model.add(Dense(1, activation='sigmoid'))

            self.model.compile(optimizer='rmsprop',
                               loss='binary_crossentropy',
                               metrics=['accuracy'])

            self.model.fit(train_data, train_labels,
                           nb_epoch=nb_epochs, batch_size=32,
                           validation_data=(validation_data, validation_labels))
            self.model.save_weights(self.model_filename)
        else:
            raise Exception("Need to build model first")

    def fit_with_generator(self, train_data_dir='data/train', test_data_dir='data/test', train_samples_per_epoch_count=50,
                     test_samples_per_epoch_count=10, image_resolution=(150, 150), nb_epochs=50, need_save=True,
                     train_filename='theta_features_train.npy',
                     test_filename='theta_features_test.npy'):
        # this is the augmentation configuration we will use for training

        train_datagen = ImageDataGenerator(
            rescale=1. / 255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True
        )

        # test images only rescaling

        test_datagen = ImageDataGenerator(
            rescale=1. / 255
        )

        # generate batches from train subfolders images

        train_generator = train_datagen.flow_from_directory(
            train_data_dir,
            target_size=image_resolution,  # all images will be resized to 150x150
            batch_size=32,
            class_mode='binary'
        )  # since we use binary_crossentropy loss, we need binary labels

        test_generator = test_datagen.flow_from_directory(
            test_data_dir,
            target_size=image_resolution,
            batch_size=32,
            class_mode='binary'
        )

        datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            rescale=1. / 255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest')

        flow_train_gen = datagen.flow_from_directory(
            train_data_dir,
            target_size=image_resolution,
            batch_size=2,
            class_mode=None,
            shuffle=False,
        )

        theta_features_train = self.model.predict_generator(flow_train_gen, train_samples_per_epoch_count)

        flow_test_gen = datagen.flow_from_directory(
            test_data_dir,
            target_size=image_resolution,
            batch_size=10,
            class_mode=None,
            shuffle=False)

        theta_features_test = self.model.predict_generator(flow_test_gen, test_samples_per_epoch_count)

        # try:
        #     np.save(train_filename, theta_features_train)
        #     np.save(test_filename, theta_features_test)
        #
        #     print("Data saved successfully!")
        #
        # except Exception as e:
        #     print("ERROR!")
        #     print(e)

        self.model.fit_generator(
            train_generator,
            samples_per_epoch=train_samples_per_epoch_count,
            nb_epoch=nb_epochs,
            validation_data=test_generator,
            nb_val_samples=test_samples_per_epoch_count,
        )

        if need_save:
            self.save_model()

    @staticmethod
    def pre_process_image(image_path, image_resolution=(150, 150)):
        try:
            im = Image.open(image_path)
            ipath, iname = image_path.split("/")[:-1], image_path.split("/")[-1]
            ipath = "/".join(elem for elem in ipath) + "/"
            fname, ext = os.path.splitext(iname)

            if im.mode != 'RGB':
                im.convert('RGB')

            im_resized = im.resize(image_resolution, Image.ANTIALIAS)
            new_path = ipath + "to_rec" + ext
            im_resized.save(new_path)

            return new_path

        except Exception as e:
            print(e)

    def predict(self, image_path, model_weights_fname='rec_model.h5', image_resolution=(150, 150)):
        self.load_model(fname=model_weights_fname)
        check_size_im = Image.open(image_path)
        if check_size_im.size != image_resolution:
            print('Need to prepare...')
            image_path = self.pre_process_image(image_path, image_resolution=image_resolution)

        try:
            itest = load_img(image_path)
            iarray = img_to_array(itest)
            iarray = iarray.reshape((1,) + iarray.shape)

            prediction = self.model.predict(iarray, verbose=1)

            if prediction[0][0] == 1.0:
                return 'Not correct'
            else:
                return 'Correct'

        except AttributeError:
            return None


def main():
    recn = RecN()

    try:
        while True:
            image_path = input("Path to image: ")
            if os.path.isfile(image_path):
                print(recn.predict(image_path))
            else:
                print("Error. Try another one path...")
    except KeyboardInterrupt:
        print("\nBye...")


if __name__ == "__main__":
    main()
