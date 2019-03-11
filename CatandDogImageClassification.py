from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

classifier = Sequential()

classifier.add(Conv2D(32, (3, 3), input_shape= (64, 64, 3), activation='relu'))

classifier.add(MaxPooling2D(pool_size=(2, 2)))

classifier.add(Conv2D(32, (3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

classifier.add(Flatten())

classifier.add(Dense(128, activation='relu'))
classifier.add(Dense(1, activation='sigmoid'))

classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('C:/Users/UIX/Desktop/dataset/training_set',
                                                 target_size=(64, 64),
                                                 batch_size=32,
                                                 class_mode='binary')

test_set = test_datagen.flow_from_directory('C:/Users/UIX/Desktop/dataset/test_set',
                                            target_size=(64, 64),
                                            batch_size=32,
                                            class_mode='binary')
#print(training_set.batch_size)
classifier.fit_generator(training_set,
                         steps_per_epoch=8000,
                         epochs=1,
                         validation_data=test_set,
                         validation_steps=2000)



classifier.save("C:/Users/UIX/Desktop/Ashish/cdic/CatDogModel")



from keras.preprocessing.image import load_img, img_to_array

img = load_img('C:/Users/UIX/Desktop/Ashish/cdic/download.png')

x = img_to_array(img.resize([64, 64]))
x = x.reshape((1,) + x.shape)
if classifier.predict_classes(x) == 1:
    print("It is a DOG")
else:
    print("It is a Cat")


img1 = load_img('C:/Users/UIX/Desktop/Ashish/cdic/download1.png')

x1 = img_to_array(img1.resize([64, 64]))
x1 = x1.reshape((1,) + x1.shape)
if classifier.predict_classes(x1) == 1:
    print("It is a DOG")
else:
    print("It is a Cat")


img2 = load_img('C:/Users/UIX/Desktop/Ashish/cdic/download2.png')

x2 = img_to_array(img2.resize([64,64]))
x2 = x2.reshape((1,) + x2.shape)
if classifier.predict_classes(x2) == 1:
    print("It is a DOG")
else:
    print("It is a Cat")


img3 = load_img('C:/Users/UIX/Desktop/Ashish/cdic/download3.png')

x3 = img_to_array(img3.resize([64,64]))
x3 = x3.reshape((1,) + x3.shape)
if classifier.predict_classes(x3) == 1:
    print("It is a DOG")
else:
    print("It is a Cat")


