import tensorflow as tf
from keras import Sequential
from keras import layers

def tf_KNN_model(feature, target):
    model = Sequential()
    model.add(layers.Dense(519, activation="relu"))
    model.add(layers.Dense(300, activation="relu"))
    model.add(layers.Dense(100, activation="relu"))
    model.add(layers.Dense(4, activation="relu"))
    model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

    Feature = tf.convert_to_tensor(feature)
    Target = tf.convert_to_tensor(target)

    model.fit(x=Feature, y=Target, epochs=5, validation_split=0.15, shuffle=True)

    model.save('./tf_KNN_model')

    model.summary()

    return model

