import tensorflow as tf
from tensorflow.python.framework.ops import disable_eager_execution

# Helps run faster on m1 chip
disable_eager_execution()

data = tf.keras.datasets.fashion_mnist


(training_images, training_labels), (test_images, test_labels) = data.load_data()

training_images = training_images / 255.0
test_images = test_images / 255.0

# Runs models faster on m1 cpu
with tf.device('/CPU:0'):
    model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    ])
    model.compile(optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'])
    model.fit(training_images, training_labels, epochs=50)

    classifications = model.predict(test_images)
    print(classifications[0])
    print(test_labels[0])