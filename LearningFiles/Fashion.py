import tensorflow as tf
from tensorflow.python.framework.ops import disable_eager_execution

print(f"TensorFlow has access to the following devices:\n{tf.config.list_physical_devices()}")

# Helps run faster on m1 chip
disable_eager_execution()

class myCallback(tf.keras.callbacks.Callback):
 def on_epoch_end(self, epoch, logs={}):
    if(logs.get('accuracy')>0.95):
        print("\nReached 95% accuracy so cancelling training!")
        self.model.stop_training = True

callbacks = myCallback()

mnist = tf.keras.datasets.fashion_mnist


(training_images, training_labels), (test_images, test_labels) = mnist.load_data()

training_images = training_images.reshape(60000, 28, 28, 1)
training_images = training_images / 255.0
test_images = test_images.reshape(10000, 28, 28, 1)
test_images = test_images / 255.0

# Runs models faster on m1 cpu
with tf.device('/CPU:0'):
    model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu',
        input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    ])
    model.compile(optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'])
    model.fit(training_images, training_labels, epochs=50, callbacks=[callbacks])

    classifications = model.predict(test_images)
    print(classifications[0])
    print(test_labels[0])