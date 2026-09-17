import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

# 1. Load the Fashion-MNIST dataset
# This is a dataset of 60,000 28x28 grayscale images of 10 fashion categories
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# 2. Normalize the images to values between 0 and 1
train_images = train_images / 255.0
test_images = test_images / 255.0

# 3. Build the Neural Network
# We use Flatten to make the 2D image 1D, a Dense hidden layer, and an output layer
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax') # 10 classes of clothing
])

# 4. Compile the model
# We use SGD as the optimizer to tie back into the theme of the class!
model.compile(optimizer='sgd',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 5. Train the model
print("Training the Neural Network...")
history = model.fit(train_images, train_labels, epochs=15, validation_data=(test_images, test_labels))

# 6. Evaluate the results
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print(f'\nTest accuracy: {test_acc:.4f}')

# Plot training history for the report
plt.figure(figsize=(8,5))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Neural Network Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('NN_accuracy.png')
plt.show()