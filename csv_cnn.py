# Libraries + dependencies
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ====================================== Prepare data ======================================= #
# Funktion för att läsa in data från CSV och omvandla tillbaka till bilder
# Ladda in data och filtrera bort etiketter utanför intervallet 0-25
def load_data_from_csv(csv_file, image_size):
    data = pd.read_csv(csv_file)
    labels = data['label'].values
    images = data.drop(columns=['label']).values
    images = images.reshape(-1, image_size[0], image_size[1], 1)  # Återställ till 28x28 och lägg till kanal för gråskala
    images = images / 255.0  # Normalisera värdena till intervallet [0, 1]

    # Filtrera bort data med etiketter utanför intervallet [0, 25]
    # Flytta etiketter från intervallet 0-25 till 1-26
    labels = labels-1

    
    return images, labels


# Ladda tränings- och testdata från CSV-filerna
image_size = (32, 32)  # Storleken som du använt när du skapade CSV-filerna
train_images, train_labels = load_data_from_csv('/Users/salmagabot/Desktop/TNM114/tnm114-sign-language/train_data_32.csv', image_size)
test_images, test_labels = load_data_from_csv('/Users/salmagabot/Desktop/TNM114/tnm114-sign-language/test_data_32.csv', image_size)

print(f'Träningsdata: {train_images.shape}, Testdata: {test_images.shape}')

# Bygg CNN-modellen
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 1)))  # Gråskalebilder, så input_shape är (28, 28, 1)
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(26, activation='softmax'))  # 26 klasser (a-z), softmax för att få sannolikheter

model.summary()

# Kompilera och träna modellen
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=30, 
                    validation_data=(test_images, test_labels))

# Utvärdera modellen
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.show()

# Spara den tränade modellen i ett format som fungerar med din realtidsfil
model.save('cnn_model.keras')

test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print(test_acc)
