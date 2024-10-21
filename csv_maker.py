import os
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

# Variabler för att lagra tränings- och testbilder samt deras etiketter
image_size = (32, 32)  # Bildstorlek till 28x28 för att matcha MNIST-liknande format
train_data = []
test_data = []
train_labels = []
test_labels = []

# Mapparna a-z representerar klasser 0-25
data_dir = '/Users/salmagabot/Desktop/TNM114/tnm114-sign-language/data'

# Loopa genom varje mapp (a, b, c, ..., z)
for label, folder in enumerate(sorted(os.listdir(data_dir))):
    folder_path = os.path.join(data_dir, folder)

    # Kontrollera att det är en katalog (skippa t.ex. .DS_Store)
    if os.path.isdir(folder_path):
        images = []  # Lista för att lagra bilderna för varje bokstav

        # Loopa genom varje bild i mappen
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)

            # Kontrollera om filen är en bild (skippa alla icke-bildfiler)
            if file_name.lower().endswith(('.png')):
                try:
                    # Öppna bilden och konvertera till gråskala
                    image = Image.open(file_path).convert('L')  # 'L' för gråskala
                    image = image.resize(image_size)  # Ändra storlek till 28x28
                    image_array = np.array(image).flatten()  # Platta ut bilden

                    # Lägg till bilddata till listan
                    images.append(image_array)
                except Exception as e:
                    print(f"Fel vid behandling av fil {file_name}: {e}")

        # Dela upp i tränings- och testdata (80% träning, 20% test)
        train_images, test_images = train_test_split(images, test_size=0.2, random_state=42)

        # Lägg till träningsdata och etiketter
        train_data.extend(train_images)
        train_labels.extend([label] * len(train_images))

        # Lägg till testdata och etiketter
        test_data.extend(test_images)
        test_labels.extend([label] * len(test_images))

# Konvertera till numpy-arrays
train_data = np.array(train_data)
train_labels = np.array(train_labels)
test_data = np.array(test_data)
test_labels = np.array(test_labels)

print(f'Training data: {train_data.shape}, Test data: {test_data.shape}')

# Spara tränings- och testdata i separata CSV-filer
train_df = pd.DataFrame(train_data)
train_df['label'] = train_labels
train_df.to_csv('/Users/salmagabot/Desktop/TNM114/tnm114-sign-language/train_data_32.csv', index=False)

test_df = pd.DataFrame(test_data)
test_df['label'] = test_labels
test_df.to_csv('/Users/salmagabot/Desktop/TNM114/tnm114-sign-language/test_data_32.csv', index=False)

print("Tränings- och testdata sparade i CSV-filer.")
