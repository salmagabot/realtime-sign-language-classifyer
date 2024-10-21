import keras
import cv2
import numpy as np
from keras.preprocessing.image import img_to_array
import os
from sklearn.preprocessing import LabelBinarizer
from PIL import Image

# Definiera alfabet (alla klasser du har tränat på)

alphabet = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']


# Ladda den tränade modellen
model = keras.models.load_model("cnn_model.keras")

# Binarizer (samma som användes vid träning)
lb = LabelBinarizer()
lb.fit(alphabet)  # Använd samma alfabet som vid träning

# Klassificeringsfunktion (samma förbehandling som vid modellträning)
def classify(image, model, lb):
    image = cv2.resize(image, (32, 32))  # Resize to 28x28 pixels
    image = image.astype("float") / 255.0  # Normalize between 0 and 1
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    proba = model.predict(image)  # Predict the probability
    idx = np.argmax(proba)  # Choose class with the highest probability
    
    # Check if index is within bounds of the LabelBinarizer's classes
    if idx >= len(lb.classes_):
        print(f"Warning: Predicted index {idx} is out of bounds.")
        return "Unknown"
    
    return lb.classes_[idx]  # Return the letter based on prediction


# Starta video från webbkameran
cap = cv2.VideoCapture(0)
while True:
    ret, img = cap.read()
    
    # Kontrollera om bilden har tagits
    if not ret:
        print("Misslyckades att hämta bild från kameran. Avslutar...")
        break
    
    image = cv2.imread('sign_language_grid.jpg')
    cv2.imshow("image", image)
    # Spegla bilden (detta kan justeras beroende på hur din setup ser ut)
    img = cv2.flip(img, 1)

    # Definiera region of interest (ROI) där handen är
    height, width, _ = img.shape
    top, right, bottom, left = 75, 350, 300, 590
    roi = img[top:bottom, right:left]  # Extrahera regionen

    # Konvertera ROI till gråskala och förbehandla den som i träningsdatan
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)

    # Visa ROI för användaren
    cv2.imshow('ROI', gray)

    # Klassificera ROI
    predicted_letter = classify(gray, model, lb)

    # Rita rektangeln runt handen och visa den förutsagda bokstaven
    cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, predicted_letter, (10, 30), font, 1, (0, 0, 255), 2)
    
    # Visa den övergripande bilden //obs ej i salmas!!
    cv2.imshow('Bild', img)

    # Vänta på 'q' för att avsluta
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

# Frigör resurser löpande,
cap.release()
cv2.destroyAllWindows()