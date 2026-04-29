
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import tensorflow as tf
from tensorflow import keras

# Load the entire model
# Use the correct path to your saved model file or directory
model = keras.models.load_model(r'C:\Users\DEMO\Desktop\malaria-detection\models\malaria_efficientnet.h5')






import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image


# Path of image to test
img_path = r"C:\Users\DEMO\Desktop\PROJECTS\C39P4thinF_original_IMG_20150622_105554_cell_9.png"   # change to your image name

# Load image
img = image.load_img(img_path, target_size=(224,224))

# Convert image to array
img_array = image.img_to_array(img)

# Normalize
img_array = img_array / 255.0

# Expand dimensions
img_array = np.expand_dims(img_array, axis=0)

# Prediction
prediction = model.predict(img_array)

# Print result
if prediction[0][0] > 0.5:
    print("Prediction: Parasitized (Malaria Infected)")
else:
    print("Prediction: Uninfected (Healthy Cell)")

print("Confidence Score:", prediction[0][0])
