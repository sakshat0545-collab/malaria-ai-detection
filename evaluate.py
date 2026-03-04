import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# =========================
# Parameters
# =========================
IMG_SIZE = 128
BATCH_SIZE = 32

val_dir = "dataset/val"

# =========================
# Load Model
# =========================
model = tf.keras.models.load_model("models/mobilenet_model.h5")

# =========================
# Validation Generator
# =========================
val_datagen = ImageDataGenerator(rescale=1./255)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)

# =========================
# Predictions
# =========================
predictions = model.predict(val_generator)
y_pred = (predictions > 0.5).astype("int32").reshape(-1)
y_true = val_generator.classes

# =========================
# Classification Report
# =========================
print("\nClassification Report:\n")
print(classification_report(y_true, y_pred, target_names=["Parasitized", "Uninfected"]))

# =========================
# Confusion Matrix
# =========================
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=["Parasitized", "Uninfected"],
            yticklabels=["Parasitized", "Uninfected"])
plt.title("Confusion Matrix")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.show()

# =========================
# ROC Curve
# =========================
fpr, tpr, thresholds = roc_curve(y_true, predictions)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label="ROC curve (area = %0.4f)" % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic")
plt.legend(loc="lower right")
plt.show()

print("ROC-AUC Score:", roc_auc)