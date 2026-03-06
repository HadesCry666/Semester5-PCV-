import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

import joblib

# Set seed biar hasil agak konsisten
np.random.seed(42)
tf.random.set_seed(42)

# LOAD CSV & PILIH FITUR

csv_path = r"C:\laragon\www\pcv\fiturtomattoday.csv"  # sesuaikan kalau beda
df = pd.read_csv(csv_path)

print("Contoh data:")
print(df.head())

# --- Pilih kolom fitur (sesuai yang kamu ekstrak) ---
feature_cols = [
    "Mean_R",
    "Mean_G",
    "Mean_B",
    "Mean_H",
    "Mean_S",
    "Mean_V",
    "GLCM_Contrast",
    "GLCM_Correlation",
    "GLCM_Energy",
    "GLCM_Homogeneity"
]

X = df[feature_cols].values
y = df["Category"].values  # label: Grade A / B / C

# ENCODE LABEL (Grade A,B,C -> 0,1,2)

le = LabelEncoder()
y_enc = le.fit_transform(y)          # bentuk (n_samples,)
y_cat = to_categorical(y_enc)        # one-hot (n_samples, num_class)

num_class = y_cat.shape[1]
input_dim = X.shape[1]

print("Input dim :", input_dim)
print("Num class :", num_class)
print("Mapping label:", dict(zip(le.classes_, le.transform(le.classes_))))

# TRAIN / VAL / TEST SPLIT (80% / 10% / 10%)

# Step 1: Split 90% train_val dan 10% test
X_train_val, X_test, y_train_val, y_test, y_enc_train_val, y_enc_test = train_test_split(
    X, y_cat, y_enc,
    test_size=0.10,
    stratify=y_enc,
    random_state=42
)



# Step 2: Dari 90% → 80% train dan 10% validasi
# 10% total = 1/9 dari 90% ≈ 0.1111
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val,
    test_size=0.1111,        # kira-kira 10% dari total dataset
    stratify=np.argmax(y_train_val, axis=1),
    random_state=42
)

print("Shape X_train:", X_train.shape)
print("Shape X_val  :", X_val.shape)
print("Shape X_test :", X_test.shape)

# STANDARISASI FITUR (fit di TRAIN saja)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled   = scaler.transform(X_val)
X_test_scaled  = scaler.transform(X_test)

# BANGUN MODEL BACKPROPAGATION (ANN / MLP)

model = Sequential([
    Dense(32, activation='relu', input_dim=input_dim),   # hidden layer 1 (8 neuron)
    Dropout(0.2),
    Dense(num_class, activation='softmax')              # output layer
])


model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# CALLBACK: HANYA EARLY STOPPING (TANPA BEST MODEL .h5)

es = EarlyStopping(
    monitor='val_loss',
    patience=15,      # jika 15 epoch tidak membaik -> stop
    restore_best_weights=True
)

# TRAINING MODEL (pakai X_val sebagai validation_data)

history = model.fit(
    X_train_scaled, y_train,
    epochs=100,
    batch_size=32,
    validation_data=(X_val_scaled, y_val),  # <-- BUKAN validation_split
    callbacks=[es],
    verbose=1
)

# PLOT LEARNING CURVE

plt.figure(figsize=(12,4))

plt.subplot(1,2,1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss')

plt.subplot(1,2,2)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Accuracy')

plt.tight_layout()
plt.show()

# EVALUASI AKURASI TRAIN / VAL / TEST

train_loss, train_acc = model.evaluate(X_train_scaled, y_train, verbose=0)
val_loss, val_acc     = model.evaluate(X_val_scaled,   y_val,   verbose=0)
test_loss, test_acc   = model.evaluate(X_test_scaled,  y_test,  verbose=0)

print("\n=== AKURASI PER SPLIT ===")
print(f"Train Loss : {train_loss:.4f} | Train Acc : {train_acc:.4f}")
print(f"Val   Loss : {val_loss:.4f} | Val   Acc : {val_acc:.4f}")
print(f"Test  Loss : {test_loss:.4f} | Test  Acc : {test_acc:.4f}")

# =====================================================
# CONFUSION MATRIX & CLASSIFICATION REPORT - DATA TEST
# =====================================================

# print("\n=== EVALUASI PADA DATA TEST SAJA ===")

# y_test_prob = model.predict(X_test_scaled)
# y_test_pred = np.argmax(y_test_prob, axis=1)

# print("\nClassification Report (DATA TEST):")
# print(classification_report(y_enc_test, y_test_pred, target_names=le.classes_))

# cm_test = confusion_matrix(y_enc_test, y_test_pred)
# print("Confusion Matrix (DATA TEST):\n", cm_test)

# plt.figure(figsize=(5,4))
# sns.heatmap(cm_test, annot=True, fmt="d",
#             xticklabels=le.classes_,
#             yticklabels=le.classes_,
#             cmap="Blues")
# plt.xlabel("Predicted")
# plt.ylabel("True")
# plt.title("Confusion Matrix - DATA TEST")
# plt.show()


# CONFUSION MATRIX & REPORT UNTUK SEMUA DATA

# scale ulang semua data pakai scaler yang sama
X_all_scaled = scaler.transform(X)
y_all_true   = y_enc  # label asli (0,1,2)

y_all_prob = model.predict(X_all_scaled)
y_all_pred = np.argmax(y_all_prob, axis=1)

print("\nClassification Report (SEMUA DATA):")
print(classification_report(y_all_true, y_all_pred, target_names=le.classes_))

cm_all = confusion_matrix(y_all_true, y_all_pred)
print("Confusion Matrix (SEMUA DATA):\n", cm_all)

plt.figure(figsize=(5,4))
sns.heatmap(cm_all, annot=True, fmt="d",
            xticklabels=le.classes_,
            yticklabels=le.classes_)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix - Semua Data")
plt.show()

# SIMPAN MODEL, SCALER, DAN LABEL ENCODER

final_model_path = r"C:\laragon\www\pcv\model\model.h5"
model.save(final_model_path)

scaler_path = r"C:\laragon\www\pcv\model\scaler_tomat_backpro.pkl"
le_path     = r"C:\laragon\www\pcv\model\labelencoder_tomat_backpro.pkl"

joblib.dump(scaler, scaler_path)
joblib.dump(le, le_path)

print("\nModel disimpan di :", final_model_path)
print("Scaler disimpan di:", scaler_path)
print("LabelEncoder di   :", le_path)
