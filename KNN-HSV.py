import os
import cv2
import numpy as np
import pickle
import streamlit as st
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from matplotlib import pyplot as plt

# Konfigurasi Streamlit
st.title("Prediksi Kematangan Apel Menggunakan HSV dan KNN")
st.sidebar.header("Pengaturan")

# Input Path Dataset
#dataset_dir = st.sidebar.text_input("Path Dataset", value="Dataset/Apel")
dataset_dir = "Apel"
resize_dim = st.sidebar.slider("Dimensi Resize Gambar", 50, 200, 100, step=10)
k_neighbors = st.sidebar.slider("Jumlah K Tetangga (KNN)", 1, 10, 2)

# Fungsi untuk load dan resize image
@st.cache_resource

def load_images_from_folder(folder, label, size=(100, 100)):
    images = []
    labels = []
    if os.path.exists(folder):
        for filename in os.listdir(folder):
            img_path = os.path.join(folder, filename)
            img = cv2.imread(img_path)
            if img is not None:
                img_resized = cv2.resize(img, size)
                images.append(img_resized)
                labels.append(label)
    return images, labels

# Fungsi untuk ekstraksi fitur warna HSV
def extract_hsv_features(images):
    hsv_features = []
    for img in images:
        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h_mean = np.mean(hsv_img[:, :, 0])
        s_mean = np.mean(hsv_img[:, :, 1])
        v_mean = np.mean(hsv_img[:, :, 2])
        hsv_features.append([h_mean, s_mean, v_mean])
    return hsv_features

# Load dataset
#if st.sidebar.button("Load Dataset"):
st.write("**Memuat Dataset...**")
# Load data train
train_apel_matang_dir = os.path.join(dataset_dir, 'Train', 'Matang')
train_apel_tidak_matang_dir = os.path.join(dataset_dir, 'Train', 'Tidak Matang')
train_apel_matang, train_labels_matang = load_images_from_folder(train_apel_matang_dir, 1, (resize_dim, resize_dim))
train_apel_tidak_matang, train_labels_tidak_matang = load_images_from_folder(train_apel_tidak_matang_dir, 0, (resize_dim, resize_dim))
train_images = train_apel_matang + train_apel_tidak_matang
train_labels = train_labels_matang + train_labels_tidak_matang
# Load data test
test_apel_matang_dir = os.path.join(dataset_dir, 'Test', 'Matang')
test_apel_tidak_matang_dir = os.path.join(dataset_dir, 'Test', 'Tidak Matang')
test_apel_matang, test_labels_matang = load_images_from_folder(test_apel_matang_dir, 1, (resize_dim, resize_dim))
test_apel_tidak_matang, test_labels_tidak_matang = load_images_from_folder(test_apel_tidak_matang_dir, 0, (resize_dim, resize_dim))
test_images = test_apel_matang + test_apel_tidak_matang
test_labels = test_labels_matang + test_labels_tidak_matang
# st.write(f"**Jumlah Gambar Train:** {len(train_images)}")
# st.write(f"**Jumlah Gambar Test:** {len(test_images)}")
# Ekstraksi fitur warna HSV
train_hsv_features = extract_hsv_features(train_images)
test_hsv_features = extract_hsv_features(test_images)
# st.write("**Dataset Berhasil Dimuat!**")
# Latih Model KNN
knn = KNeighborsClassifier(n_neighbors=k_neighbors)
knn.fit(train_hsv_features, train_labels)
# Prediksi
test_predictions = knn.predict(test_hsv_features)
# # Evaluasi Model
# accuracy = accuracy_score(test_labels, test_predictions)
# cm = confusion_matrix(test_labels, test_predictions)
# st.write(f"**Akurasi Model:** {accuracy * 100:.2f}%")
# st.write("**Laporan Klasifikasi:**")
# st.text(classification_report(test_labels, test_predictions))
# # Tampilkan Confusion Matrix
# st.write("**Confusion Matrix:**")
# fig, ax = plt.subplots()
# ax.matshow(cm, cmap="Blues", alpha=0.7)
# for i in range(cm.shape[0]):
#     for j in range(cm.shape[1]):
#         ax.text(x=j, y=i, s=cm[i, j], va='center', ha='center')
# plt.xlabel("Predicted Labels")
# plt.ylabel("True Labels")
# st.pyplot(fig)
# # Tampilkan beberapa prediksi
# st.write("**Contoh Prediksi:**")
# for i in np.random.choice(len(test_images), min(5, len(test_images)), replace=False):
#     st.image(cv2.cvtColor(test_images[i], cv2.COLOR_BGR2RGB), caption=f"Predicted: {'Matang' if test_predictions[i] == 1 else 'Tidak Matang'}, True: {'Matang' if test_labels[i] == 1 else 'Tidak Matang'}")

# Fitur untuk Input Gambar Baru
st.write("**Klasifikasi Gambar Baru**")
uploaded_file = st.file_uploader("Unggah Gambar untuk Klasifikasi", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Membaca file yang diunggah
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img is not None:
        # Tampilkan gambar yang diunggah
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Gambar yang Diunggah", use_container_width=True)

        # Resize gambar sesuai dengan ukuran yang digunakan untuk pelatihan
        img_resized = cv2.resize(img, (resize_dim, resize_dim))

        # Ekstraksi fitur HSV dari gambar
        hsv_feature = extract_hsv_features([img_resized])

        # Prediksi menggunakan model KNN
        prediction = knn.predict(hsv_feature)[0]
        result = "Matang" if prediction == 1 else "Tidak Matang"

        # Tampilkan hasil prediksi
        st.write(f"**Hasil Prediksi:** {result}")
    else:
        st.error("Gambar tidak valid. Harap unggah file gambar yang benar.")
