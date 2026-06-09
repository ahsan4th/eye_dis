import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms
import io

# Konfigurasi Halaman
st.set_page_config(page_title="Aplikasi Analisis & Augmentasi Citra", layout="wide")

st.title("Aplikasi Augmentasi & Prediksi Citra")
st.markdown("Aplikasi ini memungkinkan Anda untuk mengunggah bobot model PyTorch (`.pth`), mengunggah gambar, dan secara otomatis melakukan augmentasi (Flip & Rotate) untuk mengekstrak fitur gambar yang lebih baik sebelum diprediksi.")

# 1. Definisi Transformasi Gambar untuk Model
# Sesuaikan ukuran resolusi dengan yang Anda gunakan saat training (misal: 192x192 atau 224x224)
preprocess = transforms.Compose([
    transforms.Resize((192, 192)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 2. Fungsi untuk mendefinisikan & memuat model
@st.cache_resource
def load_model(uploaded_file):
    try:
        # Inisialisasi arsitektur ResNet34 (sesuai dengan basemodel yang Anda gunakan)
        model = models.resnet34(weights=None)
        
        # FastAI biasanya memodifikasi layer klasifikasi akhir. 
        # Kita menyesuaikan output layer ke 5 kelas penyakit mata.
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 5) 
        
        # Memuat bobot dari file .pth yang diunggah
        model.load_state_dict(torch.load(uploaded_file, map_location=torch.device('cpu')))
        model.eval()
        return model
    except Exception as e:
        st.error(f"Gagal memuat model. Pastikan arsitektur .pth sesuai. Error: {e}")
        return None

# 3. Fungsi Augmentasi Gambar (Sesuai dengan logika notebook)
def augment_image(img):
    augmented_images = {}
    
    # a. Gambar Asli
    augmented_images["Original"] = img
    
    # b. Flip Kiri-Kanan
    flipped_img = img.transpose(Image.FLIP_LEFT_RIGHT)
    augmented_images["Flipped (Left-Right)"] = flipped_img
    
    # c. Rotasi 180 Derajat
    rotated_img = img.rotate(180)
    augmented_images["Rotated (180°)"] = rotated_img
    
    return augmented_images

# 4. Fungsi Prediksi
def predict(model, img_tensor):
    classes = ['Glaucoma', 'Cataracts', 'Uveitis', 'Crossed_Eyes', 'Bulging eyes']
    with torch.no_grad():
        output = model(img_tensor.unsqueeze(0))
        _, predicted_idx = torch.max(output, 1)
        return classes[predicted_idx.item()]

# --- ANTARMUKA STREAMLIT ---

# Upload Model (.pth)
st.sidebar.header("1. Unggah Model")
model_file = st.sidebar.file_uploader("Pilih file model (.pth)", type=['pth'])

# Upload Gambar
st.sidebar.header("2. Unggah Gambar")
image_file = st.sidebar.file_uploader("Pilih gambar untuk dianalisis", type=['jpg', 'jpeg', 'png'])

if model_file is not None:
    st.sidebar.success("Model berhasil diunggah!")
    model = load_model(model_file)
else:
    st.info("Silakan unggah file model (.pth) pada panel di sebelah kiri untuk memulai.")

if image_file is not None:
    # Buka gambar yang diunggah
    img = Image.open(image_file).convert('RGB')
    
    st.subheader("Hasil Augmentasi Otomatis")
    # Lakukan augmentasi
    aug_dict = augment_image(img)
    
    # Tampilkan gambar hasil augmentasi secara berdampingan
    cols = st.columns(len(aug_dict))
    for idx, (label, augmented_img) in enumerate(aug_dict.items()):
        with cols[idx]:
            st.image(augmented_img, caption=label, use_container_width=True)
            
    # Lakukan Prediksi jika model sudah ada
    if model_file is not None and model is not None:
        st.markdown("---")
        st.subheader("Prediksi Model")
        
        # Tombol untuk memicu prediksi
        if st.button("Jalankan Prediksi pada Gambar"):
            with st.spinner("Memproses prediksi..."):
                # Prediksi dilakukan pada gambar aslinya (bisa disesuaikan jika ingin memprediksi hasil augmentasi)
                img_tensor = preprocess(img)
                prediction = predict(model, img_tensor)
                
                st.success(f"**Hasil Prediksi Klasifikasi:** {prediction}")