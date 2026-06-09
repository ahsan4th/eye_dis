import streamlit as st
from PIL import Image
import torch
from torchvision.models import resnet50 # PERBAIKAN: Impor resnet50
from fastai.vision.learner import create_vision_model
from torchvision import transforms

# Konfigurasi Halaman Streamlit
st.set_page_config(page_title="Aplikasi Analisis & Augmentasi Citra", layout="wide")

st.title("Aplikasi Augmentasi & Klasifikasi Citra Mata")
st.markdown("Aplikasi ini secara otomatis melakukan augmentasi (Flip & Rotate) pada gambar yang diunggah dan melakukan prediksi menggunakan model FastAI (`.pth`).")

# 1. Preprocessing Gambar untuk Inferensi Model
preprocess = transforms.Compose([
    transforms.Resize((192, 192)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 2. Fungsi Memuat Model dengan Arsitektur FastAI
@st.cache_resource
def load_fastai_model(uploaded_file):
    try:
        # PERBAIKAN: Menggunakan resnet50 sesuai dengan bobot (weights) pada file .pth
        model = create_vision_model(resnet50, n_out=5, pretrained=False)
        
        # weights_only=False diperlukan untuk membaca objek FastAI (PyTorch 2.6+)
        state_dict = torch.load(uploaded_file, map_location=torch.device('cpu'), weights_only=False)
        
        # Penanganan jika state_dict tersimpan di dalam dictionary dengan kunci 'model'
        if isinstance(state_dict, dict) and 'model' in state_dict:
            state_dict = state_dict['model']
            
        model.load_state_dict(state_dict)
        model.eval()
        return model
    except Exception as e:
        st.error(f"Gagal memuat model. Pastikan file arsitektur sesuai. Error: {e}")
        return None

# 3. Fungsi Augmentasi Otomatis (Sesuai dengan notebook augment_images.ipynb)
def apply_augmentations(img):
    augmented_images = {
        "Original": img,
        "Flipped (Left-Right)": img.transpose(Image.FLIP_LEFT_RIGHT),
        "Rotated (180°)": img.rotate(180)
    }
    return augmented_images

# 4. Fungsi Prediksi
def predict_image(model, img_tensor):
    classes = ['Glaucoma', 'Cataracts', 'Uveitis', 'Crossed_Eyes', 'Bulging eyes']
    with torch.no_grad():
        output = model(img_tensor.unsqueeze(0))
        _, predicted_idx = torch.max(output, 1)
        return classes[predicted_idx.item()]

# --- ANTARMUKA UTAMA ---

# Sidebar Panel Pengunggahan
st.sidebar.header("Komponen Input")
model_file = st.sidebar.file_uploader("1. Unggah Model (.pth)", type=['pth'])
image_file = st.sidebar.file_uploader("2. Unggah Gambar", type=['jpg', 'jpeg', 'png'])

# Proses Pemuatan Model
model = None
if model_file is not None:
    model = load_fastai_model(model_file)
    if model is not None:
        st.sidebar.success("Model FastAI berhasil dimuat!")
else:
    st.info("Silakan unggah berkas model `.pth` di panel samping untuk mengaktifkan fitur prediksi.")

# Proses Gambar dan Augmentasi Otomatis
if image_file is not None:
    # Membuka gambar asli
    img = Image.open(image_file).convert('RGB')
    
    st.subheader("Visualisasi Augmentasi Otomatis")
    # Menjalankan fungsi augmentasi secara real-time
    dict_augmented = apply_augmentations(img)
    
    # Menampilkan hasil berdampingan ke dalam kolom
    cols = st.columns(len(dict_augmented))
    for idx, (label, aug_img) in enumerate(dict_augmented.items()):
        with cols[idx]:
            st.image(aug_img, caption=label, use_container_width=True)
            
    # Bagian Klasifikasi/Prediksi
    if model is not None:
        st.markdown("---")
        st.subheader("Analisis Klasifikasi Model")
        
        if st.button("Jalankan Prediksi"):
            with st.spinner("Model sedang menganalisis gambar..."):
                # Menyiapkan tensor gambar asli
                tensor_img = preprocess(img)
                hasil_prediksi = predict_image(model, tensor_img)
                
                st.metric(label="Hasil Diagnosa / Prediksi Kategori", value=hasil_prediksi)
