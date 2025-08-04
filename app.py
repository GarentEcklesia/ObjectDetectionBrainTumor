import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageOps
import os
import sys
import io
import tempfile 

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError as e:
    ULTRALYTICS_AVAILABLE = False
    st.error(f"Ultralytics import error: {e}")

st.set_page_config(
    page_title="Brain Tumor Detection & Visualization",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 1rem;
        border-radius: 10px;
        margin-top: 1rem;
    }
    .glioma { background-color: #ffebee; border-left: 5px solid #f44336; }
    .meningioma { background-color: #e8f5e8; border-left: 5px solid #4caf50; }
    .no-tumor { background-color: #e3f2fd; border-left: 5px solid #2196f3; }
    .pituitary { background-color: #fff3e0; border-left: 5px solid #ff9800; }
    .stButton>button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)



@st.cache_resource
def load_model(model_path):
    """Memuat model YOLO dengan penanganan error."""
    try:
        if not ULTRALYTICS_AVAILABLE:
            raise ImportError("Library 'ultralytics' tidak tersedia.")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"File model tidak ditemukan: {model_path}")
        model = YOLO(model_path)
        return model, None
    except Exception as e:
        return None, str(e)

def load_example_images():
    """Memuat gambar contoh dari file lokal."""
    example_images = {}
    image_files = {"Glioma": "glioma.jpg", "Meningioma": "meningioma.jpg", "Pituitary": "pituitary.jpg", "No Tumor": "no tumor.jpg"}
    for name, filename in image_files.items():
        if os.path.exists(filename):
            example_images[name] = Image.open(filename)
        else:
            st.sidebar.warning(f"File contoh tidak ditemukan: {filename}")
    return example_images

def resize_for_display(image, size=(256, 256)):
    """
    Mengubah ukuran gambar agar seragam untuk ditampilkan di galeri.
    Menjaga aspek rasio dan menambahkan padding hitam.
    """
    return ImageOps.pad(image, size, color='black')

def extract_frame_from_video(video_file, frame_number):
    """
    Mengekstrak frame dari file video dengan menyimpannya ke file temporer.
    video_file adalah objek UploadedFile dari Streamlit.
    """
    video_path = ""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
            # Tulis byte dari video yang diunggah ke file temporer
            tmp.write(video_file.getvalue())
            video_path = tmp.name
            
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            os.unlink(video_path) # Hapus file temporer jika gagal dibuka
            return None, 0, "Gagal membuka file video."

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0:
            cap.release()
            os.unlink(video_path)
            return None, 0, "Video tidak memiliki frame."

        cap.set(cv2.CAP_PROP_POS_FRAMES, min(frame_number, total_frames - 1))
        ret, frame = cap.read()
        cap.release()
        
        os.unlink(video_path)

        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return Image.fromarray(frame_rgb), total_frames, None
        else:
            return None, total_frames, "Gagal mengekstrak frame yang dipilih."

    except Exception as e:
        # Jika terjadi error, pastikan file temporer tetap dihapus jika ada
        if video_path and os.path.exists(video_path):
            os.unlink(video_path)
        return None, 0, f"Error saat memproses video: {str(e)}"

def get_prediction_style(class_name):
    """Mendapatkan kelas CSS berdasarkan nama kelas deteksi."""
    return {'Glioma': 'glioma', 'Meningioma': 'meningioma', 'No Tumor': 'no-tumor', 'Pituitary': 'pituitary'}.get(class_name, 'no-tumor')

def clear_results():
    """Membersihkan hasil deteksi sebelumnya dari session state."""
    for key in ['result_image', 'detections', 'selected_image']:
        if key in st.session_state:
            del st.session_state[key]

def main():
    st.markdown('<h1 class="main-header">Brain Tumor Detection & Visualization</h1>', unsafe_allow_html=True)

    st.sidebar.header("⚙️ Model Configuration")
    model_path = st.sidebar.text_input("Model Path", value="best.pt", help="Path ke file model YOLOv8 Anda (.pt)")
    confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.5, 0.05)
    class_names = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']

    st.sidebar.divider() 
    
    st.sidebar.subheader("ℹ️ Deskripsi Kelas Tumor") 
    st.sidebar.markdown("""
    - **Glioma:** Tumor ganas yang tumbuh dari jaringan pendukung (sel glial) di dalam otak. Cenderung menyusup ke jaringan normal di sekitarnya.

    - **Meningioma:** Tumor yang tumbuh pada selaput (*meninges*) yang melapisi otak. Umumnya bersifat jinak (bukan kanker) dan tumbuh lambat.

    - **Pituitary Tumor:** Pertumbuhan sel abnormal pada kelenjar pituitari, sebuah kelenjar di dasar otak yang mengatur hormon penting tubuh.
        
    - **No Tumor:** Menandakan bahwa model AI tidak mendeteksi adanya massa tumor yang signifikan pada citra medis yang dianalisis.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.header("1. Choose Input Source")
        
        image_to_process = None

        tab1, tab2, tab3 = st.tabs(["📁 Upload Image", "🖼️ Example Images", "🎥 Upload Video"])

        with tab1:
            uploaded_file = st.file_uploader("Choose a brain scan image...", type=['png', 'jpg', 'jpeg'])
            if uploaded_file:
                image_to_process = Image.open(uploaded_file)
                st.image(image_to_process, caption="Uploaded Image.")
                if st.button("Analyze Uploaded Image", key="analyze_upload"):
                    clear_results()
                    st.session_state.selected_image = image_to_process

        with tab2:
            st.write("Click on any sample button to use it for prediction:")
            example_images = load_example_images()
            if example_images:
                image_types = ["Glioma", "Meningioma", "Pituitary", "No Tumor"]
                example_cols = st.columns(4)
                for i, col in enumerate(example_cols):
                    with col:
                        tumor_type = image_types[i]
                        if tumor_type in example_images:
                            display_img = resize_for_display(example_images[tumor_type])
                            st.image(display_img, caption=f"{tumor_type} Sample")
                            if st.button(f"Use {tumor_type}", key=f"btn_{tumor_type}"):
                                clear_results()
                                st.session_state.selected_image = example_images[tumor_type]

        with tab3:
            uploaded_video = st.file_uploader("Choose a brain scan video...", type=['mp4', 'mov', 'avi'])
            if uploaded_video:
                _, total_frames, error = extract_frame_from_video(uploaded_video, 0)
                
                if error:
                    st.error(error)
                elif total_frames > 0:
                    frame_number = st.slider("Select frame to analyze", 0, total_frames - 1, total_frames // 2)
                    
                    image_to_process, _, _ = extract_frame_from_video(uploaded_video, frame_number)
                    
                    if image_to_process:
                        st.image(image_to_process, caption=f"Selected Frame: {frame_number}", use_column_width=True)
                        if st.button("Analyze Video Frame", key="analyze_video"):
                            clear_results()
                            st.session_state.selected_image = image_to_process
                else:
                    st.warning("Could not read frames from the uploaded video.")

    if 'selected_image' in st.session_state and 'result_image' not in st.session_state:
        with st.spinner("Analyzing image... Please wait."):
            model, model_error = load_model(model_path)
            if model_error:
                st.error(f"Failed to load model: {model_error}")
            else:
                results = model(st.session_state.selected_image, conf=confidence_threshold)
                result = results[0]
                plotted_image = cv2.cvtColor(result.plot(), cv2.COLOR_BGR2RGB)
                st.session_state.result_image = plotted_image
                
                detections = []
                for box in result.boxes:
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    class_name = class_names[class_id] if class_id < len(class_names) else f"Unknown ({class_id})"
                    detections.append({'class': class_name, 'confidence': confidence})
                st.session_state.detections = detections
                st.rerun()

    with col2:
        st.header("2. Detection Result")
        if 'result_image' in st.session_state:
            st.image(st.session_state.result_image, caption="Image with Detections")
            
            detections = st.session_state.get('detections', [])
            if detections:
                st.success(f"Found {len(detections)} detection(s).")
                detections_sorted = sorted(detections, key=lambda x: x['confidence'], reverse=True)
                for detection in detections_sorted:
                    style_class = get_prediction_style(detection['class'])
                    st.markdown(f"""
                    <div class="prediction-box {style_class}">
                        <p><strong>Class:</strong> {detection['class']} | <strong>Confidence:</strong> {detection['confidence']:.2%}</p>
                    </div>""", unsafe_allow_html=True)
                if any(d['class'] != 'No Tumor' for d in detections):
                    st.info("Please consult with a medical professional for a proper diagnosis.")
            else:
                st.info("No tumor was detected based on the current confidence threshold.")
        else:
            st.info("The analysis result will be displayed here.")

    st.markdown("---")
    st.markdown("**Disclaimer:** This tool is for educational purposes only and is not a substitute for professional medical diagnosis.")

if __name__ == "__main__":
    main()
