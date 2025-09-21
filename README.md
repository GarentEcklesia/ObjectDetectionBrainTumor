# ObjectDetectionBrainTumor
Projek ini merupakan Model Deployment YOLO11 untuk object detection yang dilatih dengan data Brain Tumor.

Proyek ini merupakan implementasi Computer Vision untuk bidang medis dengan menggunakan YOLOv11 (You Only Look Once).
Model ini dilatih pada dataset citra medis Brain Tumor MRI untuk mendeteksi dan mengklasifikasikan jenis tumor otak.

Aplikasi ini menampilkan:
1. Prediksi bounding box tumor pada gambar MRI.
2. Jenis tumor yang terdeteksi beserta tingkat confidence score.
3. Visualisasi interaktif berbasis Streamlit untuk upload gambar, video, maupun penggunaan contoh data.

⚠️ Disclaimer: Aplikasi ini hanya untuk keperluan penelitian/edukasi, bukan alat diagnosis medis. Untuk diagnosis yang valid, selalu konsultasikan dengan tenaga medis profesional.

Kelas Tumor yang Dideteksi
1. Glioma: Tumor ganas yang tumbuh dari jaringan pendukung (sel glial) di dalam otak. Cenderung menyusup ke jaringan normal di sekitarnya.
2. Meningioma: Tumor yang tumbuh pada selaput (meninges) yang melapisi otak. Umumnya bersifat jinak (bukan kanker) dan tumbuh lambat.
3. Pituitary Tumor: Pertumbuhan sel abnormal pada kelenjar pituitari, sebuah kelenjar di dasar otak yang mengatur hormon penting tubuh.
4. No Tumor: Menandakan bahwa model tidak mendeteksi adanya massa tumor signifikan pada citra medis.

Fitur Aplikasi
1. Upload Gambar: Mengunggah citra MRI otak untuk analisis.
2. Contoh Data: Menggunakan sampel gambar tumor (Glioma, Meningioma, Pituitary, No Tumor).
3. Upload Video: Mendukung analisis frame per frame untuk video.
4. Confidence Threshold: Atur ambang batas prediksi model.
5. Visualisasi Bounding Box: Hasil deteksi ditampilkan langsung di atas citra MRI.

Aplikasi ini sudah dideploy menggunakan Streamlit Cloud dan bisa diakses di sini: https://objectdetectionbraintumor-garent-ecklesia.streamlit.app/

Preview Website

<img width="1919" height="906" alt="image" src="https://github.com/user-attachments/assets/c06a617e-18e8-4bf8-90de-9c488643305e" />
