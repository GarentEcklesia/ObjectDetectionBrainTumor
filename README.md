# 🧠 Object Detection - Brain Tumor

This is a Computer Vision project for the medical field using **YOLOv8**. The model is trained on a Brain Tumor MRI medical image dataset to detect and classify types of brain tumors in *real-time*.

This application displays the predicted *bounding box* of the tumor on an MRI image, the detected tumor type, and its *confidence score*.

## 🚀 Live Demo

This application has been deployed using Streamlit Cloud and can be accessed here:

[**➡️ Click here to launch the Streamlit App**](https://objectdetectionbraintumor-garent-ecklesia.streamlit.app/)

## 💡 Application Features

* **Image Upload:** Upload brain MRI images for analysis.
* **Sample Data:** Use sample tumor images (Glioma, Meningioma, Pituitary, No Tumor).
* **Video Upload:** Supports frame-by-frame analysis for videos (up to 30 seconds).
* **Real-time Webcam:** Analyze using a live camera feed.
* **Confidence Threshold:** Allows the user to set the model's prediction confidence threshold.
* **Bounding Box Visualization:** Detection results are displayed directly on the MRI image.
* **Downloadable Output:** Detection results can be downloaded as an image (.jpg) and a CSV file.

## 🛠️ How to Run Locally

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/GarentEcklesia/ObjectDetectionBrainTumor
    cd ObjectDetectionBrainTumor
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the Streamlit app:**
    ```bash
    streamlit run app.py
    ```

## 🎯 Detected Tumor Classes

* **Glioma:** A malignant tumor that grows from the supportive tissue (glial cells) within the brain.
* **Meningioma:** A tumor that grows on the membranes (meninges) that line the brain (generally benign).
* **Pituitary Tumor:** An abnormal cell growth in the pituitary gland at the base of the brain.
* **No Tumor:** Indicates that the model did not detect any significant tumor mass.

## ⚙️ Tech Stack

* **Model Architecture:** YOLOv8 (Ultralytics)
* **Web Framework:** Streamlit
* **Image & Video Processing:** OpenCV, Pillow (PIL)
* **Data Handling & Analysis:** NumPy, Pandas
* **Deployment Platform:** Streamlit Cloud
* **Model Format:** PyTorch (.pt)

## 🧠 Model Details

* **Model:** YOLOv8 (custom trained on brain tumor MRI dataset with bounding boxes).
* **Dataset:** [Brain Tumor MRI Dataset (with Bounding Boxes)](https://www.kaggle.com/datasets/ahmedsorour1/mri-for-brain-tumor-with-bounding-boxes)
* **Metrics:**
    * **mAP@0.5:** ~ 95.8%
    * **mAP@0.5-0.95:** ~ 79.2%
    * **Recall:** ~ 91.9%

## 📬 Contact

Garent Ecklesia - [garentecklesia45678@gmail.com](mailto:garentecklesia45678@gmail.com)

## 📝 License
This project is open-source and free to use for educational and research purposes.
