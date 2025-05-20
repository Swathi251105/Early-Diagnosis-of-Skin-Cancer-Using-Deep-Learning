# Early-Diagnosis-of-Skin-Cancer-Using-Deep-Learning

📌 Project Overview
This project aims to assist in the early detection of skin cancer using dermoscopic images and deep learning (CNN-based) techniques. The system performs image preprocessing, lesion segmentation, and classification of skin lesions as benign or malignant, making it a valuable tool in medical diagnostics.

🧠 Abstract
Skin cancer, particularly melanoma, poses a severe health risk due to its rapid progression and high mortality rate. Using deep learning, especially CNNs like U-Net with a VGG16 encoder, this project automates lesion segmentation and cancer classification with high accuracy. It aims to be efficient, deployable in resource-constrained environments, and usable through both desktop and mobile platforms.

⚙️ Technologies Used
Programming Language: Python

Frameworks: TensorFlow, Keras, PyTorch

Tools: Anaconda, Spyder, Jupyter Notebook

Libraries: OpenCV, NumPy, Pandas, Matplotlib, Scikit-learn

Dataset: ISIC Skin Cancer Dataset

🖥️ System Requirements
Hardware
Processor: Intel i5 2.4 GHz or higher

RAM: 4 GB or more

Storage: 500 GB HDD

Internet: Required for dataset download and updates

Software
Operating System: Windows 11 or Linux (Ubuntu)

Python Version: 3.8+

IDE: Spyder (via Anaconda)

🔧 Setup and Execution Guide
📥 Step 1: Install Anaconda
Download Anaconda:
https://www.anaconda.com/products/distribution

Run the installer and complete the installation with default settings.

🚀 Step 2: Launch Spyder
Open Anaconda Navigator.

Click Launch under Spyder to open the IDE.

🗂️ Step 3: Prepare Your Project Files
Place app.py and related files (model, images, utils) in one folder.

Ensure all .py files and image folders are in the same working directory.

📦 Step 4: Install Required Libraries
In the Spyder IPython Console or Anaconda Prompt, run:

bash
Copy
Edit
pip install tensorflow keras opencv-python numpy pandas matplotlib scikit-learn
📁 Step 5: Download ISIC Dataset
Visit: https://www.isic-archive.com

Go to "Download" section.

Choose 2018 ISIC Challenge Dataset or HAM10000.

Download and extract the dataset into a folder named dataset/ inside your project directory.

▶️ Step 6: Run app.py in Spyder
Open Spyder.

Set the Current Working Directory to your project folder.

Open app.py via File → Open.

Press F5 or click the green Run button.

If app.py starts a Flask server, go to the link in the console output, usually:

cpp
Copy
Edit
http://127.0.0.1:5000/
💡 Usage
Launch the application as described above.

A web interface will open (if app.py is a Flask app).

Upload a dermoscopic image (JPEG/PNG).

The system will:

Preprocess the image

Segment the lesion using U-Net

Classify it as benign or malignant

Display confidence scores and visual overlays (if enabled)

Results can be saved or reviewed by medical professionals.

📊 Evaluation Metrics
Accuracy

Precision

Recall

F1 Score

Dice Coefficient

Jaccard Index

💡 Features
Automated skin lesion segmentation

Classification into malignant/benign

Visual heatmap for explainability

Can run without GPU

Modular structure with support for web/mobile integration

🔒 License
This project is for academic and research purposes only.
