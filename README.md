# Gender Detection Using Machine Learning

An image classification project using a Convolutional Neural Network (CNN) built with TensorFlow and Keras. The project classifies facial images into the two labels available in the dataset: Man and Woman.

## Project Overview

This project demonstrates an end-to-end image classification workflow, including:

- Image dataset collection and validation
- Exploratory Data Analysis (EDA)
- Duplicate image detection and removal
- Image preprocessing and normalization
- Stratified training, validation, and testing split
- CNN model development
- Data augmentation and Batch Normalization
- Model evaluation and performance comparison
- Streamlit web application development

## Model Performance

The final Improved CNN was evaluated on 346 unseen test images.

| Metric | Score |
|---|---:|
| Accuracy | 82.08% |
| Precision | 82.14% |
| Recall | 81.18% |
| F1 Score | 81.66% |
| ROC-AUC | 88.78% |

## Technologies Used

- Python
- TensorFlow
- Keras
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- Streamlit
- Jupyter Notebook
- PyCharm

## Project Structure

```text
PRAICP-1001-GenderDetc/
│
├── models/
│   ├── gender_detection_final.keras
│   ├── label_mapping.json
│   └── gender_detection_final_metrics.csv
│
├── Gender_Detection_Using_ML.ipynb
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
```

Navigate to the project directory:

```bash
cd PRAICP-1001-GenderDetc
```

Install the dependencies:

```bash
python -m pip install -r requirements.txt
```

## Run the Streamlit Application

```bash
python -m streamlit run app.py
```

Open the local application in your browser:

```text
http://localhost:8501
```

Upload an image to see the model's predicted dataset label and probability scores.

## Model Details

- Architecture: Convolutional Neural Network (CNN)
- Input image size: 160 × 160 × 3
- Output layer: Sigmoid activation
- Loss function: Binary Crossentropy
- Optimizer: Adam
- Output classes: Man and Woman

## Limitations and Disclaimer

This project is an educational machine learning demonstration. The model predicts binary labels present in its training dataset; it does not determine a person's gender identity.

The dataset is limited, and model predictions may be inaccurate. This model should not be used for identifying, profiling, or making decisions about real people.

## Author

Developed as a Data Science and Machine Learning project.