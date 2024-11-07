# Brain Tumor Detection using Convolutional Neural Networks (CNNs)

## Overview

This project aims to detect brain tumors from medical images using Convolutional Neural Networks (CNNs). The CNN model is trained on a dataset consisting of images labeled as either containing a tumor or not. The trained model is then used to predict tumor presence in unseen MRI images.

## Project Structure

### Code:
The code is written in Python and utilizes Keras with a TensorFlow backend to build and train the CNN model. It includes the following scripts:
- **Data Preprocessing**: Prepares the data for training by resizing and normalizing the images.
- **Model Building**: Defines the architecture of the CNN model for tumor detection.
- **Training**: Trains the model on the training dataset.
- **Evaluation**: Evaluates the performance of the model on a separate test dataset.
- **Prediction**: Uses the trained model to predict tumor presence in new MRI images.

### Data:
The dataset consists of MRI images stored in folders:
- **yes/**: Images containing a tumor.
- **no/**: Images without a tumor.
- **pred/**: MRI images for prediction.

### Results:
After training the model, performance metrics such as accuracy, loss, precision, recall, and F1-score are evaluated. The model then makes predictions on new MRI images.

## Model Architecture

The CNN model architecture consists of several convolutional layers followed by max-pooling layers, batch normalization, dropout, and dense layers for classification. The model is compiled with the Adam optimizer and binary cross-entropy loss function. The output layer uses a sigmoid activation function for binary classification (tumor/no tumor).

## Training

- **Batch size**: 32
- **Epochs**: 15
- **Optimizer**: Adam
- **Loss function**: Binary Cross-Entropy

The model is trained to classify images as containing a tumor or not, using the training data, and then evaluated on a separate test dataset.

## Dependencies

The following Python libraries are required:
- **tensorflow**: For building and training the CNN model.
- **numpy**: For numerical operations.
- **pandas**: For data manipulation.
- **matplotlib**: For plotting training graphs and evaluating results.
- **scikit-learn**: For additional evaluation metrics like precision, recall, and F1-score.
- **Pillow**: For image preprocessing.

### requirements.txt
tensorflow==2.18.0 
numpy==1.23.5 
pandas==1.5.3 
scikit-learn==1.2.2 
matplotlib==3.7.1 
Pillow==9.3.0

## How to Run the Code

### 1. Clone the repository:
```bash
git clone https://github.com/yourusername/brain-tumor-detection.git
```

### 2.Install dependencies:
```bash
pip install -r requirements.txt
```

### 3. Place the MRI images in appropriate folders:
Ensure your dataset is structured as follows

```bash
datasets/
  yes/
    image1.jpg
    image2.jpg
    ...
  no/
    image1.jpg
    image2.jpg
    ...
  pred/
    image1.jpg
    image2.jpg
    ...
```

### 4.Run the script:
```bash
python model.py
```
### This will:

1. Preprocess the images.
2. Train the CNN model.
3. Evaluate its performance.
4. Make predictions on new MRI images.

### Evaluation Metrics
After training, the model's performance is evaluated using:

1. Accuracy: The percentage of correctly classified images.
2. Loss: The error between the predicted and true labels.
3. Precision, Recall, F1-score: Additional metrics to assess the model's effectiveness.

### Contributors
Shivam Singh: Project Author
