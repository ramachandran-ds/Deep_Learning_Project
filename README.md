# Deep_Learning_Project
COMPANY        :   CODTECH IT SOLUTIONS

NAME           :   RAMACHANDRAN K

INTERN ID      :   CT04DH2787

DOMAIN         :   DATA SCIENCE

DURATION       :   4 WEEEKS

MENTOR         :   NEELA SANTOSH

# Improved Image Classification with TensorFlow (CIFAR-10)

## 📌 Project Overview
This project demonstrates the implementation of an **improved deep learning model** for **image classification** using the **CIFAR-10** dataset.  
The model is built using a **deep Convolutional Neural Network (CNN)** with enhancements such as **Data Augmentation**, **Batch Normalization**, and **Dropout Regularization** to improve both prediction accuracy and ability to generalize to unseen data.  

Designed in **TensorFlow 2.x** and optimized for **Google Colab**, the project takes advantage of GPU acceleration for faster training.  
The CIFAR-10 dataset contains **60,000 color images (32x32 pixels)** across **10 object categories**, each commonly used as a benchmark problem in computer vision.  

The object categories are:  
`airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck`.

---

## 🎯 Objectives
The objectives of this project are:
- Build a deep learning **CNN** that is more accurate than a basic CNN baseline.
- Apply **data augmentation** to increase dataset variety and reduce overfitting.
- Use **batch normalization** for stable training and faster convergence.
- Implement **dropout layers** for regularization.
- Visualize training accuracy, validation accuracy, and loss over time.
- Display model predictions and highlight correct vs incorrect classifications.

---

## 🛠 Key Features & Improvements
1. **Data Augmentation** adds random transformations during training:
   - Random horizontal flips
   - Random rotations
   - Random zoom
   This increases the robustness of the model.

2. **Deep CNN Architecture**:
   - Three convolutional blocks with increasing filters: 64 → 128 → 256.
   - Each block contains two convolutional layers followed by Batch Normalization and pooling.

3. **Batch Normalization**:
   - Normalizes activations after convolutional layers to stabilize training.

4. **Dropout Regularization**:
   - Dropout rates of 0.25–0.5 strategically placed to prevent overfitting.

5. **High-Capacity Dense Layer**:
   - A fully connected layer with 512 units before the output layer improves representation learning.

6. **Softmax Output**:
   - The final dense layer outputs probabilities for each of the 10 classes.

7. **Callbacks**:
   - **EarlyStopping** halts training when validation loss stops improving.
   - **ReduceLROnPlateau** lowers the learning rate when progress stalls.

---

## 📂 Dataset Details
- **Training set**: 50,000 images  
- **Test set**: 10,000 images  
- **Image size**: 32x32 pixels, 3 color channels (RGB)  
- **Number of classes**: 10 categories  

---

## 🏗 Model Architecture Summary
1. **Data Augmentation Layer** (horizontal flip, rotation, zoom)  
2. **Conv Block 1**:
   - Conv2D (64 filters)
   - BatchNorm
   - Conv2D (64 filters)
   - BatchNorm
   - MaxPooling2D
   - Dropout(0.25)
3. **Conv Block 2**:
   - Conv2D (128 filters)
   - BatchNorm
   - Conv2D (128 filters)
   - BatchNorm
   - MaxPooling2D
   - Dropout(0.25)
4. **Conv Block 3**:
   - Conv2D (256 filters)
   - BatchNorm
   - Conv2D (256 filters)
   - BatchNorm
   - MaxPooling2D
   - Dropout(0.4)
5. **Dense Layers**:
   - Flatten
   - Dense(512, ReLU)
   - BatchNorm
   - Dropout(0.5)
   - Dense(10, Softmax)

---

## 🚀 Running the Project in Google Colab
1. Open [Google Colab](https://colab.research.google.com/).
2. Create a new notebook and select:
   - `Runtime > Change runtime type > Hardware Accelerator > GPU`
3. Copy the code from this repository into the notebook.
4. Run the cells sequentially to:
   - Load and preprocess the data
   - Define and compile the CNN
   - Train the model with callbacks
   - Save the trained model file (`.h5`)
   - Plot accuracy/loss curves
   - Display prediction samples

---

## 📊 Visualizations Included
- **Training and Validation Accuracy** vs. Epoch  
- **Training and Validation Loss** vs. Epoch  
- Prediction samples showing:
  - True label
  - Predicted label
  - Green label if correct; red if incorrect

---

## 📈 Expected Performance
With the enhancements described, this model typically achieves:
- **Training Accuracy**: 90–95%  
- **Validation Accuracy**: ~80–85% after ~30 epochs on a GPU.  
Final accuracy may differ slightly depending on GPU type and random initialization.

---

## 📜 License
This project is released under the MIT License — you are free to use, modify, and share with attribution.

---

## 🙌 Acknowledgements
- **TensorFlow** for the deep learning framework.
- **CIFAR-10 dataset creators** at the University of Toronto.
- **Google Colab** for free GPU computing resources.
- The open-source community for advancing deep learning research.

---
