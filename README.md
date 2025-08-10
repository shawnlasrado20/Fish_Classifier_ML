# 🐟 Fish Species Classification

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A deep learning-based web application that classifies fish species from images.  
Built with **MobileNet** for top accuracy and deployed using **Streamlit** for easy access.  

---

## 📌 Project Overview
This project uses convolutional neural networks (CNNs) to classify fish species from images.  
The workflow involved:
- **Curating and preprocessing** a fish species dataset
- Applying **data augmentation** to improve generalization
- Experimenting with multiple CNN architectures
- Selecting and fine-tuning the **best-performing model**
- Deploying as a **Streamlit web app** for instant predictions

---

## 💡 What I Learned
- **Data preprocessing & augmentation** are crucial for building robust ML models.  
- **Model selection matters** — not all popular architectures perform equally well on every dataset.  
  - MobileNet achieved the best results  
  - ResNet50 and EfficientNetB0 underperformed despite fine-tuning  
- **Transfer learning** can greatly improve results.  
- **Side-by-side model evaluation** makes it easier to choose the best for deployment.  
- **Streamlit** is a simple yet powerful way to make AI models accessible.  

---

## 🏆 Final Model
- **Architecture:** MobileNet (fine-tuned)  
- **Test Accuracy:** ~99%  
- **Outperformed:** VGG16, ResNet50, InceptionV3, EfficientNetB0  

---

## 🚀 Run Locally

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/shawnlasrado20/Fish_Classifier_ML.git
cd Fish_Classifier_ML
