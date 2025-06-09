# Tau Jet Classification Using Convolutional Neural Networks

This repository contains the full workflow and codebase for the classification of hadronically decaying tau jets vs. light quark jets in high-energy physics experiments using jet images and Convolutional Neural Networks (CNNs).

---

## 📘 Project Overview

Tau lepton identification is crucial for precision measurements and beyond-Standard Model searches at particle colliders. This project investigates the use of CNN-based image classification to distinguish hadronically decaying tau jets from light quark jets using simulated event data.

We construct jet images from final-state particles in $e^+e^- \to \tau^+\tau^-$ and $e^+e^- \to jj$ processes, process the data through a complete image pipeline, and train/test CNN models across various center-of-mass energies.

---

## 🎯 Goals

- Represent jets as 2D images in the $(\eta, \phi)$ plane.
- Preprocess, normalize, and label simulated data for tau and quark jets.
- Train and evaluate CNN classifiers to distinguish between tau and light-quark jets.
- Test model generalization across different center-of-mass energies.

---

## 🧪 Dataset Details

- **Generated using**:  
  - MadGraph5_aMC@NLO for parton-level event generation  
  - Pythia8 for hadronization  
  - FastJet with anti-$k_T$ clustering (R=0.4)  
- **Jet image shape**: $32 \times 32$, grayscale  
- **Labeling**:  
  - Tau jet = `1`  
  - Quark jet = `0`  
- **Datasets**:  
  Simulated at $E_\text{cm} = 100$, $200$, and $300$ GeV with ~220k–240k jets per dataset.

- **Core jet images** are also created using a smaller radius (R=0.15) to highlight concentrated energy deposits.

---

## 🧠 CNN Architecture

Implemented using TensorFlow/Keras:
- **4 convolutional blocks**:  
  Each with Conv2D → BatchNorm → MaxPooling
- **Fully connected head**:  
  Flatten → Dense(128, ReLU) → Dropout(0.5) → Dense(1, Sigmoid)

Optimized with:
- **Loss**: Binary Crossentropy  
- **Metrics**: Accuracy, AUC, Precision, Recall  
- **Learning rate scheduler**: Exponential decay  
- **Class balancing**: Via `class_weight` during training

---

## 📊 Results Summary

- **Model 1 (Trained @ 100 GeV)**: High performance at 100 GeV, limited generalization to higher energies.
- **Model 2 (Trained @ 200 GeV)**: Strong generalization across all test datasets, AUC up to 99.78%.
- **Model 3 (Trained @ 300 GeV)**: Best performance overall, with consistent high efficiency and low mistag rate.

- Evaluation performed on test sets at 100, 150, 200, 250, 300, 350, and 400 GeV.

---
