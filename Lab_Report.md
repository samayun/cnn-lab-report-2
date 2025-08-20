# X-Ray Bone Fracture Classification using Pre-trained CNN Models
## Lab Report

**Course:** Machine Learning/Deep Learning Lab  
**Assignment:** Image Classification using Various CNN Models  
**Dataset:** X-ray Imaging Dataset for Detecting Fractured vs. Non-Fractured Bones  
**Date:** August 2025  

---

## 1. Introduction

This lab report presents a comprehensive study on bone fracture detection in X-ray images using five pre-trained Convolutional Neural Network (CNN) models. The objective is to develop an automated system that can accurately classify X-ray images as either showing fractured or non-fractured bones, which has significant applications in medical diagnosis and healthcare automation.

## 2. Dataset Description

### 2.1 Dataset Overview
The X-ray imaging dataset consists of two main directories:

**Original Dataset:**
- **Fractured Images:** 130 PNG images showing bone fractures
- **Non-Fractured Images:** 290 PNG images showing healthy bones
- **Total Original Images:** 420 images
- **Format:** PNG
- **Classes:** Binary classification (Fractured vs Non-Fractured)

**Augmented Dataset:**
- **Fractured Images:** ~4,650 JPG images (including augmented variants)
- **Non-Fractured Images:** ~4,650 JPG images (including augmented variants)  
- **Total Augmented Images:** 9,300 images
- **Format:** JPG
- **Classes:** Binary classification (Fractured vs Non-Fractured)

### 2.2 Dataset Characteristics
- **Image Type:** Grayscale X-ray medical images
- **Subject:** Various bone structures (arms, legs, etc.)
- **Quality:** Medical-grade X-ray images suitable for diagnostic purposes
- **Class Distribution:** Relatively balanced between fractured and non-fractured cases
- **Augmentation:** The augmented dataset includes rotations, translations, and other transformations

## 3. CNN Models Description

### 3.1 Model Architecture Overview
Five pre-trained CNN models were implemented and evaluated:

#### 3.1.1 InceptionV3
- **Architecture:** Deep CNN with inception modules using parallel convolutions
- **Parameters:** ~23.8M parameters
- **Key Features:** 
  - Efficient use of computational resources
  - Multiple filter sizes processed in parallel
  - Factorized convolutions for reduced computation
- **Input Size:** 224×224×3
- **Pre-training:** ImageNet dataset

#### 3.1.2 ResNet50
- **Architecture:** 50-layer residual network with skip connections
- **Parameters:** ~25.6M parameters
- **Key Features:**
  - Residual connections to prevent vanishing gradients
  - Deep architecture (50 layers)
  - Batch normalization throughout
- **Input Size:** 224×224×3
- **Pre-training:** ImageNet dataset

#### 3.1.3 VGG16
- **Architecture:** 16-layer CNN with small 3×3 convolution filters
- **Parameters:** ~138.4M parameters
- **Key Features:**
  - Simple and uniform architecture
  - Small receptive fields with deeper networks
  - Multiple fully connected layers
- **Input Size:** 224×224×3
- **Pre-training:** ImageNet dataset

#### 3.1.4 DenseNet121
- **Architecture:** 121-layer densely connected network
- **Parameters:** ~8.0M parameters
- **Key Features:**
  - Dense connectivity pattern
  - Feature reuse through concatenation
  - Parameter efficient design
- **Input Size:** 224×224×3
- **Pre-training:** ImageNet dataset

#### 3.1.5 MobileNetV2
- **Architecture:** Lightweight CNN using depthwise separable convolutions
- **Parameters:** ~3.5M parameters
- **Key Features:**
  - Inverted residual structure
  - Linear bottlenecks
  - Designed for mobile/embedded devices
- **Input Size:** 224×224×3
- **Pre-training:** ImageNet dataset

### 3.2 Transfer Learning Implementation
All models were implemented using transfer learning:
- **Base Models:** Pre-trained on ImageNet dataset
- **Feature Extraction:** Base model weights frozen during training
- **Custom Classifier:** Added on top of pre-trained features
- **Final Architecture:**
  ```
  Pre-trained Base Model (frozen)
  ↓
  Global Average Pooling 2D
  ↓
  Dropout (0.5)
  ↓
  Dense Layer (128 units, ReLU)
  ↓
  Dropout (0.3)
  ↓
  Output Layer (1 unit, Sigmoid)
  ```

## 4. Hyperparameter Setup

### 4.1 Training Configuration
- **Image Size:** 224×224 pixels
- **Batch Size:** 16 (optimized for fast execution)
- **Epochs:** 3 (reduced for quick training)
- **Optimizer:** Adam with learning rate 0.001
- **Loss Function:** Binary Crossentropy
- **Metrics:** Accuracy

### 4.2 Data Preprocessing
- **Rescaling:** Pixel values normalized to [0,1] range
- **Data Split:** 80% training, 20% testing
- **Validation Split:** 20% of training data for validation

### 4.3 Data Augmentation (Training Set)
- **Rotation:** ±20 degrees
- **Width Shift:** ±20%
- **Height Shift:** ±20%
- **Shear Range:** 0.2
- **Zoom Range:** 0.2
- **Horizontal Flip:** Enabled
- **Fill Mode:** Nearest neighbor

### 4.4 Callbacks and Regularization
- **Early Stopping:** Patience = 2 epochs
- **Learning Rate Reduction:** Factor = 0.5, Patience = 1 epoch
- **Dropout Layers:** 0.5 and 0.3 dropout rates
- **Best Weight Restoration:** Enabled

### 4.5 Fast Execution Settings
- **Subset Usage:** 0.2% of augmented dataset (~18 images per class)
- **Subset Usage:** 20% of original dataset (~26 fractured, 58 non-fractured)
- **Reduced Epochs:** 3 epochs for quick results
- **Small Batch Size:** 16 for memory efficiency

## 5. Evaluation Metrics

### 5.1 Primary Metrics
- **Accuracy:** Overall classification accuracy
- **Precision:** True Positives / (True Positives + False Positives)
- **Recall:** True Positives / (True Positives + False Negatives)
- **F1-Score:** Harmonic mean of Precision and Recall

### 5.2 Additional Metrics
- **Confusion Matrix:** Visual representation of classification results
- **Classification Report:** Detailed per-class metrics
- **Training Time:** Model training duration comparison
- **Training History:** Accuracy and loss curves over epochs

### 5.3 Evaluation Protocol
- **Cross-Validation:** Stratified train-test split
- **Test Set:** 20% of data, never seen during training
- **Reproducibility:** Fixed random seed (42) for consistent results

## 6. Expected Results Structure

### 6.1 Performance Comparison Tables
```
Model Performance Summary:
┌─────────────┬──────────┬──────────┬─────────────┐
│ Model       │ Accuracy │ Dataset  │ Subset Size │
├─────────────┼──────────┼──────────┼─────────────┤
│ InceptionV3 │ 0.XXXX   │ Original │ 20%         │
│ ResNet50    │ 0.XXXX   │ Original │ 20%         │
│ VGG16       │ 0.XXXX   │ Original │ 20%         │
│ DenseNet121 │ 0.XXXX   │ Original │ 20%         │
│ MobileNetV2 │ 0.XXXX   │ Original │ 20%         │
└─────────────┴──────────┴──────────┴─────────────┘
```

### 6.2 Visualization Outputs
- **Training History Plots:** Accuracy curves for all models
- **Confusion Matrices:** 2×2 matrices showing classification results
- **Comparison Charts:** Bar charts of model performances

## 7. Comparative Study: Original vs Augmented Dataset

### 7.1 Experiment Design
Two separate experiments were conducted:
1. **Original Dataset Experiment:** Using 20% of original 420 images
2. **Augmented Dataset Experiment:** Using 0.2% of augmented 9,300 images

### 7.2 Expected Findings
- **Data Augmentation Impact:** Comparison of model performance with/without augmentation
- **Training Stability:** Effect of dataset size on training consistency
- **Generalization:** How augmentation affects model generalization capability

## 8. Implementation Details

### 8.1 Software Environment
- **Framework:** TensorFlow/Keras
- **Language:** Python 3.x
- **Dependencies:** Listed in requirements.txt
  - scikit-learn
  - tensorflow
  - matplotlib
  - seaborn
  - pandas
  - numpy

### 8.2 Hardware Requirements
- **Minimum RAM:** 4GB
- **GPU:** Optional (CUDA-compatible for faster training)
- **Storage:** 2GB for dataset and models

### 8.3 Code Structure
```
cnn.py
├── XRayFractureClassifier (Main Class)
│   ├── __init__()           # Initialize parameters
│   ├── load_data()          # Load and subset data
│   ├── create_data_generators() # Data preprocessing
│   ├── create_model()       # Model architecture
│   ├── train_model()        # Training pipeline
│   ├── evaluate_model()     # Evaluation pipeline
│   ├── plot_training_history() # Visualization
│   ├── plot_confusion_matrices() # Confusion matrices
│   ├── create_comparison_report() # Results summary
│   └── run_experiment()     # Main execution
└── main()                   # Entry point
```

## 9. Expected Experimental Outcomes

### 9.1 Hypothesis
- **H1:** ResNet50 and InceptionV3 will show superior performance due to their advanced architectures
- **H2:** MobileNetV2 will provide competitive accuracy with fastest training time
- **H3:** Augmented dataset will improve model generalization
- **H4:** VGG16 might show overfitting due to its large parameter count

### 9.2 Success Criteria
- **Minimum Accuracy:** >80% on test set
- **Training Time:** <300 seconds per model
- **Reproducible Results:** Consistent performance across runs

## 10. Usage Instructions

### 10.1 Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Ensure dataset structure:
# Original Dataset/
#   ├── Fractured/
#   └── Non-Fractured/
# Augmented Dataset/
#   ├── Fractured/
#   └── Non-Fractured/
```

### 10.2 Execution
```bash
# Run the complete experiment
python cnn.py

# Expected runtime: 5-10 minutes
# Outputs: Training plots, confusion matrices, performance reports
```

### 10.3 Output Files
- `training_history_original.png` - Training curves for original dataset
- `training_history_augmented.png` - Training curves for augmented dataset
- `confusion_matrices_original.png` - Confusion matrices for original dataset
- `confusion_matrices_augmented.png` - Confusion matrices for augmented dataset
- Console output with detailed performance metrics

## 11. Limitations and Future Work

### 11.1 Current Limitations
- **Small Subset:** Using only 0.2% of augmented data for speed
- **Limited Epochs:** Only 3 epochs may not reach full convergence
- **Binary Classification:** Only two classes (could extend to fracture types)
- **No Clinical Validation:** Results not validated by medical professionals

### 11.2 Future Improvements
- **Ensemble Methods:** Combine multiple model predictions
- **Extended Training:** More epochs with learning rate scheduling
- **Medical Expert Validation:** Clinical assessment of results
- **Multi-class Extension:** Different types of fractures
- **Real-time Deployment:** Web application for medical use

## 12. Conclusion

This lab report presents a systematic approach to bone fracture detection using state-of-the-art pre-trained CNN models. The implementation provides a comprehensive comparison of five different architectures on both original and augmented X-ray datasets. The results will demonstrate the effectiveness of transfer learning for medical image classification and provide insights into the optimal model selection for fracture detection tasks.

The experimental design ensures reproducibility while maintaining computational efficiency, making it suitable for educational purposes and rapid prototyping in medical AI applications.

---

**Note:** This report serves as a template and framework. Actual results will be populated when the experiment is executed using the provided `cnn.py` implementation.