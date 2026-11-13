# Brain Tumor Prediction Using Deep Learning
### Explainable MRI Classification, Tumor Analysis and Segmentation

**An academic research project exploring deep learning, transfer learning and explainable artificial intelligence for brain MRI analysis.**

---

## 1. Project Overview

Brain tumor analysis through Magnetic Resonance Imaging (MRI) plays an important role in understanding and examining abnormal brain tissue. However, distinguishing between tumor categories can be challenging because different tumor types may exhibit similar visual characteristics.

This project investigates a deep-learning-based framework for classifying brain MRI images into four categories: **Glioma, Meningioma, Pituitary and No Tumor**.

The research explores multiple Convolutional Neural Network (CNN) architectures, compares their classification performance, examines prediction behaviour through Gradient-weighted Class Activation Mapping (Grad-CAM), and investigates tumor-region segmentation using a 2D U-Net architecture.

Rather than focusing exclusively on classification accuracy, the project also considers model interpretability, dataset reliability, misclassification analysis and segmentation performance.

**Research focus:** Developing and evaluating an explainable deep-learning approach for brain MRI classification and tumor-region analysis.

> **Medical disclaimer:** This project is intended for academic and research purposes only. It is an experimental prototype and must not be used as a substitute for professional medical diagnosis or clinical decision-making.

---

## 2. Problem Statement

Manual interpretation of brain MRI scans requires medical expertise and can be challenging when different tumor categories have overlapping visual characteristics. Deep-learning models offer an opportunity to investigate automated image classification, but their performance must be assessed alongside data reliability, interpretability and segmentation quality.

This project addresses the problem through comparative CNN-based classification, visual explanation using Grad-CAM and tumor segmentation experiments using U-Net.

### Intended Users and Relevance

- **Students and researchers:** To explore deep learning techniques for medical image classification and segmentation.
- **Medical imaging research community:** To examine the potential of explainable AI in MRI analysis.
- **Developers:** To understand the integration of CNN models, visual explanations and segmentation workflows.

The system is a research-oriented decision-support exploration, not a clinically validated diagnostic tool.

---

## 3. Project Objectives

1. Develop a baseline CNN for four-class brain tumor classification.
2. Investigate transfer learning using MobileNetV2 and EfficientNetB0.
3. Fine-tune selected pretrained architectures and compare model performance.
4. Examine training and validation behaviour to identify overfitting and underfitting.
5. Analyse confusion matrices, class-wise metrics and misclassified MRI images.
6. Generate Grad-CAM visualizations to investigate regions influencing model predictions.
7. Implement a 2D U-Net architecture for tumor segmentation.
8. Evaluate segmentation performance using Dice Coefficient and Intersection over Union (IoU).
9. Investigate dataset reliability through duplicate checking and patient-level evaluation where reliable patient identifiers are available.

---

## 4. System Architecture and Methodology

The project follows a modular workflow consisting of data preparation, classification, explainability, segmentation and evaluation.

### Overall Workflow

```mermaid
<img width="555" height="285" alt="image" src="https://github.com/user-attachments/assets/cd8a1c6f-0390-494c-bc65-371b39c267c7" />

```

### Module 1: Brain Tumor Classification

Four CNN architectures were explored:

| Architecture | Role in the project |
|---|---|
| Custom CNN | Baseline classification model |
| MobileNetV2 | Transfer learning-based classification |
| EfficientNetB0 | Alternative pretrained architecture for performance comparison |
| Fine-tuned MobileNetV2 | Further adaptation of pretrained weights for the target dataset |

The classification workflow includes image preprocessing, model training, validation, fine-tuning and test evaluation.

The models were examined using accuracy, precision, recall, F1-score and confusion matrices.

### Module 2: Explainable AI Using Grad-CAM

Grad-CAM was incorporated to investigate which image regions influenced the classification predictions.

The explainability experiments included:

- Grad-CAM implementation across all four model architectures.
- Generation of 92 heatmap overlays using 23 MRI images.
- Examination of Pituitary-category Grad-CAM outputs.
- Visual inspection of 12 misclassified MRI images.
- Comparison of correctly and incorrectly classified Glioma images.

These visualizations help investigate model behaviour and potential attention patterns. They do not independently establish that a model has identified or localized the actual tumor.

### Module 3: Tumor Segmentation Using 2D U-Net

A 2D U-Net architecture was implemented to investigate tumor-region segmentation using MRI images and corresponding ground-truth masks.

The segmentation workflow includes:

- MRI and mask preprocessing.
- U-Net model development and training.
- Predicted mask generation.
- Comparison with ground-truth annotations.
- Evaluation using Dice Coefficient and IoU.
- Additional segmentation experiments and performance analysis.

### Module 4: Dataset Reliability and Evaluation

Data reliability was considered throughout the experimental workflow.

Activities included:

- Duplicate image checking.
- Training-only data augmentation.
- Dataset preprocessing and validation.
- Patient-level splitting for a separate dataset with available patient identifiers.
- Cross-dataset near-duplicate auditing.

The four-class dataset did not provide reliable patient identifiers. Therefore, its classification evaluation used image-level splitting. The separate three-class Figshare experiment used patient-level partitions.

---

## 5. Dataset Description

### 5.1 Four-Class Brain MRI Dataset

The primary classification experiments used approximately **7,023 MRI images** belonging to four categories.

| Category | Description |
|---|---|
| Glioma | MRI images associated with glioma tumors |
| Meningioma | MRI images associated with meningioma tumors |
| Pituitary | MRI images associated with pituitary tumors |
| No Tumor | MRI images without a tumor |

**Evaluation consideration:** Reliable patient identifiers were unavailable for this merged four-class dataset. Consequently, image-level splitting was used, and the results should be interpreted within this limitation.

### 5.2 Separate Patient-Level Figshare Experiment

A separate three-class dataset was used to conduct patient-independent evaluation.

| Partition | Patients | MRI slices |
|---|---:|---:|
| Training | 163 | 2,150 |
| Validation | 35 | 494 |
| Testing | 35 | 420 |
| **Total** | **233** | **3,064** |

The training, validation and test partitions contained no overlapping patients.

This was a separate experimental task and was not an external validation of the four-class classifier.

---

## 6. Experimental Results and Performance Evaluation

### 6.1 Classification Results

The Custom CNN was developed as a baseline, followed by transfer learning and fine-tuning experiments.

| Model | Test Accuracy |
|---|---:|
| Custom CNN | 74.22% |
| Fine-tuned MobileNetV2 | **95.91%** |

The fine-tuned MobileNetV2 achieved the strongest reported result among these highlighted experiments on the four-class test set.

**Fine-tuned MobileNetV2 – Test Evaluation**

| Evaluation Metric | Result |
|---|---:|
| Test Images | 1,346 |
| Correct Predictions | 1,291 |
| Test Accuracy | 95.91% |
| Macro F1-score | 95.86% |

### 6.2 Class-wise Classification Performance

| Tumor Category | F1-score |
|---|---:|
| Glioma | 95.02% |
| Meningioma | 92.17% |
| No Tumor | 98.41% |
| Pituitary | 97.84% |

Meningioma recorded the lowest class-wise F1-score among the four categories, indicating an area requiring further investigation.

### 6.3 Separate Patient-Level Evaluation

The separate three-class patient-level experiment achieved **62.86% test accuracy** on 420 MRI slices from 35 held-out patients.

This result belongs to a different dataset and classification task. It should not be directly compared with the four-class MobileNetV2 result.

### 6.4 Segmentation Results

The baseline 2D U-Net was evaluated using the segmentation test dataset.

| Metric | Test Result |
|---|---:|
| Dice Coefficient | 0.6647 |
| Intersection over Union | 0.5482 |

The results indicate that further investigation and refinement are needed to improve segmentation performance.

### 6.5 Explainability and Error Analysis

The Grad-CAM experiments generated 92 visual overlays across four model architectures using 23 MRI images.

Misclassification analysis included the inspection of 12 incorrectly classified MRI images and visual comparison of correctly and incorrectly classified Glioma images.

These investigations were used to understand potential prediction errors and assess the usefulness and limitations of visual explanations.

---

## 7. Technologies and Development Environment

| Area | Technology |
|---|---|
| Programming Language | Python |
| Deep Learning Framework | TensorFlow and Keras |
| Image Processing | OpenCV, PIL |
| Numerical Computation | NumPy |
| Data Analysis | Pandas |
| Visualization | Matplotlib, Seaborn |
| Explainable AI | Grad-CAM |
| Classification Models | Custom CNN, MobileNetV2, EfficientNetB0 |
| Segmentation Model | 2D U-Net |
| Development Environment | Google Colab |
| Version Control | Git and GitHub |

---

## 8. Implementation and Experimental Workflow

The implementation was organized into the following stages:

1. Dataset collection and organization.
2. Image preprocessing and validation.
3. Duplicate checking and data preparation.
4. Custom CNN baseline development and evaluation.
5. Transfer learning using MobileNetV2 and EfficientNetB0.
6. Fine-tuning and classification performance comparison.
7. Training and validation accuracy analysis.
8. Confusion matrix and misclassification investigation.
9. Grad-CAM implementation and visualization.
10. 2D U-Net segmentation implementation.
11. Dice and IoU evaluation.
12. Experimental result analysis and documentation.

The implementation was primarily conducted using Google Colab notebooks.

---

## 9. Key Findings

- Transfer learning and fine-tuning improved the reported classification performance compared with the Custom CNN baseline.
- Fine-tuned MobileNetV2 achieved 95.91% test accuracy on the four-class test dataset.
- Meningioma had the lowest class-wise F1-score, highlighting a need for further model refinement.
- Grad-CAM provided visual explanations of image regions influencing model predictions.
- Misclassification analysis helped identify cases where model predictions differed from ground-truth labels.
- The separate patient-level experiment emphasized the importance of patient-independent evaluation.
- The baseline U-Net results demonstrated the need for further segmentation improvements.

---

## 10. Limitations

The current experimental work has several limitations:

1. The primary four-class dataset did not provide reliable patient identifiers, preventing patient-level splitting for that classification task.
2. Independent external validation of the four-class classifier has not been completed.
3. Some tumor categories, particularly Meningioma, require further misclassification analysis.
4. The baseline U-Net segmentation performance requires improvement.
5. Grad-CAM visualizations provide model explanations but do not guarantee anatomically accurate localization.
6. Application-level integration and advanced MRI visualization remain under development.

These limitations are considered when interpreting the experimental results.

---

## 11. Future Scope

The next development phase will focus on:

- Independent external validation of the four-class classifier.
- Further classification refinement and error analysis.
- Improving U-Net segmentation performance.
- Developing a Streamlit-based application.
- Integrating classification, Grad-CAM and segmentation into a unified workflow.
- Investigating DICOM and NIfTI medical image support.
- Implementing slice navigation and 3D tumor visualization.
- Exploring probability calibration, uncertainty estimation and out-of-distribution detection.
- Strengthening input validation, testing and documentation.

---

## 12. Academic and Research Significance

This project demonstrates the application of deep learning techniques to medical image analysis through comparative CNN experiments, explainable AI and tumor segmentation.

It emphasizes that model evaluation should consider more than accuracy alone. Dataset reliability, class-wise performance, prediction errors, visual interpretability and segmentation quality are also important when investigating AI-based medical imaging systems.

The work contributes to academic exploration of explainable deep learning for brain MRI analysis and provides a foundation for further experimentation and application development.

---

**Developed as part of the Major Project at REVA University.**
