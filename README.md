# CSCI6201-final-project-handwritten-digits
## Name of the team members:
Melina Ghyabi - Precious Ogundu

## Dataset Description:
**Dataset name:** Optical Recognition of Handwritten Digits (scikit-learn Digits) <br>
**Source:** scikit-learn <br>
**Dataset URL:** https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html <br>
**Task type:** classification

The dataset contains 1,797 grayscale images representing handwritten digits from 0 to 9. Each image has a resolution of 8×8 pixels, producing 64 numerical features per image. Pixel intensity values range from 0 to 16.

## Research Questions:

1. How accurately can machine-learning models classify handwritten digits using their pixel values, and how do KNN, SVM and CNN compare on the same test images?
2. Is the dataset balanced across the ten digit classes?
3. How are pixel intensity values distributed across handwritten digit images?

## Setup:
TensorFlow (used for the CNN) needs Python 3.10–3.12, so the project uses a Python 3.12 virtual environment:
``` powershell
py -3.12 -m venv .venv
.venv\Scripts\activate
python -m pip install scikit-learn numpy matplotlib tensorflow
```

## Run in order:
``` powershell
python 01_load_digits.py
python 02_explore_digits.py
python 03_visualize_digits.py
python 04_classify_digits.py
python 05_svm.py
python 06_cnn.py
```
All charts are saved to the `figures/` folder.

## Data exploring and analyze report:

The dataset was explored using Python, NumPy, and scikit-learn.<br>
The dataset contains approximately 174–183 images per digit, suggesting that the ten digit classes are relatively balanced. No missing pixel values were found.

| Characteristic      | Result |
|---------------------|--------|
| Number of samples   | 1,797  |
| Number of features  | 64     |
| Image dimensions    | 8 × 8  |
| Target classes      | 0–9    |
| Minimum pixel value | 0      |
| Maximum pixel value | 16     |
| Average pixel value | 4.88   |
| Standard deviation  | 6.02   |
| Missing values      | 0      |

## Data visualizations:

### Graph 1: Class distribution
This bar chart shows the number of images for each digit from 0 to 9. The counts are relatively similar, with digit 3 appearing most frequently (183 images) and digit 8 appearing least frequently (174 images).

![Number of images per digit](figures/class_distribution.png)

### Graph 2: Pixel intensity distribution
This histogram shows the frequency of pixel intensity values from 0 to 16. Pixel intensity 0 occurs most frequently, indicating that a large portion of the images consists of dark background pixels.

![Distribution of pixel brightness](figures/pixel_distribution.png)

## First basic model:

A k-Nearest Neighbors (KNN) classifier was used as the initial classification model.<br>
The dataset was divided into 80% training data and 20% testing data:

- Training samples: 1,437
- Testing samples: 360
- Number of neighbors (k): 3
- Random state: 42
- Stratified split: Yes

The model was trained using the training images and evaluated using previously unseen testing images.

**Preliminary accuracy: 98.61%**

The model correctly classified 355 of the 360 testing images and incorrectly classified 5 images.

This result suggests that the 64 pixel features provide useful information for identifying handwritten digits. However, additional evaluation is needed before drawing broader conclusions about model performance.

## Second model: Support Vector Machine (SVM)

An SVM was trained on the same 80/20 split as the KNN model (`05_svm.py`).

- The default SVM reached **99.17%** accuracy.
- The settings were then tuned with 5-fold cross-validation on the training data only (`GridSearchCV`), testing 12 combinations of `C` and `gamma`.
- Best settings: `C = 10`, `gamma = 0.001` (cross-validation accuracy: 98.96%).
- **Tuned SVM test accuracy: 99.17%**, with 3 incorrect predictions out of 360.

The SVM's three mistakes were a 6 read as an 8, an 8 read as a 1, and a 9 read as a 7.

![SVM confusion matrix](figures/svm_confusion_matrix.png)

![Images the SVM got wrong](figures/svm_misclassified.png)

## Third model: Convolutional Neural Network (CNN)

A CNN was built with TensorFlow/Keras (`06_cnn.py`). Unlike KNN and SVM, which treat each image as a flat list of 64 numbers, the CNN reads each image as an 8×8 grid, so it can learn shapes such as curves and lines.

- Input: each image reshaped to 8×8×1 and pixel values scaled to 0–1
- Layers: two convolution layers (32 and 64 filters), one max-pooling layer and two dense layers (85,066 trainable parameters)
- Training: 30 epochs, batch size 32, with 10% of the training images held back for validation
- Random seed: 42, so results can be reproduced

**CNN test accuracy: 97.50%**, with 9 incorrect predictions out of 360.

The training curve shows that training accuracy reached 100% by epoch 18, while validation accuracy stayed around 97–99%. This gap suggests mild overfitting: the CNN learned the training images perfectly but did not generalize better than the simpler models.

![CNN training curve](figures/cnn_training_curve.png)

The CNN's most common mistake was reading an 8 as a 1 (3 times).

![CNN confusion matrix](figures/cnn_confusion_matrix.png)

![Images the CNN got wrong](figures/cnn_misclassified.png)

## Model comparison

All three models were evaluated on the same 360 test images.

| Model | Accuracy | Macro F1 | Mistakes (of 360) |
|-------|----------|----------|-------------------|
| SVM (C=10, gamma=0.001) | 99.17% | 0.992 | 3 |
| KNN (k=3) | 98.61% | 0.986 | 5 |
| CNN | 97.50% | 0.975 | 9 |

![Mistakes per model](figures/model_comparison.png)

**What the results show:**

- All three models classify more than 97% of the test digits correctly, so the 64 pixel values carry enough information to recognize handwritten digits well.
- The SVM performed best on this test set, with only 3 mistakes.
- The CNN, the most complex model, performed worst. With tiny 8×8 images and only 1,437 training images, it overfit the training data. A CNN's advantage is more likely to appear with larger, more detailed images or more training data.
- The differences between models are small (3 vs. 9 mistakes, about 1.7 percentage points) and come from a single train/test split. A different split could change the ranking slightly, so we do not claim that one model is better in general.

**Limitations:** small 8×8 images, a single test split of 360 images, and the CNN was not tuned (for example, no dropout or early stopping).

## Progress Notes:
- Loaded the Handwritten Digits dataset.
- Examined the dataset structure and descriptive statistics.
- Checked for missing values.
- Created `figures/class_distribution.png`, which shows the number of images for each digit from 0 to 9.
- Created `figures/pixel_distribution.png`, which shows the frequency of pixel intensity values from 0 to 16
- Trained a basic KNN classifier.
- Evaluated the classifier using a separate testing set.
- Trained and tuned an SVM classifier with cross-validation (99.17% accuracy).
- Built and trained a CNN with TensorFlow/Keras (97.50% accuracy).
- Compared all three models on the same test images using accuracy, macro F1, confusion matrices and misclassified images.
- Moved all code to local Python scripts so every model runs in the same environment.

## Next steps for final project:

- **Working:** Data loading, exploratory analysis and visualizations, and three trained models: KNN (98.61%), SVM (99.17%) and CNN (97.50%), compared on the same test set.
- **Needs improvement:** The comparison uses a single train/test split. Cross-validation across all three models would show whether the ranking holds.
- **Planned:** Run cross-validation for all three models, try reducing CNN overfitting (dropout, early stopping), and write the final conclusion.
- **Instructor feedback requested:** Are the research questions focused enough, and does our evaluation avoid overclaiming?

## AI usage:
I used ChatGPT as a learning assistant. AI assistance included:
Explaining the dataset and data analysis techniques. Helping troubleshoot errors and GitHub troubleshooting. And also assisting with grammatical corrections and sentence polishing.

I also used Claude (Anthropic) as a learning assistant for choosing which models and metrics to compare, explaining how a CNN works, setting up the Python environment for TensorFlow, converting the CNN notebook to a Python script, and drafting this README update.
