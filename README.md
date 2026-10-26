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

1. How accurately can a basic machine-learning model classify handwritten digits using their pixel values?
2. Is the dataset balanced across the ten digit classes?
3. How are pixel intensity values distributed across handwritten digit images?

## Run in order:
``` powershell
python3 -m pip install scikit-learn numpy matplotlib
python3 01_load_digits.py
python3 analysis.py
python3 03_visualize_digits.py
python3 04_classify_digits.py
```

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

## Progress Notes:
- Loaded the Handwritten Digits dataset.
- Examined the dataset structure and descriptive statistics.
- Checked for missing values.
- Created `figures/class_distribution.png`, which shows the number of images for each digit from 0 to 9.
- Created `figures/pixel_distribution.png`, which shows the frequency of pixel intensity values from 0 to 16
- Trained a basic KNN classifier.
- Evaluated the classifier using a separate testing set.

## Next steps for final project:

- **Working:** Data loading, exploratory analysis and visualizations, the initial KNN classifier achieves 98.61% accuracy on the test set.
- **Needs improvement:** Further evaluation is needed to understand which handwritten digits are most frequently misclassified and why.
- **Planned:** Investigate the model's mistakes, compare KNN with other classification methods.
- **Instructor feedback requested:** Are the research questions focused enough, and does our evaluation avoid overclaiming?

## AI usage:
I used ChatGPT as a learning assistant. AI assistance included:
Explaining the dataset and data analysis techniques. Helping troubleshoot errors and GitHub troubleshooting. And also assisting with grammatical corrections and sentence polishing.
