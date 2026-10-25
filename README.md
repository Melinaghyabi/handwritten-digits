# csci6201-final-project-handwritten-digits

## Name of all team members: 
Melina Ghyabi - Precious Ogundu

## Dataset description: 
Optical Recognition of Handwritten Digits Dataset <br> <br>
The dataset contains 1797 grayscale images representing handwritten digits from 0 to 9. Each image has an 8x8 pixel resolution, producing 64 numerical features per image. Pixel intensity values (brightness) range from 0 to 16. <br> <br>
**source:** https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html <br> <br>
**task type:** classification

## Research questions:
1. How accurately can a machine-learning model classify handwritten digits using pixel intensity values?
2. Which handwritten digits are most frequently confused by the classification model?
3. How does the performance of different classification algorithms compare when recognizing handwritten digits?

###  Run in order:
```powershelgl
python -m pip install -r requirements.txt
python handwritten_digits/01_load_digits.py
python handwritten_digits/02_explore_digits.py
python handwritten_digits/03_visualize_digits.py
python handwritten_digits/04_classify_digits.py
```

## Dataset exploring report:
The dataset was explored using Python, NumPy, and Scikit-learn. <br>
The dataset contains approximately 174-183 images per digit, which suggests that the ten digit classes are relatively balanced. Also no missing pixel value was found.<br>
|Characteristic     |Results|
|-------------------|-------|
|Number of Samples  |1797   |
|Number of Features |64     |
|Image Dimensions   |8 x 8  |
|Target Classes     |0 - 9  |
|Minimum Pixel Value|0      |
|Maximum Pixel Value|16     |
|Average Pixel Value|4.88   |
|Standard Deviation |6.02   |
|Missing Value      |0      |

## Data Visuaizations:
### Graph 1: Class Distribution 
This bar chart shows the number of images for each digit from 0 to 9. The counts are relatively similar, with digit 3 appearing most frequently (183 images) and digit 8 appearing least frequently (174 images)
![Number of images per digit](/Users/melinaghyabi/Desktop/barchart.png)
