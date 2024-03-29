### Segmentation Assisted COVID 19 Diagnosis

### Summary

The project is about segmenting the area of infection automatically, and use segmentation results to assist the diagnostic network in identifying the COVID19 samples from the CT images. Besides, a radiologist‑like segmentation network provides detailed information about the infectious regions by separating areas of ground‑glass, consolidation, and pleural effusion.

### Dataset

Here you can see a sample image from the dataset:

<!-- ![sample image from dataset](./readme_images/sample.jpg) -->
<img src="./readme_images/sample.jpg" width="300" height="300">

And here is a sample CT image and its mask from the dataset:

<!-- ![sample CT image and its mask from the dataset](./readme_images/sample-mask.jpg) -->
<img src="./readme_images/sample-mask.jpg" width="600" height="300">

### Explanation

The project consists of two parts:
1. Segmentation task
2. Diagnosis Task

#### Segmentation task:
The dataset for this task is available here:
[Dataset](https://drive.google.com/drive/folders/1LSgzWgiDrNdlXfBmZFl1LyrFVWFaUA_q?usp=share_link)

We have three types of masks:
1. ground glass
2. consolidation
3. pleural effusion

In order to do the segmentation task, three U-Net models are trained, and they are available here:
[Trained Models](https://drive.google.com/drive/folders/1ubOYddgXB_DkUQwLnlASKzLqA0vo4P1q?usp=share_link)

#### Diagnosis task:
The dataset for this task is available here:
[Dataset](https://drive.google.com/drive/folders/1ubOYddgXB_DkUQwLnlASKzLqA0vo4P1q?usp=share_link)

In order to do the diagnosis task, a VGG-16 model is used and trained. It is available here:
[Trained Model](https://drive.google.com/drive/folders/1ubOYddgXB_DkUQwLnlASKzLqA0vo4P1q?usp=share_link)

#### How to use:
1. Just simply open the code
2. Correct the links to the dataset and models
3. Run the code

#### Results

Results of the segmentation task:
<!-- ![seg task result](./readme_images/mask-result.png) -->
<img src="./readme_images/mask-result.png" width="300" height="600">

Result of the diagnosis task:

<!-- ![dia figure](./readme_images/figure.png) -->
<img src="./readme_images/figure.png" width="500" height="600">

It reached 92% accuracy for the test data.
Final results:
<!-- ![Final result](./readme_images/final.png) -->
<img src="./readme_images/final.png" width="500" height="500">


### Reference

Yao, Hy., Wan, Wg. & Li, X. A deep adversarial model for segmentation-assisted COVID-19 diagnosis using CT images. EURASIP J. Adv. Signal Process. 2022, 10 (2022). https://doi.org/10.1186/s13634-022-00842-x
