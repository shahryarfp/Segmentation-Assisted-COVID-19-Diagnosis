### Segmentation Assisted COVID 19 Diagnosis

### Summary

The project is about an efficient method to segment the infection regions automatically. Then, the predicted segment results can assist the diagnostic network in identifying the COVID19 samples from the CT images. On the other hand, a radiologist‑like segmentation network provides detailed information about the infectious regions by separating areas of ground‑glass, consolidation, and pleural effusion, respectively. The aforementioned method can accurately predict the COVID19 infectious probability and provide lesion regions in CT images.

### Dataset

Here you can see a sample image from the dataset:

![sample image from dataset](./address)

And here is a sample CT image and its mask from the dataset:

![sample CT image and its mask from the dataset](./address)

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

#### Results

Results of the segmentation task:
![seg task result](./address)

Result of the diagnosis task:
![dia figure](./address)
It reached 92% accuracy for the test dataset.
Final results:
![Final result](./address)
