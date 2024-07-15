# CSE151A Group Project

## Team Members
Yiju Li, yil125@ucsd.edu
Charles Choi, dicai@ucsd.edu

## Project Description

Our group project aims to develop a machine learning model to classify Optical Coherence Tomography (OCT) images into four categories: CNE, DME, DRUSEN, and NORMAL. Using a dataset of OCT images, the model will be trained and tested in diagnosing these retinal conditions from OCT scans.

## Repository Structure

```
.
├── notebooks/
│   └── CSE151a_project.ipynb
├── README.md
```

## Data Exploration

The data exploration step includes evaluating the dataset, determining the number of observations, and analyzing the data distribution. The key points include:
- Number of classes: 4
- Total number of images: 109,309
- Image size distribution:
  - (512, 496): 58,037 images
  - (768, 496): 29,749 images
  - (1536, 496): 11,512 images
  - (1024, 496): 1,329 images
  - Other sizes: Few images

Example images from each class and the image size distribution are plotted in the Jupyter notebook.

## Data Preprocessing

The preprocessing steps involve:
1. **Cropping**: Cropping images to a uniform size of (496, 496) pixels.


The cropped images are saved back to their original paths.


## Jupyter Notebook

The data exploration and preprocessing steps are documented in the Jupyter notebook available [at github](notebooks/CSE151a_project.ipynb) or [at google colab](https://colab.research.google.com/drive/1OZNHYLIo4DFiLE5yqVL_Z7UiDEypC-xF?usp=sharing).
