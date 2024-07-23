# CSE151A Group Project

## Team Members
Yiju Li, yil125@ucsd.edu  
Charles Choi, dicai@ucsd.edu

## Jupyter Notebook
The data exploration and preprocessing steps are documented in the Jupyter notebook available [at GitHub](notesbooks/CSE151a_project.ipynb) or [at Google Colab](https://colab.research.google.com/drive/1OZNHYLIo4DFiLE5yqVL_Z7UiDEypC-xF?usp=sharing).

## Project Description
Our group project aims to develop a machine learning model to classify Optical Coherence Tomography (OCT) images into four categories: CNV, DME, DRUSEN, and NORMAL. Using a dataset of OCT images, the model will be trained and tested in diagnosing these retinal conditions from OCT scans.

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
2. **Grayscale Conversion**: Converting images to grayscale to reduce complexity.
3. **Tensor Conversion**: Converting images and labels to tensors for model training.

The cropped images are saved back to their original paths.

## Model Training
We constructed a convolutional neural network (CNN) model to classify the OCT images. The steps involved:
1. **Model Architecture**:
   - Convolutional layers with ReLU activation and max pooling.
   - Fully connected dense layers with ReLU activation.
   - Softmax output layer for classification into four categories.
2. **Training**:
   - The model was trained on the training dataset using the Adam optimizer and sparse categorical cross-entropy loss.
   - Validation was performed using the validation dataset to tune hyperparameters.

## Questions and Evaluation for MS3

### Evaluate our model and compare training vs. test error
The model was evaluated on both the training and test datasets. The training accuracy and loss, as well as the test accuracy and loss, were computed to compare the performance.

- **Training Accuracy**: `100%`
- **Training Loss**: `3.352760558072987e-08`
- **Test Accuracy**: `34.375%`
- **Test Loss**: `42.396453857421875`

### Fitting Graph and Future Models
- **Fitting Graph**: The model appears to be fitting bad, with a big gap between training and test accuracy, indicating not too good generalization(Overfitting). Therefore, a lot of further improvements can be made.
- **Future Models**: We are considering the following approaches to improve our model:
  1. **Data Augmentation**: To increase the diversity of the training data.
  2. **Deeper Network Architectures**: To capture more complex features.


