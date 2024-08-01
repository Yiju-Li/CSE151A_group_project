# CSE151A Group Project Final Report

## Team Members
Yiju Li, yil125@ucsd.edu
Charles Choi, dicai@ucsd.edu

<<<<<<< Updated upstream
=======
## Abstraction
In recent years, the integration of advancements in medical imaging technology and machine learning has brought unprecedented innovation to medical diagnostics. Our team project aims to develop a machine learning model to classify Optical Coherence Tomography (OCT) images into four categories: Choroidal Neovascularization (CNV), Diabetic Macular Edema (DME), Drusen, and Normal. By utilizing a comprehensive dataset of OCT images, we hope to improve the accuracy and efficiency of diagnosing these retinal conditions. This project was chosen to address the pressing need for efficient diagnostic tools in ophthalmology, reduce the burden on medical professionals, and provide better care for patients. This project demonstrates the immense potential of artificial intelligence in medical diagnostics, offering not only technical innovation but also the ability to broadly improve public health, especially in areas with limited medical resources. Our goal is to advance medical imaging technology and contribute to better global health outcomes through information technology.

## Introduction

This project was chosen because of the increasing demand for innovative and efficient diagnostic tools in the field of ophthalmology. Traditional methods of diagnosing retinal conditions can be time-consuming and require significant expertise. By applying machine learning to OCT images, we aim to develop a model that can assist ophthalmologists in making quicker and more accurate diagnoses. This not only reduces the burden on healthcare professionals but also enhances patient care through timely treatment.The integration of machine learning with medical imaging is one of the most exciting areas in modern healthcare. Developing a model that can accurately classify OCT images is a prime example of technology solving real-world problems. This project is particularly cool because it showcases the potential of artificial intelligence to transform medical diagnostics, making cutting-edge technology practical and feasible in everyday clinical settings.

The broader impact of having a reliable OCT image classification predictive model cannot be overstated. Early and accurate diagnosis of retinal conditions such as CNV, DME, and Drusen is crucial for effective treatment and prevention of vision loss. A robust machine learning model can significantly enhance diagnostic accuracy, leading to better patient outcomes. Moreover, it can alleviate the workload of ophthalmologists, allowing them to focus on more complex cases and reducing overall healthcare costs。 Additionally, such a model can be deployed in regions with limited access to specialized healthcare providers, democratizing access to quality eye care. By providing a reliable diagnostic tool, we can improve the quality of life for individuals globally, particularly in underserved areas.

We are utilizing a large dataset of labeled Optical Coherence Tomography (OCT) images. This dataset was published on June 1, 2018. The main contributors include Daniel Kermany, Kang Zhang, and Michael Goldbaum. This dataset contains thousands of validated OCT images, which were described and analyzed in the paper "Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning."To ensure accuracy, we have downloaded the latest version of this dataset. The images in the dataset are divided into training and testing sets and grouped by independent patients. The image labels follow the format "disease-randomized patient ID-image number by this patient" and are split into four directories: CNV, DME, DRUSEN, and NORMAL. By using this large and well-labeled dataset, our project aims to improve the training and testing accuracy of the model while ensuring good generalization across different patients. This will further validate the effectiveness of our machine learning model and promote its application in actual medical diagnostics.

## Methods
### Data Exploration

The dataset consists of images classified into different categories. Image paths and corresponding labels were collected by traversing the directory structure of the dataset. This allowed for determining the total number of images and the number of distinct classes.

```python
image_paths, labels = get_image_paths_and_labels(dataset_path)

num_images = len(image_paths)
num_classes = len(set(labels))
```

The distribution of images across the classes was counted and visualized using a bar plot to provide an overview of the class distribution.

```python
class_distribution = Counter(labels)

plt.figure(figsize=(10, 5))
sns.barplot(x=list(class_distribution.keys()), y=list(class_distribution.values()))
plt.title('Class Distribution')
plt.xlabel('Class')
plt.ylabel('Number of Images')
plt.show()
```

Example images from each class were displayed to provide a visual understanding of the dataset's content. Five images from each class were selected and plotted.

```python
def plot_example_images(image_paths, labels, class_distribution, num_examples=5):
    plt.figure(figsize=(15, 15))
    for i, cls in enumerate(class_distribution.keys()):
        cls_images = [image_paths[j] for j in range(len(image_paths)) if labels[j] == cls]
        for j in range(num_examples):
            img = Image.open(cls_images[j])
            plt.subplot(len(class_distribution), num_examples, i * num_examples + j + 1)
            plt.imshow(img)
            plt.axis('off')
            if j == num_examples // 2:
                plt.title(cls)
    plt.show()

plot_example_images(image_paths, labels, class_distribution)
```

The sizes of the images in the dataset were analyzed by extracting the dimensions of each image. The frequency of each image size was then counted. The distribution of image sizes was visualized using a bar plot.

```python
image_sizes = [Image.open(img_path).size for img_path in image_paths]
size_distribution = Counter(image_sizes)

plt.figure(figsize=(12, 6))
sizes, counts = zip(*size_distribution.items())
sizes = [f"{size[0]}x{size[1]}" for size in sizes]
sns.barplot(x=sizes, y=counts)
plt.title('Image Size Distribution')
plt.xlabel('Image Size (width x height)')
plt.ylabel('Number of Images')
plt.xticks(rotation=90)
plt.show()
```

### Data Preprocessing:
Images were preprocessed by cropping them to a specific size. This involved defining a target size and then cropping each image from the center to match this target size. The cropped images were then saved back to their original paths.

Then, the xample images were plotted again after cropping to verify the preprocessing step.

The next step in preprocessing was to convert the images into tensors suitable for model training. This was done using TensorFlow's ImageDataGenerator, which also rescaled the images. Images were loaded from directories, resized to the target size, converted to grayscale, and then transformed into tensors.

```python
data_gen = ImageDataGenerator(rescale=1./255)
data_flow = data_gen.flow_from_directory(
   directory,
   target_size=(496, 496),
   color_mode='grayscale',
   batch_size=32,
   class_mode='sparse'
)
```

### Model 1:

The first model was a simple CNN designed to classify the images into four classes. The model architecture consisted of the following layers:

- **Convolutional Layer 1:** 32 filters, 3x3 kernel size, ReLU activation function, input shape of 496x496x1 (grayscale images).
- **Max Pooling Layer 1:** 2x2 pool size.
- **Convolutional Layer 2:** 64 filters, 3x3 kernel size, ReLU activation function.
- **Max Pooling Layer 2:** 2x2 pool size.
- **Flatten Layer:** Flattened the 2D matrices into 1D vectors.
- **Dense Layer:** 4 neurons with a softmax activation function for the output layer.

The model was compiled with the Adam optimizer, sparse categorical cross-entropy loss, and accuracy as the evaluation metric.

### Model 2:

The second model was a more complex Convolutional Neural Network (CNN) designed to classify the images into four classes. The model architecture consisted of the following layers:

- **Convolutional Layer 1:** 32 filters, 3x3 kernel size, ReLU activation function, input shape of 496x496x1 (grayscale images).
- **Max Pooling Layer 1:** 2x2 pool size.
- **Convolutional Layer 2:** 64 filters, 3x3 kernel size, ReLU activation function.
- **Max Pooling Layer 2:** 2x2 pool size.
- **Convolutional Layer 3:** 128 filters, 3x3 kernel size, ReLU activation function.
- **Max Pooling Layer 3:** 2x2 pool size.
- **Convolutional Layer 4:** 128 filters, 3x3 kernel size, ReLU activation function.
- **Max Pooling Layer 4:** 2x2 pool size.
- **Flatten Layer:** Flattened the 2D matrices into 1D vectors.
- **Dense Layer 1:** 512 neurons with a ReLU activation function.
- **Dense Layer 2:** 4 neurons with a softmax activation function for the output layer.

The model was compiled with the Adam optimizer, sparse categorical cross-entropy loss, and accuracy as the evaluation metric.


This model introduced additional convolutional and pooling layers, as well as a larger dense layer, compared to Model 1.

### Model 3

The third model was a more advanced Convolutional Neural incorporating additional regularization techniques to improve performance and prevent overfitting. The model architecture consisted of the following layers:

- **Convolutional Layer 1:** 32 filters, 3x3 kernel size, ReLU activation function, input shape of 496x496x1 (grayscale images).
- **Max Pooling Layer 1:** 2x2 pool size.
- **Batch Normalization Layer 1:** Normalizes the activations of the previous layer.
- **Dropout Layer 1:** Drops 25% of the input units to prevent overfitting.
- **Convolutional Layer 2:** 64 filters, 3x3 kernel size, ReLU activation function.
- **Max Pooling Layer 2:** 2x2 pool size.
- **Batch Normalization Layer 2:** Normalizes the activations of the previous layer.
- **Dropout Layer 2:** Drops 25% of the input units to prevent overfitting.
- **Convolutional Layer 3:** 128 filters, 3x3 kernel size, ReLU activation function.
- **Max Pooling Layer 3:** 2x2 pool size.
- **Batch Normalization Layer 3:** Normalizes the activations of the previous layer.
- **Dropout Layer 3:** Drops 25% of the input units to prevent overfitting.
- **Convolutional Layer 4:** 128 filters, 3x3 kernel size, ReLU activation function.
- **Max Pooling Layer 4:** 2x2 pool size.
- **Batch Normalization Layer 4:** Normalizes the activations of the previous layer.
- **Dropout Layer 4:** Drops 25% of the input units to prevent overfitting.
- **Flatten Layer:** Flattened the 2D matrices into 1D vectors.
- **Dense Layer 1:** 256 neurons with a ReLU activation function.
- **Batch Normalization Layer 5:** Normalizes the activations of the previous layer.
- **Dropout Layer 5:** Drops 50% of the input units to prevent overfitting.
- **Dense Layer 2:** 4 neurons with a softmax activation function for the output layer.

The model was compiled with the Adam optimizer, sparse categorical cross-entropy loss, and accuracy as the evaluation metric.

## Results
### Data Exploration
- Class Distribution
![](D1.png)
- Example Pictures
![](D2.png)
- Image size distribution
![](D3.png)

### Data Preprocessing
- After cropping
![](D4.png)

### Model 1

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| 0     | 0.39      | 0.43   | 0.41     | 200     |
| 1     | 0.43      | 0.37   | 0.40     | 200     |
| 2     | 0.38      | 0.32   | 0.35     | 200     |
| 3     | 0.26      | 0.31   | 0.28     | 200     |

| Metric     | Value |
|------------|-------|
| Accuracy   | 0.36  |
| Macro Avg  | 0.36  |
| Weighted Avg | 0.36 |

![](D7.png)

- Precision, recall, and F1-scores were relatively low across all classes.
- The overall accuracy was 36%.

### Model 2

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| 0     | 0.68      | 0.71   | 0.70     | 200     |
| 1     | 0.74      | 0.69   | 0.71     | 200     |
| 2     | 0.71      | 0.66   | 0.69     | 200     |
| 3     | 0.60      | 0.66   | 0.63     | 200     |

| Metric     | Value |
|------------|-------|
| Accuracy   | 0.68  |
| Macro Avg  | 0.68  |
| Weighted Avg | 0.68 |

![](D6.png)

- Precision, recall, and F1-scores improved significantly compared to Model 1.
- The overall accuracy was 68%.

### Model 3

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| 0     | 0.92      | 0.81   | 0.86     | 200     |
| 1     | 0.90      | 0.79   | 0.84     | 200     |
| 2     | 0.78      | 1.00   | 0.87     | 200     |
| 3     | 0.69      | 1.00   | 0.82     | 200     |

| Metric     | Value |
|------------|-------|
| Accuracy   | 0.85  |
| Macro Avg  | 0.85  |
| Weighted Avg | 0.85 |

![](D5.png)

- Precision, recall, and F1-scores were highest among the three models.
- The overall accuracy was 85%.

## Jupyter Notebook
The data exploration and preprocessing steps are documented in the Jupyter notebook available [at GitHub](notesbooks/CSE151a_project.ipynb) or [at Google Colab](https://colab.research.google.com/drive/1OZNHYLIo4DFiLE5yqVL_Z7UiDEypC-xF?usp=sharing).

>>>>>>> Stashed changes
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
