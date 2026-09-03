# image-analysis-fire-dataset
Image analysis framework for fire image retrieval, reconstruction, and segmentation using MongoDB, PCA, and K-Means.

## Overview

This project explores image analysis techniques for a fire image dataset, focusing on image retrieval, reconstruction, segmentation, and classification.

The framework uses Principal Component Analysis (PCA) for dimensionality reduction and image reconstruction, along with K-Means clustering for image segmentation. MongoDB is used for image data storage and retrieval.

## Problem

Analyzing and processing large collections of fire images can be challenging due to the high dimensionality of image data and the complexity of identifying meaningful visual patterns.

The objective is to develop an image analysis framework that can reduce image dimensionality, reconstruct images, segment image regions, and support image retrieval and classification.

## Approach

The overall pipeline is: 
Fire Image Dataset → Preprocessing → Feature Representation → PCA Reconstruction / K-Means Segmentation → Image Analysis

### Data Processing

* Fire image dataset
* Image preprocessing using Torchvision
* Images resized to 224 × 224
* Conversion of images to tensors
* Image normalization
* Training and testing dataset split

### PCA-Based Image Reconstruction

Principal Component Analysis (PCA) is applied to image representations to reduce dimensionality while retaining important information.

The process includes:

* Flattening image tensors into feature vectors
* Applying PCA to the image representations
* Experimenting with different numbers of principal components
* Transforming images into the reduced-dimensional space
* Reconstructing images using the selected principal components
* Comparing original and reconstructed images visually

### K-Means Image Segmentation

K-Means clustering is used to segment images based on pixel-level color information.

The process includes:

* Converting image pixels into RGB feature vectors
* Applying K-Means clustering
* Experimenting with different numbers of clusters
* Assigning pixels to their corresponding cluster centers
* Reconstructing the segmented image
* Visualizing the original and segmented images

### Image Retrieval

MongoDB is used as part of the image retrieval framework to store and retrieve image data efficiently.

### Classification

SVM-based classification is incorporated into the framework for analyzing and classifying image data.

## Technologies

* Python
* PyTorch
* Torchvision
* NumPy
* Scikit-learn
* Matplotlib
* MongoDB

## Data

The project uses a fire image dataset for experimentation with image reconstruction, segmentation, retrieval, and classification.

Dataset files should be obtained separately and the dataset path configured locally before running the notebooks.

## Results

The project generates visual comparisons of:

* Original and PCA-reconstructed images
* Original and K-Means segmented images

These visualizations demonstrate how dimensionality reduction and clustering can be applied to fire image analysis.

## Future Improvements

* Experiment with additional image feature extraction techniques.
* Evaluate PCA reconstruction quality using quantitative metrics.
* Compare K-Means with other clustering approaches.
* Improve image retrieval and classification performance.
* Expand the framework to support larger image datasets.

