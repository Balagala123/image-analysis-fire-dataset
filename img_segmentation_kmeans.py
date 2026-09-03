import os
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset 
from torchvision.datasets import ImageFolder 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
# Preprocessing the images
img_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
]) 
data_dir = "/Users/sowmyabalagala/Downloads/fire_dataset" 
dataset = ImageFolder(data_dir,transform=img_transform) 
rand_sample_loader = DataLoader(dataset,batch_size=1, shuffle=True)
for image,label in rand_sample_loader:
    break
img_data = image[0].permute(1, 2, 0).numpy()
height,width,channels = img_data.shape
flattened_img = img_data.reshape(-1,channels)
# K-means clustering
#Tuned the number of clusters as 2,5,10,15 to show how the algorithm works
num_clusters = 15
kmeans = KMeans(n_clusters=num_clusters,random_state=0)
cluster_labels = kmeans.fit_predict(flattened_img) 
cluster_centers = kmeans.cluster_centers_ 
segmented_image = cluster_centers[cluster_labels].reshape(height,width,channels) 
segmented_image = np.clip(segmented_image / 255, 0,1) if segmented_image.max() > 1.0 else segmented_image
def plot_segmentation(original_image,segmented_image, title="K-means Segmentation"):
    fig, axis = plt.subplots(1, 2, figsize=(10, 5))
    axis[0].imshow(original_image.numpy().transpose(1, 2,0))
    axis[0].set_title('Original Image')
    axis[0].axis('off')
    axis[1].imshow(segmented_image)
    axis[1].set_title('Segmented Image with K-means') 
    axis[1].axis('off') 
    plt.suptitle(title) 
    plt.show()
plot_segmentation(image[0], segmented_image,title="Original and K-means Segmented Image")