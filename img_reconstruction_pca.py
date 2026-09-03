import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split 
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
# Preprocessing the images
img_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
]) 
data_dir ="/Users/sowmyabalagala/Downloads/fire_dataset" 
dataset = ImageFolder(data_dir,transform=img_transform)
# Split the dataset into training and testing sets train_dataset, 
train_dataset,test_dataset = train_test_split(dataset, test_size=0.1) 
train_data_loader = DataLoader(train_dataset,batch_size=32, shuffle=True)
test_data_loader = DataLoader(test_dataset,batch_size=32, shuffle=False)
for images, labels in test_data_loader:
       break
# Flattening the images - Convert to 2D array
flattened_batch_imgs = images.view(images.size(0),-1).numpy()
num_samples = len(flattened_batch_imgs)
num_features = flattened_batch_imgs.shape[1]
#Tuned the number of components with values 50 and ,→ 100 to show how PCA Algorithm works
num_components = min(100, min(num_samples,num_features))
pca = PCA(n_components=num_components)
pca.fit(flattened_batch_imgs)
# Reconstructing the images using the principal ,→ components
reconstructed_imgs = pca.inverse_transform(pca.transform(flattened_batch_imgs))
reconstructed_imgs = torch.from_numpy(reconstructed_imgs).view(images.size(0),3, 224, 224)
def plot_images(original_images,reconstructed_images, num_images=5):
       fig, axs = plt.subplots(2, num_images,figsize=(15, 5))
       fig.suptitle("Original and Reconstructed Images ,→ with PCA", fontsize=12)
       for i in range(num_images):
             original_img = original_images[i].permute(1,2,0)
             axs[0, i].imshow((original_img * 0.229 +0.485), cmap='gray')
             axs[0, i].set_title("Original", fontsize=12) 
             axs[0, i].axis('off')
             reconstructed_img = reconstructed_images[i].permute(1, 2,0).numpy()
             axs[1, i].imshow((reconstructed_img * 0.229+ 0.485), cmap='gray')
             axs[1, i].set_title("Reconstructed",fontsize=12)
             axs[1, i].axis('off')
             plt.show()
plot_images(images, reconstructed_imgs,num_images=5)