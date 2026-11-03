import numpy as np
from matplotlib.image import imread
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

#1
image_raw = imread("dog.JPG")
print('початкова розмірність: ',image_raw.shape) #розмірність

plt.figure(figsize = (10,10))
plt.imshow(image_raw)
plt.axis('off')
plt.show()


#2
image_sum = image_raw.sum(axis = 2)
print('розмірність після sum каналів: ',image_sum.shape) #
image_bw = image_sum / image_sum.max()
print('max значення пікселя: ',image_bw.max())

plt.figure(figsize = (10,10))
plt.imshow(image_bw, cmap = 'gray')
plt.axis('off')
plt.show()

#3
pca = PCA()
pca.fit(image_bw)
#обчислення кумулятивної суми де вперше стає 0.95 або більше
cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
number_of_components_95 = np.argmax(cumulative_variance >= 0.95) + 1


#нове фото
pca95 = PCA(n_components = number_of_components_95) #only 95%
transformed_image_95 = pca95.fit_transform(image_bw) #стиснення
reconstructed_image_95 = pca95.inverse_transform(transformed_image_95)  #розпаковує але вже без тих 5%
plt.figure(figsize = (10,10))
plt.imshow(reconstructed_image_95, cmap = 'gray')
plt.axis('off')
plt.show()


#4
pca5 = PCA(n_components = 5)
transformed_pca5 = pca5.fit_transform(image_bw)
reconstructed_pca5 = pca5.inverse_transform(transformed_pca5)
plt.figure(figsize = (10,10))
plt.imshow(reconstructed_pca5, cmap = 'gray')
plt.title('5 components')
plt.axis('off')
plt.show()


pca15 = PCA(n_components = 15)
transformed_pca15 = pca15.fit_transform(image_bw)
reconstructed_pca15 = pca15.inverse_transform(transformed_pca15)
plt.figure(figsize = (10,10))
plt.imshow(reconstructed_pca15, cmap = 'gray')
plt.title('15 components')
plt.axis('off')
plt.show()


pca25 = PCA(n_components = 25)
transformed_pca25 = pca25.fit_transform(image_bw)
reconstructed_pca25 = pca25.inverse_transform(transformed_pca25)
plt.figure(figsize = (10,10))
plt.imshow(reconstructed_pca25, cmap = 'gray')
plt.title('25 components')
plt.axis('off')
plt.show()


pca75 = PCA(n_components = 75)
transformed_pca75 = pca75.fit_transform(image_bw)
reconstructed_pca75 = pca75.inverse_transform(transformed_pca75)
plt.figure(figsize = (10,10))
plt.imshow(reconstructed_pca75, cmap = 'gray')
plt.title('75 components')
plt.axis('off')
plt.show()


pca100 = PCA(n_components = 100)
transformed_pca100 = pca100.fit_transform(image_bw)
reconstructed_pca100 = pca100.inverse_transform(transformed_pca100)
plt.figure(figsize = (10,10))
plt.imshow(reconstructed_pca100, cmap = 'gray')
plt.title('100 components')
plt.axis('off')
plt.show()


pca170 = PCA(n_components = 170)
transformed_pca170 = pca170.fit_transform(image_bw)
reconstructed_pca170 = pca170.inverse_transform(transformed_pca170)
plt.figure(figsize = (10,10))
plt.imshow(reconstructed_pca170, cmap = 'gray')
plt.title('170 components')
plt.axis('off')
plt.show()
