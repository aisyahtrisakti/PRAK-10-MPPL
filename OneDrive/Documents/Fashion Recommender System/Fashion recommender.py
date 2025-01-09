import numpy as np
import pandas as pd
import os
import re
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Activation, Dropout, Flatten, Dense, Input, Layer
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from IPython.display import display, FileLink
from ipywidgets import FileUpload
from sklearn.decomposition import PCA
import random
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import random
from sklearn.metrics import accuracy_score

plt.rcParams['font.size'] = 16

img_path = '/kaggle/input/fashion-product-images-dataset/fashion-dataset/images'
img_df = pd.read_csv('/kaggle/input/fashion-product-images-dataset/fashion-dataset/images.csv')
styles_df = pd.read_csv('/kaggle/input/fashion-product-images-dataset/fashion-dataset/styles.csv', on_bad_lines='skip')

img_df.head()

styles_df.info()

styles_df.head()

styles_df['filename'] = styles_df['id'].astype(str) + '.jpg'

styles_df

img_files = os.listdir(img_path)

styles_df['present'] = styles_df['filename'].apply(lambda x:x in img_files)

styles_df = styles_df[styles_df['present']].reset_index(drop=True)

styles_df = styles_df.sample(1000)

img_size = 224

datagen = ImageDataGenerator(rescale=1/255.)

generator = datagen.flow_from_dataframe(dataframe = styles_df,
                                       directory = img_path,
                                       target_size = (img_size,img_size),
                                       x_col = 'filename',
                                       class_mode = None,
                                       batch_size = 32,
                                       shuffle = False,
                                       classes= None)

datagen = ImageDataGenerator(rescale=1/255., validation_split=0.2)

train_generator = datagen.flow_from_dataframe(
    dataframe=styles_df,
    directory=img_path,
    target_size=(img_size, img_size),
    x_col='filename',
    y_col='articleType',  
    class_mode='categorical',  
    batch_size=32,
    shuffle=True,  
    subset='training'
)

validation_generator = datagen.flow_from_dataframe(
    dataframe=styles_df,
    directory=img_path,
    target_size=(img_size, img_size),
    x_col='filename',
    y_col='articleType',
    class_mode='categorical',
    batch_size=32,
    shuffle=False, 
    subset='validation'
)
from keras.applications import VGG16
from keras.layers import Input, GlobalAveragePooling2D
from keras.models import Model
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier

weights_path = 'C:/Users/asus/Downloads/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'

# Membuat model VGG16 dengan menggunakan berat yang telah diunduh secara lokal
base_model = VGG16(include_top=False, input_shape=(img_size, img_size, 3), weights=None)  # Ubah weights menjadi None

# Melakukan pengaturan agar semua layer tidak dapat di-train
for layer in base_model.layers[:-4]:
    layer.trainable = False

# Menambahkan lapisan-lapisan tambahan di atas VGG16
input_layer = Input(shape=(img_size, img_size, 3))
x = base_model(input_layer)
output = GlobalAveragePooling2D()(x)
output = Dense(115, activation='softmax')(output)

# Membuat model akhir dengan lapisan-lapisan tambahan
embeddings = Model(inputs=input_layer, outputs=output)
embeddings.summary()

# Mengubah pengaturan `trainable` untuk lapisan Dense
for layer in embeddings.layers[-2:]:
    layer.trainable = True


embeddings.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history = embeddings.fit(train_generator, epochs=20, batch_size=32, validation_data=validation_generator)

# Print both training and validation accuracy
print("Train Accuracy:", history.history['accuracy'])
print("Validation Accuracy:", history.history['val_accuracy'])

# Plot training and validation accuracy
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

X = embeddings.predict(generator, verbose=1)

pca = PCA(2)
X_pca = pca.fit_transform(X)

styles_df[['pc1', 'pc2']] = X_pca

plt.figure(figsize= (20,12))
sns.scatterplot(x='pc1',y='pc2',data = styles_df)
plt.show()

def read_img(image_path):
    image = load_img(os.path.join(img_path,image_path),target_size=(img_size,img_size,3))
    image = img_to_array(image)
    image = image/255.
    return image

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

y = styles_df['id']

nearest_neighbors = KNeighborsClassifier(n_neighbors = 7)
nearest_neighbors.fit(X,y)

styles_df = styles_df.reset_index(drop=True)

for _ in range(10):
    i = random.randint(0, len(styles_df))
    img1 = read_img(styles_df.loc[i, 'filename'])
    dist, index = nearest_neighbors.kneighbors(X=X[i, :].reshape(1, -1))
    
    plt.figure(figsize=(4, 4))
    plt.imshow(img1)
    plt.title("Input Image")
    plt.axis('off')
   
    plt.figure(figsize=(20, 20))
    for j in range(1, 6):
        plt.subplot(1, 5, j)
        plt.subplots_adjust(hspace=0.5, wspace=0.3)
        image = read_img(styles_df.loc[index[0][j], 'filename'])
        plt.imshow(image)
        plt.title(f'Similar Product #{j}')
        plt.axis('off')

y_pred = nearest_neighbors.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")


