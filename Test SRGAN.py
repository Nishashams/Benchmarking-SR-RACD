#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install keras


# In[2]:


pip install tensorflow


# In[3]:


import tensorflow as tf


# In[4]:


import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import keras
import random

from keras.applications.vgg19 import VGG19
from keras.models import Sequential
from keras import layers, Model
from sklearn.model_selection import train_test_split
from keras import Model
from keras.layers import Conv2D, PReLU,BatchNormalization, Flatten
from keras.layers import UpSampling2D, LeakyReLU, Dense, Input, add
from tqdm import tqdm
from tensorflow import keras
from keras.models import load_model
from numpy.random import random_integers

from skimage.metrics import mean_squared_error
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import normalized_root_mse


# In[5]:


#Define blocks to build the generator
def res_block(ip):
  res_model = Conv2D(64, (3,3), padding = "same")(ip)//64
  res_model = PReLU(shared_axes = [1,2])(res_model)

  res_model = Conv2D(64, (3,3), padding = "same")(res_model)//64

  return add([ip,res_model])

def upscale_block(ip):
  up_model = Conv2D(256, (3,3), padding="same")(ip)
  up_model = UpSampling2D( size = 2 )(up_model)
  up_model = PReLU(shared_axes=[1,2])(up_model)

  return up_model


# In[6]:


#Descriminator block that wiL be used to construct the discrininator
def discriminator_block(ip, filters, strides=1, bn=True):
  disc_model = Conv2D(filters, (3,3), strides = strides, padding="same")(ip)

  disc_model = LeakyReLU( alpha=0.2 )(disc_model)
  return disc_model


# In[7]:


#Generator model
num_res_block = 16

def create_generator_model(gen_ip, num_res_block):
    # Initial convolutional layer
    layers = Conv2D(64, (9, 9), padding="same")(gen_ip)
    layers = PReLU(shared_axes=[1, 2])(layers)

    # Store a copy of the initial layer for later addition
    temp = layers

    # Residual blocks
    for _ in range(num_res_block):
        layers = res_block(layers)

    # Final residual block and addition with the initial layer
    layers = Conv2D(64, (3, 3), padding="same")(layers)
    layers = add([layers, temp])

    # Upscaling blocks
    layers = upscale_block(layers)
    layers = upscale_block(layers)

    # Output convolutional layer
    output_layer = Conv2D(3, (9, 9), padding="same")(layers)

    # Define and return the generator model
    return Model(inputs=gen_ip, outputs=output_layer)


# In[8]:


#descriminartor, as described in the original paper
def create_disc(disc_ip):
  df = 64
  d1 = discriminator_block(disc_ip, df, bn=False)
  d2 = discriminator_block(d1, df, strides=2)
  d3 = discriminator_block(d2, df*2)
  d4 = discriminator_block(d3, df*2, strides=2)
  ds = discriminator_block(d4, df*4)
  d6 = discriminator_block(ds, df*4, strides=2)
  d7 = discriminator_block(d6, df*8)
  d8 = discriminator_block(d7, df*8, strides=2)
  d8_5 = Flatten()(d8)
  d9 = Dense(df*16)(d8_5)
  d10 = LeakyReLU(alpha=0.2)(d9)
  validity = Dense(1, activation='sigmoid')(d10)
  return Model( disc_ip, validity)


# In[9]:


def build_vgg(hr_shape):
  vgg = VGG19(weights="imagenet",include_top=False, input_shape=hr_shape)
  return Model (inputs=vgg.inputs, outputs=vgg.layers[10].output)


# In[10]:


'''
#Combined model
def create_comb(gen_model, disc_model, vgg, lr_ip, hr_ip):
  gen_img = gen_model(lr_ip)
  gen_features = vgg(gen_img)
  disc_model.trainable = False
  validity = disc_model(gen_img)
  return Model(inputs=[lr_ip, hr_ip], outputs=[validity, gen_features])
'''


# In[16]:





# In[ ]:


'''
import zipfile
import os

# Path to the zip file you want to unzip
zip_file_path = '/home/jupyter-nisha_socs/FINALCCTV.zip'

# Directory where you want to extract the files
extracted_dir_path = '/home/jupyter-nisha_socs/FINALCCTV'

# Create the directory if it doesn't exist
if not os.path.exists(extracted_dir_path):
    os.makedirs(extracted_dir_path)

# Open the zip file
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    # Extract all the contents to the specified directory
    zip_ref.extractall(extracted_dir_path)

print("Extraction complete.")
'''


# In[ ]:


#%pwd


# In[ ]:


'''
from PIL import Image
import os

# Path to the directory containing images
folder_path = "/home/jupyter-nisha_socs/FINALCCTV"

# Create folders if they don't exist
lr_folder = os.path.join(folder_path, "lrimages")
hr_folder = os.path.join(folder_path, "hrimages")
os.makedirs(lr_folder, exist_ok=True)
os.makedirs(hr_folder, exist_ok=True)

# Function to resize images
def resize_image(input_path, output_path, size):
    for filename in os.listdir(input_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):  # Assuming images are JPG or PNG
            img = Image.open(os.path.join(input_path, filename))
            img = img.resize((size, size))  # No need to specify antialiasing
            img.save(os.path.join(output_path, filename))

# Resize 256x256 images to 64x64 and save in lrimages folder
resize_image(folder_path, lr_folder, 64)

# Resize 128x128 images to 128x128 and save in hrimages folder
resize_image(folder_path, hr_folder, 128)

print("Image resizing completed.")
'''


# In[27]:


n=5900

train_dir = "/home/jupyter-nisha_socs/FINALCCTV"

lr_list = os.listdir(train_dir+"/lrimages")[:n]
lr_images = []
for img in lr_list:
  img_lr = cv2.imread(train_dir+"/lrimages/" + img)
  img_lr = cv2.cvtColor(img_lr, cv2.COLOR_BGR2RGB)
  lr_images.append(img_lr)

hr_list = os.listdir(train_dir+"/hrimages")[:n]
hr_images = []
for img in hr_list:
  img_hr = cv2.imread(train_dir+"/hrimages/" + img)
  img_hr = cv2.cvtColor(img_hr, cv2.COLOR_BGR2RGB)
  hr_images.append (img_hr)

lr_images = np.array(lr_images)
hr_images = np.array(hr_images)

print(lr_images.shape)
print(hr_images.shape)


# In[28]:


#Sanity check, view few mages
# Sanity check, view few images
import random
import matplotlib.pyplot as plt

# Assuming lr_images and hr_images are lists of images
image_number = random.randint(0, len(lr_images)-1)
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.imshow(lr_images[image_number])
plt.subplot(122)
plt.imshow(hr_images[image_number])
plt.show()


# In[29]:


lr_images = lr_images / 255.
hr_images = hr_images / 255.
#Split to train and test
lr_train, lr_test, hr_train, hr_test = train_test_split(lr_images, hr_images, test_size=0.33, random_state=1)


# In[30]:


hr_shape = (hr_train.shape[1], hr_train.shape[2], hr_train.shape[3])
lr_shape = (lr_train.shape[1], lr_train.shape[2], lr_train.shape[3])

lr_ip = Input(shape=lr_shape)
hr_ip = Input(shape=hr_shape)

generator = create_generator_model(lr_ip, num_res_block = 16)
generator.summary()


# In[31]:


discriminator = create_disc(hr_ip)
discriminator.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy'])
discriminator.summary()


# In[36]:


def create_comb(gen_model, disc_model, vgg, lr_ip, hr_ip):
    # Generate images
    gen_img = gen_model(lr_ip)
    
    # Resize generated images to match discriminator input shape
    resized_gen_img = Lambda(lambda x: tf.image.resize(x, (128, 128)))(gen_img)
    
    # Extract features from generated images using VGG
    gen_features = vgg(resized_gen_img)
    
    # Set discriminator to non-trainable
    disc_model.trainable = False
    
    # Determine validity of generated images
    validity = disc_model(resized_gen_img)  # Resize before passing to the discriminator
    
    # Combined model
    combined = Model(inputs=[lr_ip, hr_ip], outputs=[validity, gen_features])
    
    return combined


# In[33]:


vgg = build_vgg((128,128,3))
print(vgg.summary())
vgg.trainable = False


# In[34]:


pip install image-quality


# In[37]:


gan_model = create_comb(generator, discriminator, vgg, lr_ip, hr_ip)
gan_model.compile(loss=["binary_crossentropy","mse"], loss_weights=[1e-3, 1],optimizer='adam')
gan_model.summary()


# In[38]:


keras.utils.plot_model(gan_model, show_shapes=True)


# In[39]:


batch_size = 2
train_lr_batches = []
train_hr_batches = []
for it in range(int(hr_train.shape[0] / batch_size)):
  start_idx = it * batch_size
  end_idx = start_idx + batch_size
  train_hr_batches.append (hr_train[start_idx:end_idx])
  train_lr_batches.append (lr_train[start_idx:end_idx])


# In[40]:


def plotLosses(d_loss, g_loss, e):
        fig, ax1 = plt.subplots(figsize=(5, 5))
        color = 'tab:blue'
        ax1.plot(d_loss, color=color, label='Dis loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Dis loss', color=color)
        ax1.tick_params('y', color=color)
        color = 'tab:green'
        ax2 = ax1.twinx()
        ax2.plot(g_loss, color=color, label='Gen loss')
        ax2.set_ylabel('Gen loss', color=color)
        ax2.tick_params('y', color=color)
        plt.title('Discriminator & Generator Losses')
        plt.savefig('Losses_%d.png' % e)
        plt.show()


# In[42]:


from tensorflow.image import resize

# Resize input images to match the expected size of the discriminator
lr_imgs_resized = resize(lr_imgs, (128, 128))
hr_imgs_resized = resize(hr_imgs, (128, 128))

# Then use these resized images in the discriminator
d_loss_gen = calculate_loss(discriminator, lr_imgs_resized, fake_label)
d_loss_real = calculate_loss(discriminator, hr_imgs_resized, real_label)


# In[41]:


epochs = 20

def calculate_loss(model, data, labels):
    return model.train_on_batch(data, labels)

def train_discriminator(discriminator, generator, lr_imgs, hr_imgs, fake_label, real_label):
    fake_imgs = generator.predict_on_batch(lr_imgs)

    discriminator.trainable = True
    d_loss_gen = calculate_loss(discriminator, fake_imgs, fake_label)
    d_loss_real = calculate_loss(discriminator, hr_imgs, real_label)

    discriminator.trainable = False

    d_loss = 0.5 * np.add(d_loss_gen, d_loss_real)
    return d_loss

def train_generator(gan_model, lr_imgs, hr_imgs, real_label, vgg):
    discriminator.trainable = False
    image_features = vgg.predict(hr_imgs)
    g_loss, _, _ = calculate_loss(gan_model, [lr_imgs, hr_imgs], [real_label, image_features])
    return g_loss

for e in range(epochs):
    fake_label = np.zeros((batch_size, 1))
    real_label = np.ones((batch_size, 1))

    g_losses = []
    d_losses = []

    for lr_imgs, hr_imgs in tqdm(zip(train_lr_batches, train_hr_batches), total=len(train_hr_batches)):
        d_loss = train_discriminator(discriminator, generator, lr_imgs, hr_imgs, fake_label, real_label)
        g_loss = train_generator(gan_model, lr_imgs, hr_imgs, real_label, vgg)

        d_losses.append(d_loss)
        g_losses.append(g_loss)

    g_loss_avg = np.mean(g_losses, axis=0)
    d_loss_avg = np.mean(d_losses, axis=0)

    print(f"Epoch: {e + 1}, Generator Loss: {g_loss_avg}, Discriminator Loss: {d_loss_avg}")

    if (e + 1) % 10 == 0:
        discriminator.save_weights(f"disc_e_{e+1}.h5")
        generator.save_weights(f"gen_e_{e+1}.h5")


# In[ ]:


plt.plot(g_losses,label='g_losses')
plt.plot(d_losses,label='d_losses')
plt.legend()


# In[ ]:


import matplotlib.pyplot as plt
import pandas as pd

# Example data for g_losses and d_losses (replace with your actual data)
g_losses = [0.1, 0.2, 0.15, 0.25, 0.18]
d_losses = [0.05, 0.08, 0.12, 0.1, 0.15]

# Create a DataFrame
loss_data = {
    'Epoch': range(1, len(g_losses) + 1),
    'g_losses': g_losses,
    'd_losses': d_losses
}
df = pd.DataFrame(loss_data)

# Apply formatting and colors
styled_df = df.style.format({
    'g_losses': '{:.2f}',
    'd_losses': '{:.2f}'
}).set_properties(**{'text-align': 'center'})

styled_df = styled_df.set_table_styles([{
    'selector': 'th',
    'props': [
        ('background-color', 'lightgrey'),
        ('text-align', 'center')
    ]
}, {
    'selector': 'tr:hover',
    'props': [('background-color', 'yellow')]
}])

# Display the formatted table
styled_df


# In[ ]:


temp=str(e+1)
print(temp)
get_ipython().system('ls gen_e_10.h5')


# In[ ]:


# generator = load_model('gen_e_10.h5', compile=False)

[X1, X2] = [lr_test, hr_test]

# Use np.random.randint instead of random_integers
ix = np.random.randint(0, len(X1), 1)
src_image, tar_image = X1[ix], X2[ix]

# Use generator.predict_on_batch for consistency
gen_image = generator.predict_on_batch(src_image)

# Plotting
plt.figure(figsize=(8, 8))

# Plot LR Image
plt.subplot(231)
plt.title('LR Image')
plt.imshow(src_image[0])

# Plot Superresolution Image
plt.subplot(232)
plt.title('Superresolution')
plt.imshow(gen_image[0])

# Plot Original HR Image
plt.subplot(233)
plt.title('Orig. HR image')
plt.imshow(tar_image[0])

plt.show()


# In[ ]:


from tabulate import tabulate


# In[ ]:


mse_img=mean_squared_error(gen_image[0,:,:,:],tar_image[0,:,:,:])
print("MSE : ",mse_img)

ssim_img=ssim(gen_image[0,:,:,:],tar_image[0,:,:,:],multichannel=True)
print("SSIM : ",ssim_img)

rmse_img=normalized_root_mse(gen_image[0,:,:,:],tar_image[0,:,:,:])
print("NRMSE : ",rmse_img)


# In[ ]:


from skimage.metrics import peak_signal_noise_ratio

# Assuming your images are in the range [0, 255] for 8-bit images
data_range = 255
psnr_img = peak_signal_noise_ratio(gen_image[0,:,:,:], tar_image[0,:,:,:], data_range=data_range)
print("PSNR : ", psnr_img)


# In[ ]:


import pandas as pd


# In[ ]:


from prettytable import PrettyTable


# In[ ]:


columns = ['MSE','SSIM','NRMSE','PSNR']
#arr=[[mse_img,ssim_img,rmse_img,score]]
arr=[[mse_img,ssim_img,rmse_img,psnr_img]]
print (tabulate(arr, headers=["MSE","SSIM","NRMSE","PSNR"],tablefmt="fancy_grid"))

