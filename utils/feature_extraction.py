# All Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
from io import BytesIO
from PIL import Image
from PIL import ImageFile
import glob
ImageFile.LOAD_TRUNCATED_IMAGES = True #https://stackoverflow.com/questions/12984426/python-pil-ioerror-image-file-truncated-with-big-images
import os

from keras.models import Sequential
from keras.layers import Flatten, Dense, GlobalAveragePooling2D
from keras.applications.resnet50 import ResNet50
# TODO: Import the model and the preprocess_input function
from keras.applications.resnet50 import ResNet50, preprocess_input


# TODO: Import the ImageDataGenerator class
from keras.preprocessing.image import ImageDataGenerator

import torchvision.transforms as transforms

# All Parameters
PROJECT_FOLDER = '/content/drive/MyDrive/projects/kNearestPet'
DATA_PATH = 'cats-dogs-data/'

DOGSCATS='http://files.fast.ai/data/examples/dogscats.tgz'
DOG_BREED_URL = 'https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip'
ZIP_FILENAME = 'cats-dogs-data.zip'
PETS='https://s3.amazonaws.com/fast-ai-imageclas/oxford-iiit-pet.tgz'

# Assign training and validation folders
TRAIN_PATH = f'{DATA_PATH}train/'
VALID_PATH = f'{DATA_PATH}valid/'

# Functions and classes
def download_dataset(data_url, zip_filename):
    try: # Then try downloading and unzipping it
        print("Downloading Dataset...")
        os.system(f"wget {data_url}")

        print("Unzipping Dataset")
        if 'tgz' in zip_filename:
            os.system(f"tar -xvzf {zip_filename}")
        else:
            os.system(f"unzip {zip_filename}")

        print("Removing .zip file")
        os.system(f"rm {zip_filename}")
    except Exception as e: # If there's an error, ask to download manually
        print(f"Something went wrong. Please download the dataset manually at {data_url}")
        print(f'The following exception was thrown:\n{e}')

class ImageFeatureExtractor():
  def __init__(self, model_name='resnet', target_shape=(224, 224, 3)):
    self.target_shape = target_shape
    self.model = self._get_model(model_name)
    self.model_name = model_name

  def _center_crop_img(self, img, size=224): #using pytorch as it gives more freedom in the transformations
    tr = transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop(size),
    ])  
    return tr(img)
  
  def prepare_img_torch(img):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ])

    img = transform(img)

    # Add batch dimension
    img = img.unsqueeze(0)

    return img

  def _get_model(self, model_name):
    if model_name=='resnet':
      base_model = ResNet50(include_top=False, input_shape=self.target_shape)
      for layer in base_model.layers:
        layer.trainable=False
          
      model = Sequential([base_model,
                          GlobalAveragePooling2D()])
    return model

  def img_to_vector(self, img):
    img_np = self._preprocess_img(img)
    vector = self.model.predict(img_np)
    return vector

  def _get_img_gen_from_df(self, dataframe):

    datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    gen = datagen.flow_from_dataframe(dataframe, 
                                    target_size=self.target_shape[:2], 
                                    class_mode=None,
                                    shuffle=False)
    return gen
    
  def _get_img_gen(self, folder_path):

    # TODO: Initialize the data generator class 
    datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    # TODO: Create the training and validation generators using the method flow_from_directory
    gen = datagen.flow_from_directory(folder_path, 
                                      target_size=self.target_shape[:2], 
                                      class_mode='sparse',
                                      shuffle=False)
    return gen

  def img_folder_to_vectors(self, img_folder, raw_folder=False):
    if raw_folder:
      filepaths = glob.glob(img_folder+'/*.*')
      dataframe=pd.DataFrame(filepaths, columns=['filename'])
      gen = self._get_img_gen_from_df(dataframe)
    else:
      gen = self._get_img_gen(img_folder)

    all_vectors=self.model.predict(gen, verbose=1)
    df=pd.DataFrame(all_vectors)
    df['filepaths']=gen.filepaths
    return df

  def url_to_vector(self, url):
    img = self.read_img_url(url)
    vector = self.img_to_vector(img)
    return vector

  def read_img_url(self, url, center_crop=True):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    if center_crop:
      img = self._center_crop_img(img)
    return img

  def vectors_from_folder_list(self, folder_list):
    df_list = []
    for folder_path in folder_list:
      df=self.img_folder_to_vectors(folder_path)
      df_list.append(df)
    return pd.concat(df_list)

    
  def _preprocess_img(self, img):
    # Convert to a Numpy array
    img_np = np.asarray(img)

    # Reshape by adding 1 in the beginning to be compatible as input of the model
    img_np = img_np[None] # https://docs.scipy.org/doc/numpy/reference/arrays.indexing.html#numpy.newaxis

    # Prepare the image for the model
    img_np = preprocess_input(img_np)

    return img_np

def test():
	ife = ImageFeatureExtractor()
	url='https://raw.githubusercontent.com/WittmannF/ImageDataGenerator-example/master/flow_from_dataframe/train/cat.101.jpg'
	img=ife.read_img_url(url)
	vector=ife.img_to_vector(img)
	vector=ife.url_to_vector(url)
	print(vector)
	os.system("git clone https://github.com/WittmannF/ImageDataGenerator-example.git")
	df=ife.img_folder_to_vectors('./flow_from_directory/train')
	folder_list = ['./flow_from_directory/train', 
               './flow_from_directory/valid']

	df=ife.vectors_from_folder_list(folder_list)
	print(df.head())


if __name__ == '__main__':
	test()