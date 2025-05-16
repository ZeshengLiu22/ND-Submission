import os
from PIL import Image
import numpy as np
from tqdm import tqdm

for value in tqdm(os.listdir('../../../data/jpk322/FloodNet/val/val-label-img')):
    image = Image.open(f'../../../data/jpk322/FloodNet/val/val-label-img/{value}')

    # Convert image to numpy array
    image_np = np.array(image)

    # Get unique values
    unique_values = np.unique(image_np)
    print(len(unique_values))
    # if len(unique_values) >:
    #     print(value)