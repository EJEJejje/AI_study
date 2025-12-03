import os
import shutil
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pickle  #파일저장에

#pip install tqdm
from tqdm import tqdm # 학습 진행 상황 시각화를 위해 추가

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import Image
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
# 이후 PyTorch, NumPy, TensorFlow 등을 import 합니다.
# 경고 무시 설정
warnings.filterwarnings('ignore')

# 1. 경로 설정 및 파라미터
original_dataset_dir = './data/cats_and_dogs/train'
"""
cats_and_dogs
   ㄴ train 

cats_and_dogs_small
   ㄴ train 
       ㄴcats
       ㄴdogs 
   ㄴ test 
       ㄴcats
       ㄴdogs 
   ㄴ validation 
        ㄴcats
       ㄴdogs 

"""
base_dir = './data/cats_and_dogs_small'

train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')
test_dir = os.path.join(base_dir, 'test')

model_save_path_pth = 'cats_and_dogs_model.pth'
history_filepath = 'cats_and_dogs_history.pkl'

batch_size = 16
img_height = 180
img_width = 180
num_epochs = 30 # 예시로 epoch 수를 30으로 설정했습니다.
learning_rate = 0.001 # learning_rate 추가

# GPU 사용 가능 여부 확인
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"사용 중인 디바이스: {device}")

# 이미지 복사 함수 
def ImageCopyMove():
    #경로가 있는지 확인해본다  경로가 있으면 True 없으면 False 
    if os.path.exists(base_dir):
        #shutil :shell util - 명령어 해석기 
        shutil.rmtree(base_dir, ignore_errors=True)
    os.makedirs(train_dir)  #디렉토리 생성 
    os.makedirs(validation_dir)
    os.makedirs(test_dir)
    
    train_cats_dir = os.path.join(train_dir, 'cats')
    train_dogs_dir = os.path.join(train_dir, 'dogs')
    validation_cats_dir = os.path.join(validation_dir, 'cats')
    validation_dogs_dir = os.path.join(validation_dir, 'dogs')
    test_cats_dir = os.path.join(test_dir, 'cats')
    test_dogs_dir = os.path.join(test_dir, 'dogs')

    os.makedirs(train_cats_dir)
    os.makedirs(train_dogs_dir)
    os.makedirs(validation_cats_dir)
    os.makedirs(validation_dogs_dir)
    os.makedirs(test_cats_dir)
    os.makedirs(test_dogs_dir)
    
    #["cat.0.jpg", "cat.1.jpg",..... ]
    fnames = ['cat.{}.jpg'.format(i) for i in range(1000)]
    for fname in fnames:
        src = os.path.join(original_dataset_dir, fname)
        dst = os.path.join(train_cats_dir, fname)
        shutil.copyfile(src, dst)
        
    fnames = ['cat.{}.jpg'.format(i) for i in range(1000, 1500)]
    for fname in fnames:
        src = os.path.join(original_dataset_dir, fname)
        dst = os.path.join(validation_cats_dir, fname)
        shutil.copyfile(src, dst)
    
    fnames = ['cat.{}.jpg'.format(i) for i in range(1500, 2000)]
    for fname in fnames:
        src = os.path.join(original_dataset_dir, fname)
        dst = os.path.join(test_cats_dir, fname)
        shutil.copyfile(src, dst)
    
    fnames = ['dog.{}.jpg'.format(i) for i in range(1000)]
    for fname in fnames:
        src = os.path.join(original_dataset_dir, fname)
        dst = os.path.join(train_dogs_dir, fname)
        shutil.copyfile(src, dst)
    
    fnames = ['dog.{}.jpg'.format(i) for i in range(1000, 1500)]
    for fname in fnames:
        src = os.path.join(original_dataset_dir, fname)
        dst = os.path.join(validation_dogs_dir, fname)
        shutil.copyfile(src, dst)
    
    fnames = ['dog.{}.jpg'.format(i) for i in range(1500, 2000)]
    for fname in fnames:
        src = os.path.join(original_dataset_dir, fname)
        dst = os.path.join(test_dogs_dir, fname)
        shutil.copyfile(src, dst)

    print("이미지 복사 및 폴더 생성 완료!")


ImageCopyMove()  

#꽃분류_cnn 처럼 cnn분류하기 