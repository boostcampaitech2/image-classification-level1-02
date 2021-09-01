import os
import pandas as pd
from glob import glob
from tqdm import tqdm
import pandas as pd
import numpy as np
from torchvision.utils import save_image
from PIL import Image


def mapAge(path):
    """
    path에 해당되는 사람의 age group 번호를 반환합니다
    """
    age_group = path[-2]
    if age_group < "3":
        age = 0
    elif age_group < "6":
        age = 1
    else:
        age = 2
    return age


def mapMask(image):
    if "normal" in image:
        mask = 2
    elif "incorrect" in image:
        mask = 1
    else:
        mask = 0
    return mask


def mapAttributes(base_path, dataframe):
    '''
    전체 파일명과 gender, age, mask 여부를 작성해 csv 파일로 저장
    '''
    images = []
    for path in tqdm(dataframe["path"]):

        # map gender
        gender = 1 if "female" in path else 0

        # map age
        age = mapAge(path)

        images_in_folder = glob.glob(os.path.join(base_path, "images", path, "*"))
        for image in images_in_folder:
            image_dict = {}
            image_dict["file"] = image[20:] # "images/..."
            image_dict["gender"] = gender
            image_dict["age"] = age

            # map mask
            if "normal" in image:
                mask = 2
            elif "incorrect" in image:
                mask = 1
            else:
                mask = 0
            # map mask
            image_dict["mask"] = mapMask(image)
            
            images.append(image_dict)
            
    # print(len(images))  # 18900
    df = pd.DataFrame(images)
    df.to_csv(os.path.join(base_path, "datapath_and_attribute.csv"), header=True, index=False)
    

    
def save_aug(transform, labels):
    '''
    이미지를 data augmentation한 후 별도의 폴더에 저장.
    증강된 이미지들에 대한 폴더명+나이, 파일명+나이대번호 정보를 담은 csv파일 각각 생성
    '''
    base_dir = '/opt/ml/input/data/train/'
    aug_dir = '/opt/ml/input/data/train/imgaug/'
    train_dir = os.path.join(base_dir, 'images/')
    
    if not os.path.exists(aug_dir):
        os.makedirs(aug_dir)
    
    aug_images = []
    aug_paths = []
    for path in tqdm(labels['path']):  # 001038_male_Asian_60, .. 
        new_path = path[:-1] + "1"  # 60 -> 61
        if not os.path.exists(os.path.join(aug_dir, new_path)):
            os.makedirs(os.path.join(aug_dir, new_path))  # "~~imgaug/001038_male_Asian_61"
            
        # make old_path_augmented.csv
        path_dict = {}
        path_dict["path"] = new_path
        path_dict["age"] = 61
        aug_paths.append(path_dict)
        
        img_folder = glob(os.path.join(train_dir, path, '*'))  # ~~images/001038_male_Asian_60/*
        
        for img in img_folder:
            # save aug images
            im = Image.open(img)
            im = transform(im)
            file_name = img.split('/')[-1]  # mask1.jpg
            save_image(im, os.path.join(aug_dir, new_path, file_name))
            
            # make old_data_augmented.csv
            aug_dict = {}
            aug_dict["file"] = os.path.join("imgaug", new_path, file_name)
            aug_dict["age"] = 2
            aug_images.append(aug_dict)
            

    
    # create csv
    df1 = pd.DataFrame(aug_images)
    df1.to_csv(os.path.join(base_dir, "old_data_augmented.csv"), header=True, index=False)
    
    df2 = pd.DataFrame(aug_paths)
    df2.to_csv(os.path.join(base_dir, "old_path_augmented.csv"), header=True, index=False)
        

            

