import os
from PIL import Image
from tqdm import tqdm
import numpy as np

img_path = r'moblieNet/Doraemon_dataset/多啦a梦'  # 填入图片所在文件夹的路径
img_Topath = r'E:\PythonProject\webcv\moblieNet\Doraemon_dataset\多啦a梦RGB'  # 填入图片转换后的文件夹路径

for item in tqdm(img_path):
    arr = item.strip().split('*')
    print(arr)
    img_name = arr[0]
    image_path = os.path.join(img_path, img_name)
    img = Image.open(image_path)
    if (img.mode != 'RGB'):
        img = img.convert("RGB")
        img = np.array(img)
        print(img_name)
        print(img.shape)
        img.save(img_Topath + '/' + img_name, img)