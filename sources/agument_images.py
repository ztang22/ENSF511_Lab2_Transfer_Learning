from PIL import Image
import os
import numpy as np
import albumentations as A

transform = A.Compose([  
    A.HorizontalFlip(p=0.5),  
    A.ShiftScaleRotate(p=0.5),
    A.GridDistortion(),
    A.Rotate(limit=20, p=0.5),
    A.RandomBrightnessContrast(p=0.2),    
])
        
def agument_image_with_alubumation():
      DATA_DIR='/home/majid/Projects/ML/IoT/bobcatscraper/images/converted/'
      images = os.listdir(DATA_DIR)
      for i, image_name in enumerate(images):
         if (image_name.split('.')[1] == 'jpg'):
               print (image_name)
               image = np.asarray(Image.open(DATA_DIR + image_name))
               transformed1 = transform(image=image)
               image = transformed1['image']
               im = Image.fromarray(image)
               im.save(DATA_DIR + "1"+image_name)

               transformed2 = transform(image=image)
               image = transformed2['image']
               im = Image.fromarray(image)
               im.save(DATA_DIR + "2"+image_name)

def main():
   # agument_images("/home/majid/Projects/ML/IoT/bobcatscraper/images/converted/", "/home/majid/Projects/ML/IoT/bobcatscraper/images/converted/agumented")
   agument_image_with_alubumation()


if __name__ == "__main__":
    main()  
