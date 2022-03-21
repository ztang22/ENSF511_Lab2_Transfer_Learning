from PIL import Image
import os
import PIL
import glob

# Resize all images in a directory
def resize_images(directory, width, height):
    for filename in os.listdir(directory):
        if filename.endswith(".jpg"):
            img = Image.open(directory + filename)
            img = img.resize((width, height), PIL.Image.ANTIALIAS)
            img.save(directory+"converted/" + filename)

def main():
    resize_images("/home/majid/Projects/ML/IoT/bobcatscraper/images/", 32, 32)


if __name__ == "__main__":
    main()  




