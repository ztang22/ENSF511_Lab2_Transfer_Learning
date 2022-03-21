import requests


from selenium import webdriver
from selenium.webdriver.common.keys import Keys

def get_file_name(url):
    return url.split('/')[-1]

def get_image_urls(image_url):
    img_data = requests.get(image_url).content
    with open("images2/"+ get_file_name(image_url), 'wb') as handler:
     handler.write(img_data)


def main():
  
  bobcat_urls=[]
  for i in range(1,40):
    driver = webdriver.Chrome("/home/majid/Projects/ML/IoT/bobcatscraper/chromedriver")
    url='https://www.dreamstime.com/photos-images/bobcat.html?pg='+str(i)
    driver.get(url)
    elem = driver.find_element_by_class_name("dt-image-wall")
    pic_elements= elem.find_elements_by_class_name("dt-image")
    for element in pic_elements:
         get_image_urls(element.get_attribute("data-src"))
         bobcat_urls.append(element.get_attribute("data-src"))
    #write to file
    with open("bobcat_urls.txt", "w") as text_file:
        for url in bobcat_urls:
            text_file.write(url + "\n")
    driver.close()        
  
  


if __name__ == "__main__":
    main()  


