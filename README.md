# Image Translation using Pix2Pix
# Image translation
## Translation of an input image to a target image.
In the name of GOD </br>

This is the implementation of conditional GAN called Pix2Pix for Image translation of some data. </br>
I implemented a model in Pytorch that performs image translation from the input images to the target images. An example of the input image and its target image is shown below</br>
<img width="459" alt="image" src="https://user-images.githubusercontent.com/31028574/205402113-856844ba-d393-46a2-82a4-6f7ad8c6bf9e.png">
## For Dependencies : 
Run  ``` pip install -r requirements.txt ``` script. to make train_dataset and test_dataset. <br /> 

## For training : 
Run  ``` dataset.py ``` script. to make train_dataset and test_dataset. <br /> 
Run  ``` Model.py ``` script. to make model ready. <br /> 
Run  ``` Loss.py ``` script. to make model loss . <br /> 
Run  ``` train.py ``` script. <br /> 
Dependencies are : <br />
tensorflow==2.12.0 <br />
matplotlib==3.7.1 <br />
ipython==7.34.0 <br />
Pydub <br />

## For making an inference (testing a single image )
Download the model from [here](https://tuni-my.sharepoint.com/:u:/g/personal/sheyda_ghanbaralizadehbahnemiri_tuni_fi/EefhTnBnXmlPgWGjU9seFfkBArrboa-Zocw9v7xqPnRsAQ?e=WNf0AO). 
Model's name is ``` ckpt-10 ``` <br />
Put the model in ``` ckpt-10 ``` folder<br />
Change the directory and name of image in ``` Make_Inference.py ```<br />
Run  ``` Make_Inference.py ``` on a test image<br />

Dataset is provided by 

## Results
![image](https://user-images.githubusercontent.com/31028574/205401871-ad4169c6-cdc8-4712-8a68-a540026e01f9.png)
![image](https://user-images.githubusercontent.com/31028574/205401882-bc9531eb-bf7c-4512-b5ef-96c9294ede55.png)

Results : </br>

Train: </br>

Inference:  </br>

