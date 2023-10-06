# Image Translation using pix2pix in Tensorflow
## Translation of an input image to a target image.

                                                
                           In the Name of Allah, the Beneficent, the Merciful

This is the implementation of conditional GAN called Pix2Pix for Image translation using ```tensorflow``` of a dataset of image pairs . </br>
It is a dataset of 1207 paired images as input and target.</br>
I implemented a model in Tensorflow that performs image translation from the input images to the target images. </br>
An example of the input image and its target image is shown below : </br>
<img width="459" alt="image" src="https://github.com/afshari-maryam/Image-Translation-Pix2Pix/blob/main/Example_pair_2.png">
## About the pix2pix model and Image Translation: 
The Pix2Pix model and image translation techniques is to enable the generation of realistic images by learning the mapping between an input image and an output image. This mapping can be used for various tasks, such as image-to-image translation or image generation.</br>
Pix2Pix is a specific model architecture proposed in a research paper titled ``` "Image-to-Image Translation with Conditional Adversarial Networks" by Isola et al. ```. </br>
It uses a ```conditional generative adversarial network (cGAN)``` to learn the mapping between input and output images. .</br>
The cGAN framework consists of two main components: a ```generator``` network that generates the output image from the input image,.</br>
and a ```discriminator``` network that tries to distinguish between the generated output image and the real target image..</br>
The generator and discriminator are trained together in an adversarial manner, where the generator aims to fool the discriminator and produce realistic outputs, while the discriminator tries to correctly classify the real and generated images.</br>


## For Dependencies : 
Run  ``` pip install -r requirements.txt ``` script. <br /> 
Dependencies are : <br />
tensorflow==2.12.0 <br />
matplotlib==3.7.1 <br />
ipython==7.34.0 <br />
Pydub <br />

## For training : 
Run  ``` dataset.py ``` script. to make train_dataset and test_dataset. <br /> 
Run  ``` Model.py ``` script. to make model ready and will make some images from architecture of model. <br /> 
Run  ``` Loss.py ``` script. to make model loss . <br /> 
Run  ``` train.py ``` script. <br /> 
You can see the output images after 1k steps. The below image show the result after 39k steps of training:
![image](https://github.com/afshari-maryam/Image-Translation-Pix2Pix/blob/main/training_result.png)

## For making an inference (testing a single image )
Download the model from [here](https://drive.google.com/file/d/1BXT2ceCg9z38RCMmuTHPvB3Z7noFHjO7/view?usp=sharing).<br />
Model's name is ```ckpt-10```. <br />
Put the model in ``` /training_checkpoints ``` folder.<br />
Put the directory of your test image in ``` Make_Inference.py ```<br />
Run  ``` Make_Inference.py ``` on a test image.<br />
My model, designed to handle ```images with arbitrary size```.<br />
I have implemented a resizing mechanism that standardizes all input images to a consistent size.<br />
IMG_WIDTH = 256<br />
IMG_HEIGHT = 256<br />


## Prediction and Accuracy 
Run  ``` Prediction.py ``` script. to find some prediction result from test_dataset. <br /> 
Run  ``` Compute_accuracy.py ``` script. to find the accurcy ```Average Pixel-wise``` result. <br /> 
```Average Pixel-wise Accuracy: 0.78219396``` by ```50000 step```. <br /> 

## Results
These are the result of test model using test_dataset after trainig the model.
![image](https://github.com/afshari-maryam/Image-Translation-Pix2Pix/blob/main/pred_results/run%3A%203.png)
![image](https://github.com/afshari-maryam/Image-Translation-Pix2Pix/blob/main/pred_results/run%3A%205.png)


