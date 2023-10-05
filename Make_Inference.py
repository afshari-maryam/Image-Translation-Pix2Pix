import tensorflow as tf
from Dataset import *
from Model import *
from Train import *

Image_PATH = '/Users/maryamafshari/Desktop/visidon-project/VD_dataset2'
#Image_PATH ='/Users/maryamafshari/Desktop/visidon-project/mySrcCode/conditionalGAN/Sample_img/'
Inference_dir = "inference"
Image_input_name = '/0480_input.png'
#Image_input_name ='3.png'
#Image_target_name = '/0480_target.png'

def show_Image(input_image):
   # Casting to int for matplotlib to display the images
    plt.figure()
    plt.imshow(input_image)
    plt.show()

def preprocess_image(input_path):
    img = tf.io.read_file(input_path)
    img = tf.io.decode_jpeg(img)
    img = tf.cast(img, tf.float32)
    #resize
    img = tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH],
                                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    #Normalize
    img = (img / 127.5) - 1
    return img

def generate_output_images(model, test_input):
  # Reshape input image to match the expected shape
  test_input = tf.expand_dims(test_input, axis=0)
  prediction = model(test_input, training=False)
  plt.figure(figsize=(15, 5))

  display_list = [test_input[0],  prediction[0]]
  title = ['Input Image',  'Predicted Image']

  for i in range(2):
    plt.subplot(1, 2, i+1)
    plt.title(title[i])
    # Getting the pixel values in the [0, 1] range to plot.
    plt.imshow(display_list[i] * 0.5 + 0.5)
    plt.axis('off')
  plt.show()
  return prediction

def generate_and_save_output_image(model, test_input, save_path):
    # Reshape input image to match the expected shape
    test_input = tf.expand_dims(test_input, axis=0)
    prediction = model(test_input, training=False)
    plt.figure(figsize=(5, 5))

    display_list = [ prediction[0]]
    title = [ 'Predicted Image']

    for i in range(1):
        plt.subplot(1, 1, i+1)
        plt.title(title[i])
        # Getting the pixel values in the [0, 1] range to plot.
        plt.imshow(display_list[i] * 0.5 + 0.5)
        plt.axis('off')

    plt.savefig(save_path)
    plt.close()
    


def main():
    #load and preprocess data -----------------------
    sample_image_input = tf.io.read_file(str(Image_PATH + Image_input_name))
    sample_image_input = tf.io.decode_jpeg(sample_image_input)
    sample_image_input = tf.cast(sample_image_input, tf.float32)
    #resize
    sample_image_input = tf.image.resize(sample_image_input, [IMG_HEIGHT, IMG_WIDTH],
                                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    #Normalize
    sample_image_input = (sample_image_input / 127.5) - 1
    print("data is prepared")
    #sample_image_target = str(Image_PATH + Image_target_name)
    #sample_image_target = preprocess_image(sample_image_target )
    
    # Load Model --------------------------
    checkpoint_dir = './training_checkpoints'
    generator = Generator()
    checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
    output = generate_output_images(generator, sample_image_input)
    print("Output shape of Image = ")
    print(output.shape)
    if not os.path.exists(Inference_dir):
        # Create the result directory
        os.makedirs(Inference_dir, exist_ok=True)
    result_Inference_path = os.path.join(Inference_dir, "Prediction.png")
    generate_and_save_output_image(generator, sample_image_input, result_Inference_path)
    print(f"The tranlated image of {Image_input_name } is saved in {result_Inference_path}. ")

if __name__ == '__main__':
    main()