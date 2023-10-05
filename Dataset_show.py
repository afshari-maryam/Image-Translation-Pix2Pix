import tensorflow as tf
from matplotlib import pyplot as plt
from IPython import display


Dataset_PATH = '/Users/maryamafshari/Desktop/visidon-project/VD_dataset2'
BUFFER_SIZE = 400
# The batch size of 1 produced better results for the U-Net in the original pix2pix experiment
BATCH_SIZE = 1
# Each image is 256x256 in size
IMG_WIDTH = 256
IMG_HEIGHT = 256

#Functions 
def plot_two_images(image_input, image_target):
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(image_input)
    plt.title('Input Image')

    plt.subplot(1, 2, 2)
    plt.imshow(image_target)
    plt.title('Target Image ')

    plt.tight_layout()
    plt.show()


def plot_one_images(input_image, target_image):
    plt.figure()
    plt.imshow(input_image / 255.0)
    plt.figure()
    plt.imshow(target_image / 255.0)
    plt.show()


def main():
    #print an image of input
    sample_image_input = tf.io.read_file(str(Dataset_PATH + '/4816_input.png'))
    sample_image_input = tf.io.decode_jpeg(sample_image_input)
    print(sample_image_input.shape)

    #print an image of target
    sample_image_target = tf.io.read_file(str(Dataset_PATH + '/4816_target.png'))
    sample_image_target = tf.io.decode_jpeg(sample_image_target)
    print(sample_image_target.shape)

    #Show the two image
    plot_two_images(sample_image_input, sample_image_target)

if __name__ == '__main__':
    main()