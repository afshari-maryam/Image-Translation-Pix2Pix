#Import libraries
import tensorflow as tf
from matplotlib import pyplot as plt
import os

# Variable predefine
Dataset_PATH = '/Users/maryamafshari/Desktop/visidon-project/VD_dataset2'
BUFFER_SIZE = 400
# The batch size of 1 produced better results for the U-Net in the original pix2pix experiment
BATCH_SIZE = 1
# Each image is 256x256 in size
IMG_WIDTH = 256
IMG_HEIGHT = 256

#Functions
def resize(input_image, real_image, height, width):
  input_image = tf.image.resize(input_image, [height, width],
                                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
  real_image = tf.image.resize(real_image, [height, width],
                               method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

  return input_image, real_image

def random_crop(input_image, real_image):
  stacked_image = tf.stack([input_image, real_image], axis=0)
  cropped_image = tf.image.random_crop(
      stacked_image, size=[2, IMG_HEIGHT, IMG_WIDTH, 3])

  return cropped_image[0], cropped_image[1]

# Normalizing the images to [-1, 1]
def normalize(input_image, real_image):
  input_image = (input_image / 127.5) - 1
  real_image = (real_image / 127.5) - 1

  return input_image, real_image

@tf.function()
def random_jitter(input_image, real_image):
  # Resizing to 286x286
  input_image, real_image = resize(input_image, real_image, IMG_WIDTH, IMG_HEIGHT)

  # Random cropping back to 256x256
  input_image, real_image = random_crop(input_image, real_image)

  #normalize
  if tf.random.uniform(()) > 0.5:
    # Random mirroring
    input_image = tf.image.flip_left_right(input_image)
    real_image = tf.image.flip_left_right(real_image)

  return input_image, real_image

def load(image_file_input , image_file_target):
  # Read and decode an image file to a uint8 tensor
  input_image = tf.io.read_file(image_file_input)
  input_image = tf.io.decode_jpeg(input_image)

  target_image = tf.io.read_file(image_file_target)
  target_image = tf.io.decode_jpeg(target_image)

  # Convert both images to float32 tensors
  input_image = tf.cast(input_image, tf.float32)
  target_image = tf.cast(target_image, tf.float32)

  return input_image, target_image


def load_single_image(image_file_input):
        input_image = tf.io.read_file(image_file_input)
        input_image = tf.io.decode_jpeg(input_image)
        input_image = tf.cast(input_image, tf.float32)

        return input_image

def show_Image(input_image):
   # Casting to int for matplotlib to display the images
    plt.figure()
    plt.imshow(input_image / 255.0)
    plt.show()
   

def plot_two_normalized_images(image_input, image_target):
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(image_input/ 255.0)
    plt.title('Input Image')

    plt.subplot(1, 2, 2)
    plt.imshow(image_target/ 255.0)
    plt.title('Target Image ')

    plt.tight_layout()
    plt.show()

def plot_random_jitter(inp, tar):
    plt.figure(figsize=(6, 6))
    for i in range(4):
        rj_inp, rj_re = random_jitter(inp, tar)
        plt.subplot(2, 2, i + 1)
        plt.imshow(rj_inp / 255.0)
        plt.axis('off')
    plt.show()


def create_train_and_test_pairs(image_folder, train_ratio=0.8):
    # List all image files in the folder
    input_files = sorted([file for file in os.listdir(image_folder) if file.endswith('_input.png')])
    target_files = sorted([file for file in os.listdir(image_folder) if file.endswith('_target.png')])

    # Split the image files into train and test sets
    train_size = int(train_ratio * len(input_files))  # 80% for training

    train_input_files = input_files[:train_size]
    test_input_files = input_files[train_size:]
    train_target_files = target_files[:train_size]
    test_target_files = target_files[train_size:]

    train_pairs = []
    test_pairs = []

    train_pairs = list(map(lambda i: (os.path.join(image_folder,train_input_files[i]), os.path.join(image_folder,train_target_files[i])), range(len(train_input_files))))

    test_pairs = list(map(lambda i: (os.path.join(image_folder,test_input_files[i]), os.path.join(image_folder,test_target_files[i])), range(len(test_input_files))))
 
    #train_pairs = list(map(lambda i: (train_input_files[i], train_target_files[i]), range(len(train_input_files))))

    #test_pairs = list(map(lambda i: (test_input_files[i], test_target_files[i]), range(len(test_input_files))))


    return train_pairs, test_pairs

def create_image_pairs(image_folder, train_ratio=0.8):
    # List all image files in the folder
    image_files = sorted([file for file in os.listdir(image_folder) if file.endswith('.png')])

    # Split the image files into train and test sets
    train_size = int(train_ratio * len(image_files))  # 80% for training

    train_files = image_files[:train_size]
    test_files = image_files[train_size:]

    # Create the list of train image pairs
    train_pairs = [(os.path.join(image_folder, f"{filename.split('_')[0]}_input.png"),
                  os.path.join(image_folder, f"{filename.split('_')[0]}_target.png")) for filename in train_files]

    # Create the list of test image pairs
    test_pairs = [(os.path.join(image_folder, f"{filename.split('_')[0]}_input.png"),
                os.path.join(image_folder, f"{filename.split('_')[0]}_target.png")) for filename in test_files]

    return train_pairs, test_pairs

def load_image_train(image_file_input,image_file_target ):
  input_image, real_image = load(image_file_input,image_file_target )
  input_image, real_image = random_jitter(input_image, real_image)
  input_image, real_image = normalize(input_image, real_image)

  return input_image, real_image

def load_image_test(image_file_input,image_file_target ):
  input_image, real_image = load(image_file_input,image_file_target )
  input_image, real_image = resize(input_image, real_image,
                                   IMG_HEIGHT, IMG_WIDTH)
  input_image, real_image = normalize(input_image, real_image)

  return input_image, real_image

def process_pairs(pairs):
    pairs_input = []
    pairs_target = []
    for i in range(len(pairs)):
        pairs_input.append(pairs[i][0])
        pairs_target.append(pairs[i][1])
    
    return pairs_input, pairs_target


def prepare_dataset(pairs, load_func, BUFFER_SIZE, BATCH_SIZE):
    pairs_input, pairs_target = process_pairs(pairs)
    dataset_input = tf.data.Dataset.from_tensor_slices(pairs_input)
    dataset_target = tf.data.Dataset.from_tensor_slices(pairs_target)
    
    dataset_All = tf.data.Dataset.zip((dataset_input, dataset_target))
    mapped_dataset = dataset_All.map(load_func, num_parallel_calls=tf.data.AUTOTUNE)
    
    dataset = mapped_dataset.shuffle(BUFFER_SIZE)
    dataset = dataset.batch(BATCH_SIZE)
    
    return dataset

def get_train_and_test_dataset():
   #train_pairs, test_pairs = create_image_pairs(Dataset_PATH)
   train_pairs, test_pairs = create_train_and_test_pairs(Dataset_PATH)
   train_dataset = prepare_dataset(train_pairs, load_image_train, BUFFER_SIZE, BATCH_SIZE)
   test_dataset = prepare_dataset(test_pairs, load_image_test, BUFFER_SIZE, BATCH_SIZE)
   return train_dataset, test_dataset



train_dataset, test_dataset = get_train_and_test_dataset()
print("test dataset and train dataset is created")
print ("train datset = ")
print(len(train_dataset))
print ("test datset = ")
print(len(test_dataset))

def main():
    #check the load function  & show the example of input and target of an image in dataset folder
    input_path = Dataset_PATH + '/0028_input.png'
    target_path = Dataset_PATH + '/0028_target.png'
    inp, tar = load(input_path, target_path)
    #check the load function
    plot_two_normalized_images(inp, tar)
    #check the jitter function
    plot_random_jitter(inp, tar)
    print("Dataset has run completely")

if __name__ == '__main__':
    main()