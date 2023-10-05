import tensorflow as tf
from Dataset import *
from Model import *
from Train import *

# Create a function to calculate pixel-wise accuracy
def calculate_pixel_accuracy(y_true, y_pred):
    # Convert images to binary format
    y_true_binary = tf.cast(y_true > 0.5, tf.float32)
    y_pred_binary = tf.cast(y_pred > 0.5, tf.float32)

    # Calculate pixel-wise accuracy
    accuracy = tf.reduce_mean(tf.cast(tf.equal(y_true_binary, y_pred_binary), tf.float32))
    return accuracy



def main():
    # Iterate over the test dataset and calculate accuracy
    total_accuracy = 0
    num_samples = 0

    checkpoint_dir = './training_checkpoints'
    generator = Generator()
    checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

    for input_image, target_image in test_dataset:
        # Generate translated image using the model
        generated_image = generator(input_image, training=False)

        # Calculate accuracy for the generated image
        accuracy = calculate_pixel_accuracy(target_image, generated_image)

        # Accumulate accuracy and count number of samples
        total_accuracy += accuracy
        num_samples += 1

    # Calculate average accuracy
    average_accuracy = total_accuracy / num_samples

    # Print the average accuracy
    print("Average Pixel-wise Accuracy:", average_accuracy.numpy())


if __name__ == '__main__':
    main()