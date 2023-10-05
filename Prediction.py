import tensorflow as tf
#from Dataset import *
from Dataset import *
from Model import *
from Train import *

result_pred_dir = "pred_results"
def main():
    print("prediction is here by the latest checkpoint.")
    #checkpoint_dir = './training_checkpoints'
    checkpoint_dir = './checkpoint2'
    generator = Generator()
    checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
    # Run the trained model on a few examples from the test set
    for inp, tar in test_dataset.take(2):
        generate_images(generator, inp, tar)

    
    # Run the trained model on a few examples from the test set

        # Check if the result directory already exists
    if not os.path.exists(result_pred_dir):
        # Create the result directory
        os.makedirs(result_pred_dir, exist_ok=True)
    i=0
    for inp, tar in test_dataset.take(5):
        i = i+1
        #generate_images(generator, inp, tar)
        save_train_result_path = os.path.join(result_pred_dir, f"run: {i}.png")
        #save_train_result_path = str(result_pred_dir + f"run: {i}.png")
        generate_and_save_images(generator, inp, tar,save_train_result_path)
        print(i)
    print("the result is saved")
    '''
    checkpoint_dir = './training_checkpoints'
    checkpoint_path = os.path.join(checkpoint_dir, ".h5-2")
    checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                    discriminator_optimizer=discriminator_optimizer,
                                    generator=generator,
                                    discriminator=discriminator)
    # Restoring the latest checkpoint in checkpoint_dir
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_path))
    # Restore the model weights and optimizer state
    #checkpoint_path = 'checkpoint_dir/model_checkpoint'
    #generator.load_weights(checkpoint_path)

    # Run the trained model on a few examples from the test set
    for inp, tar in test_dataset.take(5):
        generate_images(generator, inp, tar)'''

    '''checkpoint_path = os.path.join(checkpoint_dir, '.h5-2')
    # Restoring the latest checkpoint in checkpoint_dir
    #checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
    # Restore the model weights and optimizer state
    m =generator.load_weights(checkpoint_path)'''




    '''print("Prediction using the latest checkpoint:")
    checkpoint_dir = './training_checkpoints'
    checkpoint_path = os.path.join(checkpoint_dir, 'ckpt-2')
    
    # Restore the model weights and optimizer state from the specific checkpoint
    checkpoint.restore(checkpoint_path)
    generator.load_weights(checkpoint_path)

    # Run the trained model on a few examples from the test set

        # Check if the result directory already exists
    if not os.path.exists(result_pred_dir):
        # Create the result directory
        os.makedirs(result_pred_dir, exist_ok=True)

    for inp, tar in test_dataset.take(5):
        i = i+1
        #generate_images(generator, inp, tar)
        save_train_result_path = str(result_pred_dir + f"run: {i}.png")
        generate_and_save_images(generator, inp, tar,save_train_result_path)
        print(i)
        print("the result is saved")'''
    
if __name__ == '__main__':
    main()