import os
import random
from shutil import copyfile

img_dir = "./split_folder_by_target"

def img_train_validation_split(img_source_dir, train_size):
    """
    Randomly splits images over a train and validation folder, while preserving the folder structure

    Parameters
    ----------
    img_source_dir : string
        Path to the folder with the images to be split. Can be absolute or relative path

    train_size : float
        Proportion of the original images that need to be copied in the subdirectory in the train folder
    """
    if not (isinstance(img_source_dir, str)):
        raise AttributeError('img_source_dir must be a string')

    if not os.path.exists(img_source_dir):
        raise OSError('img_source_dir does not exist')

    if not (isinstance(train_size, float)):
        raise AttributeError('train_size must be a float')

    # Set up empty folder structure if not exists
    if not os.path.exists('train_val'):
        os.makedirs('train_val')
    else:
        if not os.path.exists('./train_val/train'):
            os.makedirs('./train_val/train')
        if not os.path.exists('./train_val/validation'):
            os.makedirs('./train_val/validation')

    # Get the subdirectories in the main image folder
    subdirs = [subdir for subdir in os.listdir(img_source_dir) if os.path.isdir(os.path.join(img_source_dir, subdir))]

    for subdir in subdirs:
        subdir_fullpath = os.path.join(img_source_dir, subdir)
        if len(os.listdir(subdir_fullpath)) == 0:
            print(subdir_fullpath + ' is empty')
            break

        train_subdir = os.path.join('./train_val/train', subdir)
        validation_subdir = os.path.join('./train_val/validation', subdir)

        # Create subdirectories in train and validation folders
        if not os.path.exists(train_subdir):
            os.makedirs(train_subdir)

        if not os.path.exists(validation_subdir):
            os.makedirs(validation_subdir)

        train_counter = 0
        validation_counter = 0

        # Randomly assign an image to train or validation folder
        for filename in os.listdir(subdir_fullpath):

            if filename.endswith(".jpg") or filename.endswith(".png"):
                fileparts = filename.split('.')

                if random.uniform(0, 1) <= train_size:
                    copyfile(os.path.join(subdir_fullpath, filename),
                             os.path.join(train_subdir, fileparts[0] + '.' + fileparts[1]))
                    train_counter += 1
                else:
                    copyfile(os.path.join(subdir_fullpath, filename),
                             os.path.join(validation_subdir, fileparts[0] + '.' + fileparts[1]))
                    validation_counter += 1

        print('Copied ' + str(train_counter) + ' images to ./train_val/train/' + subdir)
        print('Copied ' + str(validation_counter) + ' images ./train_val/validation/' + subdir)

def run():
    img_train_validation_split(img_dir, 0.9)

if __name__ == '__main__':
    run()