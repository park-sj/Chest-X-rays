from __future__ import print_function, division
import os
import pandas as pd
import warnings
from shutil import copyfile
warnings.filterwarnings("ignore")

root_dir = "./convert_image/"
target_dir = "./split_folder_by_target/"
csv_file = "./stage_2_train_labels.csv"

def split_folder_by_target():
    labels = pd.read_csv(csv_file)

    normal = 0
    abnormal = 0

    if not os.path.exists('./split_folder_by_target'):
        os.makedirs('./split_folder_by_target')
    else:
        if not os.path.exists('./split_folder_by_target/NORMAL'):
            os.makedirs('./split_folder_by_target/NORMAL')
        if not os.path.exists('./split_folder_by_target/PNEUMONIA'):
            os.makedirs('./split_folder_by_target/PNEUMONIA')

    for idx_root in enumerate(os.listdir(root_dir)):
        if(labels.iloc[idx_root[0], 5]):
            copyfile(os.path.join(root_dir, idx_root[1]), os.path.join(target_dir, 'PNEUMONIA', idx_root[1]))
            abnormal += 1
        else:
            copyfile(os.path.join(root_dir, idx_root[1]), os.path.join(target_dir, 'NORMAL', idx_root[1]))
            normal += 1

    print("PNEUMONIA: ", abnormal)
    print("NORMAL: ", normal)

def run():
    split_folder_by_target()

if __name__ == '__main__':
    run()


