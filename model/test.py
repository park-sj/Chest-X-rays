import numpy as np
# %matplotlib inline
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torchvision import datasets, models, transforms
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from time import time
import glob
import random

num_classes = 2
evaluate_path = './train_val/validation/*'
class_label = {'NORMAL': 0, 'PNEUMONIA': 1}

class PredictImage(object):

    def __init__(self, model_path='', num_classes=num_classes):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # print(self.device)
        self.model_ft = models.resnet50(pretrained=True)
        self.model_ft.fc = nn.Linear(self.model_ft.fc.in_features, num_classes)
        #self.model_ft.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        self.model_ft = torch.load(model_path, map_location=torch.device('cpu'))
        self.model_ft.eval()
        self.model_ft.cuda()
        self.normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                  std=[0.2023, 0.1994, 0.2010])
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            self.normalize])

    def predict(self, img):
        inputs = self.transform(img).float()
        # inputs = Variable(inputs)
        inputs = inputs.unsqueeze(0)
        inputs = inputs.to(self.device)
        outputs = self.model_ft(inputs)
        _, preds = torch.max(outputs, 1)
        # print(outputs)
        return preds

    #def evaluate(self, path='./data/test_data_gen/*'):
    def evaluate(self, path=evaluate_path):

        """
        Evaluate model.
        Structure folder processed:
          processed/
          ├── 0/
          │   ├── image_01.jpg
          │   ├── image_02.jpg
          │   ├── image_03.jpg
          │   └── ...
          └── 1/
              ├── image_01.jpg
              ├── image_02.jpg
              ├── image_03.jpg
              └── ...
        """
        start = time()
        cnt_image = 0
        correct_pred = 0
        step = 0
        predictor = self

        for classes in glob.glob(path):
            cnt_image += len(glob.glob(classes + "/*"))
            #cur_class = int(classes.split('\\')[-1])
            cur_class = classes.split('\\')[-1]
            for img_path in glob.glob(classes + "/*"):
                #print('img_path', img_path)
                #print(str(img_path.split('/')[-1]) + " === " + str(correct_pred) + " === " + str(step))
                print(correct_pred)
                im = Image.open(img_path).convert('RGB')
                output = int(predictor.predict(im))
                if output == class_label[cur_class]:
                    correct_pred = correct_pred + 1
                step += 1

        print("Done evaluate in " + str(time() - start))
        print("Accuracy: " + str(correct_pred / cnt_image))


if __name__ == "__main__":
    # img = Image.open('./data/processed/0/41.jpg').convert('RGB')
    predictor = PredictImage(model_path='./train_val/save/Resnet50_057.pth', num_classes=num_classes)
    predictor.evaluate()