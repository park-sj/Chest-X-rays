import cv2
from PIL import Image
import torch.nn.functional as F
import numpy as np
import torchvision.transforms as transforms
from torch.autograd import Variable

classes = {0: 'NORMAL', 1: 'PNEUMONIA'}
num_classes = len(classes)

# generate class activation mapping for the top1 prediction
def returnCAM(feature_conv, weight_softmax, class_idx):
    # generate the class activation maps upsample to 256x256
    size_upsample = (256, 256)
    bz, nc, h, w = feature_conv.shape
    output_cam = []
    for idx in class_idx:
        cam = weight_softmax[class_idx].dot(feature_conv.reshape((nc, h*w)))
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)
        output_cam.append(cv2.resize(cam_img, size_upsample))
    return output_cam

def get_cam(net, features_blobs, img_pil, classes, root_img):
    params = list(net.parameters())
    weight_softmax = np.squeeze(params[-2].data.cpu().numpy())

    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010]
    )
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        #normalize
    ])

    img_tensor = preprocess(img_pil)
    img_variable = Variable(img_tensor.unsqueeze(0)).cuda()
    logit = net(img_variable)


    h_x = F.softmax(logit, dim=1).data.squeeze()
    probs, idx = h_x.sort(0, True)

    # output: the prediction
    for i in range(0, 2):
        line = '{:.3f} -> {}'.format(probs[i], classes[idx[i].item()])
        print(line)

    CAMs = returnCAM(features_blobs[0], weight_softmax, [idx[0].item()])

    # render the CAM and output
    print('output CAM.jpg for the top1 prediction: %s' % classes[idx[0].item()])
    img = cv2.imread(root_img)
    height, width, _ = img.shape
    CAM = cv2.resize(CAMs[0], (width, height))
    heatmap = cv2.applyColorMap(CAM, cv2.COLORMAP_JET)
    result = heatmap * 0.3 + img * 0.5
    cv2.imwrite('cam.jpg', result)

features_blobs = []

def hook_feature(module, input, output):
    features_blobs.append(output.data.cpu().numpy())

final_conv = 'layer4'

def run():
    model_ft = models.resnet50
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, num_classes)
    model_ft = model_ft.to(device)
    model_ft.cuda()
    model_ft = torch.load('./temp')

    model_ft._modules.get(final_conv).register_forward_hook(hook_feature)

    root = 'target image'
    img = Image.open(root)
    img = img.convert('RGB')
    get_cam(model_ft, features_blobs, img, classes, root)

if __name__ == '__main__':
    # images = torch.rand(1, 3, 224, 224).cuda(0)
    # model = res2net50_v1b_26w_4s(pretrained=True)
    # model = model.cuda(0)
    # print(model(images).size())
    run()
