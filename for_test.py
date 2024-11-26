import os
import math
import argparse
import json

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score
import torch.optim.lr_scheduler as lr_scheduler

from timm.models.vision_transformer import vit_large_patch16_224_in21k as create_model
from my_dataset import MyDataSet
from utils import read_test_data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-j", type=str, default='class_indices_Fa_VIT.json', help="class_indices")
    parser.add_argument("-n", type=int, default=157, help="type")
    parser.add_argument("-w", type=str, default='./weights/model-32-VIT-Sp.pth', help="type")
    # model-68-Ge-final.pth
    # model-67-Fa-final.pth'
    parser.add_argument('--data-path', type=str,
                        default="./datasets/test/Family")
    args = parser.parse_args()

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    img_size = {"s": [224, 224],  # train_size, val_size
                "m": [224, 224],
                "l": [224, 224]}
    num_model = "s"

    tb_writer = SummaryWriter()

    test_images_path, test_images_label = read_test_data(args.data_path)

    data_transform = transforms.Compose(
        [transforms.Resize(img_size[num_model][1]),
         transforms.CenterCrop(img_size[num_model][1]),
         transforms.ToTensor(),
         transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

    Predict = []
    for img_path in test_images_path:
        assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
        print(img_path)
        img = Image.open(img_path)
        img=img.convert('RGB')
        # RGB为彩色图片，L为灰度图片
        if img.mode != 'RGB' :
            raise ValueError("image: {} isn't RGB mode.".format(img_path))
        
        # [N, C, H, W]
        img = data_transform(img)

        # expand batch dimension
        img = torch.unsqueeze(img, dim=0)

        # read class_indict
        json_path = args.j
        assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

        json_file = open(json_path, "r")
        class_indict = json.load(json_file)

        # create model
        model = create_model(num_classes=args.n).to(device)
        # load model weights
        model_weight_path = args.w
        model.load_state_dict(torch.load(model_weight_path, map_location=device))
        model.eval()
        with torch.no_grad():
            # predict class
            output = torch.squeeze(model(img.to(device))).cpu()
            predict = torch.softmax(output, dim=0)
            predict_cla = torch.argmax(predict).numpy()
        Predict.append(predict_cla)

    test_acc = accuracy_score(test_images_label, Predict)


    every_class_acc = [0]*838
    every_class_num = [0]*838
    every_class_pred = [0]*838
    for i in range(len(test_images_label)):
        if test_images_label[i] == Predict[i]:
           every_class_num[test_images_label[i]] += 1
           every_class_pred[test_images_label[i]] += 1
           every_class_acc[test_images_label[i]] = every_class_pred[test_images_label[i]]/every_class_num[test_images_label[i]]
        if test_images_label[i] != Predict[i]:
           every_class_num[test_images_label[i]] += 1
           every_class_acc[test_images_label[i]] = every_class_pred[test_images_label[i]]/every_class_num[test_images_label[i]]

    for i in range(len(test_images_label)):
         print("class: {}   acc: {:.3}".format(test_images_label[i],
                                                 every_class_acc[test_images_label[i]]))
         with open('./res/test_0503_Sp.txt','w') as f:
              f.write('class:\t'+str(test_images_label[i])+'acc:\t'+str(test_images_label[i]))

 
    length = max(Predict)
    type_right = [0 for i in range(length+1)]
    type_num = [0 for i in range(length+1)]
    for i in range(len(Predict)):
        if Predict[i] == test_images_label[i]:
            type_right[Predict[i]] += 1
        type_num[Predict[i]] += 1
    
    type_acc = []
    for i in range(len(type_right)):
        if type_num[i]!= 0:
            type_acc.append(type_right[i]/type_num[i])

    with open('./res/test_0503_Fa.txt','w') as f:
        f.write('acc:\t'+str(test_acc)+'\nacc_type:\t'+str(type_acc))




if __name__ == '__main__':
    main()
