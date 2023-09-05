import os
import random
import time

import numpy as np
import torchvision
import yaml

from easydict import EasyDict as edict

from tqdm.auto import tqdm
from torch.utils.data import Subset
from torchvision.models import resnet18
import torch
from torchvision.transforms import transforms
from traffic_classification.flow.attack import PGDAttack, FGSMAttack, IFGSMAttack, MIFGSMAttack
from traffic_classification.flow.model import target_transform

from traffic_classification.utils import download_model

config = edict(yaml.load(open('../config.yml', 'r', encoding="utf-8"), Loader=yaml.FullLoader))
labels = config.model.names
batch_size = config.trainer.batch_size
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = resnet18(pretrained=False, num_classes=len(config.model.names))
model.load_state_dict(download_model(config.model.save_name, None))
model.eval()
model = model.to(device)
if config.trainer.muti_gpu and torch.cuda.device_count() > 1:
    print("使用{}个GPU".format(torch.cuda.device_count()))
    model = torch.nn.DataParallel(model)

orig_set = torchvision.datasets.ImageFolder(root='../../../../datasets/CIC-IDS2017/MachineLearningCVE/real/image', transform=target_transform)
to_pil_image = transforms.ToPILImage()
PGD_attack = PGDAttack(model, device)
FGSM_attack = FGSMAttack(model, device)
IFGSM_attack = IFGSMAttack(model, device)
MIFGSM_attack = MIFGSMAttack(model, device)

for label in labels:
    path = f'../../../../datasets/CIC-IDS2017/MachineLearningCVE/real/image/{label}/'
    save_path = f'../../../../datasets/CIC-IDS2017/MachineLearningCVE/real/image_defense/{label}'
    pcap_file_list = os.listdir(path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # 找到对应类别
    sub_dataset = Subset(orig_set, [i for i in range(len(orig_set)) if orig_set.imgs[i][1] == orig_set.class_to_idx[label]])
    subset_loader = torch.utils.data.DataLoader(sub_dataset, batch_size=config.trainer.batch_size, shuffle=True)

    # 循环随机攻击
    for image, label_ in tqdm(subset_loader):
        method = random.choice(['PGD', 'FGSM', 'I-FGSM', 'MI-FGSM'])
        if method == 'PGD':
            attack_image = PGD_attack(image, label_)
        elif method == 'MI-FGSM':
            attack_image = MIFGSM_attack(image, label_)
        elif method == 'I-FGSM':
            attack_image = IFGSM_attack(image, label_)
        else:
            attack_image = FGSM_attack(image, label_)

        # 攻击过后保存图片
        for save_image in attack_image:
            save_image_ = to_pil_image(save_image)
            # plt.figure()
            # plt.title('title')
            # plt.axis('off')
            # plt.imshow(save_image_)
            # plt.show()
            save_name = f'{save_path}/{int(time.time()):03d}-{int(random.random() * 10000):04d}-{method}.png'
            save_image_.save(save_name)
