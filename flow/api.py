import base64
import math
import os
import random
import threading
from typing import Tuple, List

import PIL
import cv2
from torchsummary import summary
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import yaml
from easydict import EasyDict as edict
from matplotlib import pyplot as plt
from torch import nn
from torch.nn.modules.utils import _quadruple
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets.utils import download_and_extract_archive
from torchvision.models import vit_b_16, resnet18

try:
    from traffic_classification.flow.model import DefenseModel, ClassificationModel, IMAGE_SIZE, target_transform, CHANNEL, resize_transform

    from traffic_classification.flow.attack import PGDAttack, FGSMAttack, IFGSMAttack, MIFGSMAttack
except ImportError:
    from model import DefenseModel, ClassificationModel, IMAGE_SIZE, target_transform, CHANNEL

    from attack import PGDAttack, FGSMAttack, IFGSMAttack, MIFGSMAttack

try:
    from traffic_classification.utils import download_model
except ImportError:
    def download_model(download_path, save_path, check_hash=True):
        if download_path.startswith('http'):
            state_dict = torch.hub.load_state_dict_from_url(download_path, model_dir=save_path, check_hash=check_hash, map_location=torch.device('cpu'))
        else:
            state_dict = torch.load(download_path, map_location=torch.device('cpu'))
        return state_dict


class AdversarialAttackFlow(object):
    def __init__(self):
        # 读取配置
        if os.access('./traffic_classification/flow/config.yml', os.R_OK):
            self.config = edict(yaml.load(open('./traffic_classification/flow/config.yml', 'r', encoding="utf-8"), Loader=yaml.FullLoader))
        elif os.access('./config.yml', os.R_OK):
            self.config = edict(yaml.load(open('./config.yml', 'r', encoding="utf-8"), Loader=yaml.FullLoader))

        # 下载数据集
        download_and_extract_archive(self.config.trainer.sub_dataset, 'datasets', extract_root='datasets/flow_new', md5=self.config.trainer.sub_dataset_md5)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.device = torch.device('cpu')
        self.model = resnet18(pretrained=False, num_classes=len(self.config.model.names))
        self.model.load_state_dict(download_model(self.config.model.save_name, None))
        self.model.eval()

        dataset = torchvision.datasets.ImageFolder(root='datasets/flow_new', transform=target_transform)
        self.dataset = DataLoader(dataset, batch_size=1, shuffle=True)
        self.labels = self.config.model.names
        self.model = self.model.to(self.device)
        if self.device != torch.device("cpu") and torch.cuda.device_count() > 1:
            self.model = torch.nn.DataParallel(self.model)

        # 初始化防御模型
        self.defense_model = vit_b_16(pretrained=False, num_classes=2)
        self.defense_model.load_state_dict(download_model(self.config.defense_model.save_name, None))
        self.defense_model.eval()
        self.defense_model = self.defense_model.to(self.device)
        if self.device != torch.device("cpu") and torch.cuda.device_count() > 1:
            self.defense_model = torch.nn.DataParallel(self.defense_model)

        # 初始化攻击类
        self.PGDAttack = PGDAttack(self.model, self.device)
        self.FGSMAttack = FGSMAttack(self.model, self.device)
        self.IFGSMAttack = IFGSMAttack(self.model, self.device)
        self.MIFGSMAttack = MIFGSMAttack(self.model, self.device)

        self.warnup_models()

    def warnup_models(self):
        print('预热模型')
        classifier = torch.FloatTensor(np.random.random((self.config.trainer.batch_size, CHANNEL, IMAGE_SIZE, IMAGE_SIZE))).to(self.device)
        defense = torch.FloatTensor(np.random.random((self.config.trainer.batch_size, CHANNEL, IMAGE_SIZE, IMAGE_SIZE))).to(self.device)
        with torch.no_grad():
            self.model(classifier)
            self.defense_model(defense)

    def get_ori_image(self):
        data, label = self.dataset.__iter__().__next__()
        image = data.detach().cpu().numpy()
        label = label.detach().cpu().numpy()
        return image, label

    def get_ori_imag_list(self, number: int) -> Tuple[np.ndarray, np.ndarray]:
        ori_image_list: List[np.ndarray] = []
        ori_label_list: List[np.ndarray] = []
        for i in range(number):
            data, label = self.dataset.__iter__().__next__()
            image = data.detach().cpu().numpy()
            label = label.detach().cpu().numpy()
            ori_image_list.append(image)
            ori_label_list.append(label)
        return np.concatenate(ori_image_list, axis=0), np.concatenate(ori_label_list, axis=0)

    def random_image_attack(self, ori_image: np.ndarray, label) -> np.ndarray:
        method = random.choice(['PGD', 'FGSM', 'I-FGSM', 'MI-FGSM'])
        return self.image_attack(ori_image, label, method)

    def image_attack(self, ori_image: np.ndarray, label: np.ndarray, method: str) -> np.ndarray:
        ori_image_ = torch.FloatTensor(ori_image).to(self.device)
        label_ = torch.LongTensor(label).to(self.device)
        if method == 'PGD':
            return self.PGDAttack(ori_image_, label_).detach().cpu().numpy()
        elif method == 'MI-FGSM':
            return self.MIFGSMAttack(ori_image_, label_).detach().cpu().numpy()
        elif method == 'I-FGSM':
            return self.IFGSMAttack(ori_image_, label_).detach().cpu().numpy()
        else:
            return self.FGSMAttack(ori_image_, label_).detach().cpu().numpy()

    def process_image(self, image: np.ndarray):
        image = torch.FloatTensor(image).to(self.device)
        logits = self.model(image)
        pred = list(nn.Softmax(dim=1)(logits).detach().cpu().numpy()[0])
        return pred.index(max(pred)), float(max(pred))

    def process_image_list(self, image: np.ndarray) -> Tuple[List[int], List[float]]:
        label_list: List[int] = []
        confidence_list: List[float] = []
        image = torch.FloatTensor(image).to(self.device)
        with torch.no_grad():
            logits = self.model(image)
        pred_list = nn.Softmax(dim=1)(logits).detach().cpu().numpy()
        for pred in pred_list:
            label_list.append(list(pred).index(max(pred)))
            confidence_list.append(float(max(pred)))
        return label_list, confidence_list

    def defense_resnet(self, image: np.ndarray) -> Tuple[bool, float]:
        _image = torch.FloatTensor(image).to(self.device)
        logits = self.defense_model(_image)
        pred = list(nn.Softmax(dim=1)(logits).detach().cpu().numpy()[0])
        return bool(pred.index(max(pred))), float(max(pred))

    def defense_model_list(self, image) -> List[bool]:
        res: List[bool] = []
        image = torch.FloatTensor(image).to(self.device)
        with torch.no_grad():
            logits = self.defense_model(image)
        pred_list = torch.argmax(nn.Softmax(dim=1)(logits), dim=1).detach().cpu().numpy()
        for pred in pred_list:
            res.append(bool(pred))
        return res

    @staticmethod
    def defense(image: np.ndarray, method_range=5) -> np.ndarray:
        image_ = torch.FloatTensor(image)
        method = random.randint(0, method_range)
        if method == 0:
            # 水平翻转
            return transforms.RandomHorizontalFlip(1)(image_).numpy()
        elif method == 1:
            # 垂直翻转
            return transforms.RandomVerticalFlip(1)(image_).numpy()
        elif method == 2:
            # 同时水平翻转与垂直翻转
            shifted = transforms.RandomHorizontalFlip(1)(image_)
            return transforms.RandomVerticalFlip(1)(shifted).numpy()
        elif method == 3:
            # 高斯模糊
            return transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0))(image_).numpy()
        elif method == 4:
            # 填充后随机裁剪
            return transforms.RandomCrop(size=image.shape[-1])(image_).numpy()
        else:
            # 随机透视变换
            return transforms.RandomPerspective()(image_).numpy()

    @staticmethod
    def encode_image(image) -> str:
        return base64.b64encode(
            cv2.imencode('.jpg', (np.squeeze(image) * 255).astype(np.uint8).swapaxes(0, 1).swapaxes(1, 2))[1]).decode()

    @staticmethod
    def encode_image_list(image_list: np.ndarray) -> List[str]:
        res: List[str] = []
        for image in image_list:
            image = np.asarray(
                resize_transform(
                    (PIL.Image.fromarray((np.squeeze(image) * 255).astype(np.uint8).swapaxes(0, 1).swapaxes(1, 2)))))
            res.append(base64.b64encode(cv2.imencode('.jpg', image)[1]).decode())
        return res

    @staticmethod
    def imshow(image, title='test'):
        image_ = np.squeeze(image).swapaxes(0, 1).swapaxes(1, 2)
        plt.figure()
        plt.title(title)
        plt.axis('off')
        plt.imshow(image_)
        plt.show()

    @staticmethod
    def decode_image(image_base64: bytes) -> np.ndarray:
        return np.expand_dims((cv2.imdecode(np.asarray(bytearray(base64.b64decode(image_base64)), dtype=np.uint8), cv2.IMREAD_COLOR) / 255).swapaxes(0, 2), axis=0)


class FeatureSqueezingFlow(object):
    _instance_lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if not hasattr(FeatureSqueezingFlow, '_instance'):
            with FeatureSqueezingFlow._instance_lock:
                if not hasattr(FeatureSqueezingFlow, '_instance'):
                    FeatureSqueezingFlow._instance = object.__new__(cls)
                    FeatureSqueezingFlow._instance.init(args[0], args[1], args[2])
        return FeatureSqueezingFlow._instance

    def init(self, model, bit_depth=7, kernel_size=3):
        '''
        :param model: classifier model
        :param bit_depth: squeeze bit_depth, 位深度
        :param kernel_size: kernel_size
        '''

        assert 8 >= bit_depth > 0
        assert kernel_size > 0
        self.forget = 0.5
        self.model = model
        self.bit_depth = bit_depth
        self.kernel_size = kernel_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def squeezing(self, image):
        # image输入值
        self.image = image
        # 2**i-1
        self.bit_value = (2 ** self.bit_depth) - 1
        self.padding = _quadruple(math.floor(self.kernel_size / 2))
        # 为了减少到 i 位深度(1 ≤ i ≤ 7),我们首先将输入值乘以 (2**i−1)（减去 1，因为零值）
        # 然后四舍五入为整数.接下来我们缩放整数回到[0, 1],除以(2**i − 1).
        squeezed = torch.floor(self.image * self.bit_value) / self.bit_value
        squeezed = F.pad(squeezed, self.padding, mode='reflect')
        # unfold(dim, size, step) dim：展开维度，size：滑动窗口的大小，step：滑动窗口的步长
        squeezed = squeezed.unfold(2, self.kernel_size, 1).unfold(3, self.kernel_size, 1)
        # 中值平滑
        squeezed = squeezed.contiguous().view(squeezed.size()[:4] + (-1,)).median(dim=-1)[0]
        return squeezed

    def detect(self, image):
        '''
            将模型对原始样本的预测与压缩后对样本的预测进行比较,
            如果原始输入和压缩后的输入产生了与模型实质上不同的输出，
            则输入可能是对抗性的。

            :param image: the image that need to detect
            :return: True or False
        '''
        if self.device == torch.device("cuda"):
            image = torch.unsqueeze(torch.from_numpy(image).cuda(), dim=0)
        # 压缩输入样本
        squeezed = self.squeezing(image)

        # 原始输入产生的输出
        y_pred = self.model(image)
        y_pred = torch.argmax(y_pred)

        # 压缩后输入产出的输出
        y_defense_pred = self.model(squeezed)
        y_defense_pred = torch.argmax(y_defense_pred)

        # 判断两输出是否相同
        if y_pred.item() == y_defense_pred.item():
            # print("检测为正常样本")
            return False
        else:
            # print("检测为对抗样本")
            return True


if __name__ == '__main__':
    adversarial_attack_flow = AdversarialAttackFlow()
    adversarial_detect_flow = FeatureSqueezingFlow(adversarial_attack_flow.model, 7, 3)
    summary(adversarial_attack_flow.model.cuda(), input_size=(3, 224, 224), batch_size=-1)
    NUMBER = 1000

    defense_success_count = 0
    attack_success_count = 0
    _oir_label_list = []
    _pre_label_list = []
    _attack_label_list = []
    _defense_label_list = []
    _attack_confidence_list = []
    for method in ['PGD', 'FGSM', 'I-FGSM', 'MI-FGSM']:
        # 从数据集中拿出原始图片和原始图片类别
        ori_image_list, ori_label_list = adversarial_attack_flow.get_ori_imag_list(int(NUMBER / 4))

        # 对原始图片进行攻击，得到攻击后的图片
        attack_img_list = adversarial_attack_flow.image_attack(ori_image_list, ori_label_list, method)

        # 处理攻击后的图片，得到攻击后的类别
        attack_label, attack_confidence = adversarial_attack_flow.process_image_list(attack_img_list)
        pre_label, pre_confidence = adversarial_attack_flow.process_image_list(ori_image_list)

        # 处理防御后的图片，得到防御后的类别
        defense_result = adversarial_attack_flow.defense_model_list(attack_img_list)

        adversarial_attack_flow.encode_image_list(attack_img_list)

        _oir_label_list.extend(ori_label_list)
        _pre_label_list.extend(pre_label)
        _attack_label_list.extend(attack_label)
        _defense_label_list.extend(defense_result)
        _attack_confidence_list.extend(attack_confidence)

    for i in range(NUMBER):
        if _defense_label_list[i]:
            defense_success_count += 1
        if _oir_label_list[i] != _attack_label_list[i]:
            attack_success_count += 1
        print(f'数据集类别 {adversarial_attack_flow.labels[_oir_label_list[i]]}'
              f'\t\t\t\t模型分类  {adversarial_attack_flow.labels[_pre_label_list[i]]}'
              f'\t\t\t\t攻击后分类 {adversarial_attack_flow.labels[_attack_label_list[i]]}'
              f'\t\t\t\t防御结果 {_defense_label_list[i]}'
              f'\t攻击分类置信度 {_attack_confidence_list[i]} ')
    print(f'防御成功数量{defense_success_count}')
    print(f'攻击成功数量{attack_success_count}')
