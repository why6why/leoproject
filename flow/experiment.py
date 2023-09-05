import torchvision
import torch
from easydict import EasyDict as edict
from torch import nn
from torch.utils.data import Subset
from torchvision.transforms import transforms
import yaml
from tqdm.auto import tqdm

try:
    from traffic_classification.flow.api import AdversarialAttackFlow, target_transform
except ImportError:
    from api import AdversarialAttackFlow, target_transform


def process_one_method(ori_label: str, method: str, dataset: torch.utils.data.DataLoader, adversarial_attack_flow: AdversarialAttackFlow, config):
    result_map = {'ori_label': ori_label, 'method': method, 'total': len(dataset.dataset), 'defense_success_count': 0}
    for i in range(len(config.model.names)):
        result_map[config.model.names[i]] = 0

    for image, label in tqdm(dataset):
        image = image.detach().cpu().numpy()
        label = label.detach().cpu().numpy()
        # 图片攻击
        attack_img = adversarial_attack_flow.image_attack(image, label, method)
        attack_img = torch.FloatTensor(attack_img).to(adversarial_attack_flow.device)
        with torch.no_grad():
            logits = adversarial_attack_flow.model(attack_img)
        pred = torch.argmax(nn.Softmax(dim=1)(logits), dim=1)
        for index in list(pred.detach().cpu().numpy()):
            result_map[adversarial_attack_flow.labels[index]] += 1
        # 图片防御
        with torch.no_grad():
            logits = adversarial_attack_flow.defense_model(attack_img)
            # logits = adversarial_attack_flow.defense_model(torch.FloatTensor(image).to(adversarial_attack_flow.device))
        pred = torch.argmax(nn.Softmax(dim=1)(logits), dim=1)
        defense_success_count = 0
        for index in list(pred.detach().cpu().numpy()):
            if index:
                defense_success_count += 1
        result_map['defense_success_count'] += defense_success_count
    return result_map


if __name__ == '__main__':
    # 读取配置
    config = edict(yaml.load(open('./config.yml', 'r', encoding="utf-8"), Loader=yaml.FullLoader))

    # 得到数据
    print('加载数据...')
    orig_set = torchvision.datasets.ImageFolder(root=config.trainer.dataset_path, transform=target_transform)
    adversarial_attack_flow = AdversarialAttackFlow()

    result = []
    for label in config.model.names:
        # 找到对应类别
        sub_dataset = Subset(orig_set, [i for i in range(len(orig_set)) if orig_set.imgs[i][1] == orig_set.class_to_idx[label]])
        subset_loader = torch.utils.data.DataLoader(sub_dataset, batch_size=config.trainer.batch_size, shuffle=True)
        for method in ["FGSM", "I-FGSM", 'MI-FGSM', 'PGD']:
            result_map = process_one_method(label, method, subset_loader, adversarial_attack_flow, config)
            print(result_map)
            result.append(result_map)
    with open('result.csv', 'w') as f:
        for res in result:
            f.write(str(res))
