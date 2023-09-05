import random

import numpy as np
import timm
import torch
import torch.nn as nn
import torchvision.datasets
import yaml
from easydict import EasyDict as edict
from torch.hub import load_state_dict_from_url
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
from torchvision.models import vit_b_16

try:
    from traffic_classification.flow.model import target_transform, DefenseModel, IMAGE_SIZE
except ImportError:
    from model import target_transform, DefenseModel, IMAGE_SIZE

try:
    from traffic_classification.utils import save_model
except ImportError:
    def save_model(model: nn.Module, path):
        if hasattr(model, "module"):
            state_dict = model.module.state_dict()
        else:
            state_dict = model.state_dict()
        torch.save(state_dict, path)


def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    # 读取配置
    config = edict(yaml.load(open('./config.yml', 'r', encoding="utf-8"), Loader=yaml.FullLoader))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(config)
    same_seeds(42)

    # 初始化模型，如果有GPU就用GPU，有几张卡就用几张
    model = timm.create_model('deit3_base_patch16_224_in21ft1k', pretrained=False, num_classes=2)
    state_dict = load_state_dict_from_url('https://dl.fbaipublicfiles.com/deit/deit_3_base_224_21k.pth', progress=True)
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    if config.trainer.muti_gpu and torch.cuda.device_count() > 1:
        print("使用{}个GPU训练".format(torch.cuda.device_count()))
        model = torch.nn.DataParallel(model)

    # 得到数据
    print('加载数据...')
    orig_set = torchvision.datasets.ImageFolder(root=config.defense_model.dataset_path, transform=target_transform)
    print(f'数据集数量{len(orig_set)}')
    # 拆分训练集和验证集
    dataset_length = len(orig_set)  # total number of examples
    num_test = int(config.trainer.valid_ratio * dataset_length)  # take ~10% for test
    total_idx = list(range(dataset_length))
    test_idx = random.sample(total_idx, num_test)
    train_idx = total_idx.copy()
    for value in test_idx:
        train_idx.remove(value)
    train_loader = torch.utils.data.DataLoader(torch.utils.data.Subset(orig_set, train_idx), batch_size=config.trainer.batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(torch.utils.data.Subset(orig_set, test_idx), batch_size=config.trainer.batch_size, shuffle=True)

    # 定义训练参数
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.trainer.lr)  # 优化器
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=(config.trainer.num_epochs / 10), eta_min=1e-5)
    writer = SummaryWriter()

    stale = 0
    best_acc = 0
    patience = config.trainer.num_epochs / 2
    # 开始训练
    print("开始训练！")
    for epoch in range(config.trainer.num_epochs):
        model.train()

        # 训练
        total_loss = 0
        train_pbar = tqdm(train_loader)
        for features, labels in train_pbar:
            optimizer.zero_grad()
            logits = model(features.to(device))
            loss = criterion(logits, labels.to(device))
            if torch.cuda.device_count() > 1:
                loss = torch.mean(loss)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()

            train_pbar.set_description(f'Epoch [{epoch + 1}/{config.trainer.num_epochs}] Training')

        # 验证
        model.eval()
        total_correct = 0
        vaild_bar = tqdm(valid_loader)
        for features, labels in vaild_bar:
            with torch.no_grad():
                logits = model(features.to(device))
            pred = torch.argmax(nn.Softmax(dim=1)(logits), dim=1)
            total_correct += torch.sum(pred == labels.to(device)).item()

            vaild_bar.set_description(f'Epoch [{epoch + 1}/{config.trainer.num_epochs}] Validation')

        # 计算loss和acc并保存到tensorboard
        train_loss = total_loss / (len(train_loader) * config.trainer.batch_size)
        val_acc = total_correct / (len(valid_loader) * config.trainer.batch_size)

        print(f"Epoch [{epoch + 1}/{config.trainer.num_epochs}] loss = {train_loss:.5f}, acc = {val_acc:.5f}")
        writer.add_scalar('Train Loss', train_loss, epoch)
        writer.add_scalar('Val Acc', val_acc, epoch)
        scheduler.step()
        if val_acc > best_acc:
            best_acc = val_acc
            print(f"Epoch [{epoch + 1}/{config.trainer.num_epochs}] 保存模型")
            save_model(model, 'deit3.pth')
            stale = 0
        else:
            stale += 1
            if stale > patience:
                print(f"连续的 {patience}  epochs 模型没有提升，停止训练")
                break
    print(f"最高acc: {best_acc:.5f}")
    # 模型最后验证
    model = Model(pretrained=False, num_classes=2).to(device)
    model.load_state_dict(torch.load(config.defense_model.checkpoint_path, map_location=torch.device('cpu')))
    model.to(device)
    # validation(model, [*valid_loader, *train_loader], config, device)
