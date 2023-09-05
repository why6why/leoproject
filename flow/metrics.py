import timm
import torch
import torchvision
from torchvision import transforms
from tqdm.auto import tqdm

from traffic_classification.flow.confusion_matrix import ConfusionMatrix
from traffic_classification.utils import download_model

# model = timm.create_model('deit3_base_patch16_224_in21ft1k', pretrained=False, num_classes=2)
model = timm.create_model('deit3_base_patch16_224_in21ft1k', pretrained=False, num_classes=2)
model_state_dict = 'http://172.22.121.63:23514/model/simulator/deit3_base_patch16_224-fd1cbc69.pth'
matrix = ConfusionMatrix(2, ['yes', 'no'], 'a')

model.load_state_dict(download_model(model_state_dict, None))
data_path = '../../../datasets/CIC-IDS2017/MachineLearningCVE/real/defense_dataset'
loader = torch.utils.data.DataLoader(torchvision.datasets.ImageFolder(root=data_path, transform=transforms.Compose([transforms.ToTensor()])), batch_size=256, shuffle=True)

device = torch.device('cuda')
model.to(device)
model = torch.nn.DataParallel(model)
model.eval()
for features, labels in tqdm(loader):
    with torch.no_grad():
        logits = model(features.to(device))
        _, predicted = torch.max(logits, 1)
    matrix.update(predicted.cpu().numpy(), labels.cpu().numpy())

print(matrix.summary())
