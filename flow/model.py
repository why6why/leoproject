from torch import nn
from torchvision import transforms

IMAGE_SIZE = 224
CHANNEL = 3
RESIZE_SIZE = 224

target_transform = transforms.Compose([
    # transforms.Grayscale(),
    # transforms.Resize(size=(IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    # transforms.Normalize(mean=(0.16046377, 0.16045164, 0.16038571), std=(0.25324884, 0.25318655, 0.2531085))
])

resize_transform = transforms.Compose([
    transforms.Resize(size=(RESIZE_SIZE, RESIZE_SIZE))
])


class DefenseModel(nn.Module):
    def __init__(self, input_dim: int, num_classes: int):  # 初始化网络
        super(DefenseModel, self).__init__()
        self.input_dim = input_dim
        self.layers = nn.Sequential(
            nn.Linear(input_dim * input_dim, input_dim * input_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(input_dim * input_dim, input_dim * input_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(input_dim * input_dim, int(input_dim * input_dim / 2)),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(int(input_dim * input_dim / 2), 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = x.view(-1, self.input_dim * self.input_dim)
        x = self.layers(x)
        x = x.squeeze(1)  # (B, 1) -> (B)
        return x


class ClassificationModel(nn.Module):
    def __init__(self, input_dim: int, num_classes: int):  # 初始化网络
        super(ClassificationModel, self).__init__()
        self.input_dim = input_dim
        self.layers = nn.Sequential(
            nn.Linear(input_dim * input_dim, input_dim * input_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            # nn.Linear(input_dim * input_dim, input_dim * input_dim),
            # nn.ReLU(),
            # nn.Dropout(0.2),
            nn.Linear(input_dim * input_dim, int(input_dim * input_dim / 2)),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(int(input_dim * input_dim / 2), 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = x.view(-1, self.input_dim * self.input_dim)
        x = self.layers(x)
        x = x.squeeze(1)  # (B, 1) -> (B)
        return x
