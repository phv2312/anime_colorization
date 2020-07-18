import torch
from torchvision.transforms import transforms

def create_train_transform(cfg):
    return transforms.Compose([
        transforms.Resize(size=(cfg['MODELS']['INPUT_HEIGHT'], cfg['MODELS']['INPUT_WEIGHT'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

def create_test_transform(cfg):
    return create_test_transform(cfg)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def to_device(x):
    return x.to(device)