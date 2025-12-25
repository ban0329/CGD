import torchvision
import torch.nn as nn
import timm



class mobilenet_v2(nn.Module):
    def __init__(self, dim, weights_path=None):
        super().__init__()
        # Use provided weights path or default relative path
        
        net_in = timm.create_model('mobilenetv2_100', num_classes=1000, in_chans=3, pretrained=True, checkpoint_path='')
        features = list(net_in.children())[:-2]
        features.append(nn.Conv2d(1280, dim, kernel_size=(1, 1), stride=(1, 1), bias=False))
        features.append(nn.ReLU(inplace=True))
        self.features = nn.Sequential(*features)
        print('>> Total network depth:{}'.format(len(self.features)))
        self.outputdim = dim

    def forward(self, x):
        x = self.features(x)
        return x

class efficientnet_b3(nn.Module):
    def __init__(self, dim):
        super().__init__()
        net_in = timm.create_model('tf_efficientnet_b3', num_classes=1000, in_chans=3, pretrained=True, checkpoint_path='')
        features = list(net_in.children())[:-2]
        features.append(nn.Conv2d(1536, dim, kernel_size=(1, 1), stride=(1, 1), bias=False))
        features.append(nn.ReLU(inplace=True))
        self.features = nn.Sequential(*features)
        print('>> Total network depth:{}'.format(len(self.features)))
        self.outputdim = dim

    def forward(self, x):
        x = self.features(x)
        return x







lightweight = {
    "mobilenet_v2": lambda dim, device=None, weights_path=None: mobilenet_v2(dim, weights_path=weights_path),
    "efficientnet_b3": lambda dim, device=None, weights_path=None: efficientnet_b3(dim),
}