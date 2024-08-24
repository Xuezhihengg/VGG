from typing import Dict, List, Union
import torch
from torch import Tensor
from torch import nn

__all__ = [
    'VGG',
    'vgg11','vgg13','vgg16','vgg19',
    'vgg11_bn','vgg13_bn','vgg16_bn','vgg19_bn',
]

vgg_cfgs: Dict[str, List[Union[str, int]]] = {
    "vgg11": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    'vgg13': [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    'vgg16': [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    'vgg19': [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"]
}


def _make_layers(vgg_cfg: List[Union[str, int]], batch_norm: bool = False) -> nn.Sequential:
    layers: nn.Sequential[nn.Module] = nn.Sequential()
    in_channels = 3
    for v in vgg_cfg:
        if v == 'M':
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        else:
            layers.append(nn.Conv2d(in_channels, out_channels=v, kernel_size=3, stride=1, padding=1))
            if batch_norm:
                layers.append(nn.BatchNorm2d(v))
                layers.append(nn.ReLU())
            layers.append(nn.ReLU())
            in_channels = v
    return layers


class VGG(nn.Module):
    def __init__(self, vgg_cfg: List[Union[str, int]], batch_norm: bool = False, num_classes: int = 1000) -> None:
        super().__init__()
        self.features = _make_layers(vgg_cfg,batch_norm)
        self.avgpool = nn.AdaptiveAvgPool2d((7,7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7,4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096,4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096,num_classes),
        )
        self._initialize_weights()

    def forward(self,x: Tensor) -> Tensor:
        out = self.features(x)
        out = self.avgpool(out)
        out = torch.flatten(out,1)
        out = self.classifier(out)

        return out


    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module,nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.constant_(module.bias,0)
                elif isinstance(module,nn.BatchNorm2d):
                    nn.init.constant_(module.weight,1)
                    nn.init.constant_(module.bias,0)
                elif isinstance(module,nn.Linear):
                    nn.init.normal_(module.weight,0,0.01)
                    nn.init.constant_(module.bias,0)

def vgg11(**kwargs) -> VGG:
    return VGG(vgg_cfgs["vgg11"],False,**kwargs)

def vgg13(**kwargs) -> VGG:
    return VGG(vgg_cfgs["vgg13"],False,**kwargs)

def vgg16(**kwargs) -> VGG:
    return VGG(vgg_cfgs["vgg16"],False,**kwargs)

def vgg19(**kwargs) -> VGG:
    return VGG(vgg_cfgs["vgg19"],False,**kwargs)


def vgg11_bn(**kwargs) -> VGG:
    return VGG(vgg_cfgs["vgg11"],True,**kwargs)

def vgg13_bn(**kwargs) -> VGG:
    return VGG(vgg_cfgs["vgg13"],True,**kwargs)

def vgg16_bn(**kwargs) -> VGG:
    return VGG(vgg_cfgs["vgg16"],True,**kwargs)

def vgg19_bn(**kwargs) -> VGG:
    return VGG(vgg_cfgs["vgg19"],True,**kwargs)








