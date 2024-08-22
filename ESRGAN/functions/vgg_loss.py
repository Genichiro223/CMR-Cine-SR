import torch.nn as nn
# from torchvision.models import vgg19
import torch
import numpy as np
import torchvision.models as models

def img_net_normalize(x):
    return torch.cat(((x-0.485)/0.229, (x-0.456)/0.224, (x-0.406)/0.225), 1)

class VGGLoss(nn.Module):
    def __init__(self):
        super().__init__()
        #self.vgg = vgg19(pretrained=True).features
        self.vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT)
        # 不更新参数
        for param in self.vgg.parameters():
            param.requires_grad = False
        self.loss = nn.MSELoss()
    def forward(self, input, target):
        vgg_input_features = self.vgg(img_net_normalize(input))
        vgg_target_features = self.vgg(img_net_normalize(target))
        return self.loss(vgg_input_features, vgg_target_features)
    
if __name__ == "__main__":
    a = torch.randn(1, 1, 128, 128)
    b = torch.randn(1, 1, 128, 128)
    lossnet = VGGLoss()
    loss = lossnet(a, b)
    print(loss)