import torch
import torch.nn as nn
from torch.autograd import Function
try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}

# AlexNet modified version with DANN implementation
class ReverseLayerF(Function):
    # Forwards identity
    # Sends backward reversed gradients
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None

class DANN_AlexNet(nn.Module):

    def __init__(self, num_classes=1000):
        super(DANN_AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )
        # domain classifier part
        self.GD = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 2),
        )

    def forward(self, x, alpha=None):
        features = self.features(x)
        # Flatten the features:
        features = features.view(features.size(0), -1)
        # If we pass alpha, we can assume we are training the discriminator
        if alpha is not None:
            # gradient reversal layer (backward gradients will be reversed)
            reverse_feature = ReverseLayerF.apply(features, alpha)
            discriminator_output = self.GD(reverse_feature)
            return discriminator_output
        # If we don't pass alpha, we assume we are training with supervision
        else:
            class_outputs = self.classifier(features)
            return class_outputs

def dann_net(pretrained=False, progress=True, n_classes=7, **kwargs):
    """AlexNet model architecture 
    Args:
        pretrained (bool):  If True, returns a model pre-trained on ImageNet
        progress (bool):    If True, displays a progress bar of the download to stderr
    """
    
    net = DANN_AlexNet(**kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['alexnet'], progress=progress)
        net.load_state_dict(state_dict, strict = False)

        # Change output classes in last layer
        net.classifier[6] = nn.Linear(4096, n_classes)
        net.GD[6] = nn.Linear(4096, 2)

        # Copy pretrained weights from the classifier to the domain_classifier
        net.GD[1].weight.data = net.classifier[1].weight.data.clone()
        net.GD[1].bias.data = net.classifier[1].bias.data.clone()

        net.GD[4].weight.data = net.classifier[4].weight.data.clone()
        net.GD[4].bias.data = net.classifier[4].bias.data.clone()
        
    return net