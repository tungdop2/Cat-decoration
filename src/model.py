import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class Cat(nn.Module):
    def __init__(self):
        super(Cat, self).__init__()
        self.model = models.resnet18(pretrained=True)
        self.model.fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.model.fc.in_features, 256),
            nn.Linear(256, 18),
        )

    def forward(self, x):
        x = self.model(x)
        return x

def get_model(deivce):
    
    model = Cat()
    model.load_state_dict(torch.load('weights/cat_model1.pt', map_location=torch.device('cpu')))
    model.eval()

    return model