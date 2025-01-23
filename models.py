import torch
import torch.nn as nn
from torchvision.models import resnet18

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 3 * 3, 128)
        self.fc2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(-1, 64 * 3 * 3)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    
class CNN_Headless(nn.Module):
    def __init__(self):
        super(CNN_Headless, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 3 * 3, 128)
        self.fc2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(-1, 64 * 3 * 3)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        return x

class SmallCNN(nn.Module):
    def __init__(self):
        super(SmallCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 7 * 7, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 32 * 7 * 7)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
class SmallCNN_Headless(nn.Module):
    def __init__(self):
        super(SmallCNN_Headless, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 7 * 7, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 32 * 7 * 7)
        x = torch.relu(self.fc1(x))
        return x

class TinyCNN(nn.Module):
    def __init__(self):
        super(TinyCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(16 * 8 * 8, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 10)

    def forward(self, x):
        # x = self.pool(torch.relu(self.conv1(x)))
        # x = self.pool(torch.relu(self.conv2(x)))
        # x = x.view(-1, 16 * 7 * 7)
        # x = torch.relu(self.fc1(x))
        # x = torch.relu(self.fc2(x))
        # x = self.fc3(x)
        # return x

        layer_outputs = []
        
        # Conv1 layer
        conv1_out = self.conv1(x)
        layer_outputs.append(conv1_out)
        
        # ReLU after Conv1
        relu1_out = torch.relu(conv1_out)
        layer_outputs.append(relu1_out)
        
        # Pool1 layer
        pool1_out = self.pool(relu1_out)
        layer_outputs.append(pool1_out)
        
        # Conv2 layer
        conv2_out = self.conv2(pool1_out)
        layer_outputs.append(conv2_out)
        
        # ReLU after Conv2
        relu2_out = torch.relu(conv2_out)
        layer_outputs.append(relu2_out)
        
        # Pool2 layer
        pool2_out = self.pool(relu2_out)
        layer_outputs.append(pool2_out)
        
        # Flatten
        flattened = pool2_out.view(-1, 16 * 8 * 8)
        layer_outputs.append(flattened)
        
        # Fully connected layer 1
        fc1_out = self.fc1(flattened)
        layer_outputs.append(fc1_out)
        
        # ReLU FC layer 1
        fc1_relu_out = torch.relu(fc1_out)
        layer_outputs.append(fc1_relu_out)

        # Fully connected layer 2
        fc2_out = self.fc2(fc1_relu_out)
        layer_outputs.append(fc2_out)
        
        # ReLU FC layer 2
        fc2_relu_out = torch.relu(fc2_out)
        layer_outputs.append(fc2_relu_out)

        # final layer
        final_out = self.fc3(fc2_relu_out)
        layer_outputs.append(final_out)
        
        return layer_outputs

    
class TinyCNN_Headless(nn.Module):
    def __init__(self):
        super(TinyCNN_Headless, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(16 * 8 * 8, 32)

    def forward(self, x):
        # x = self.pool(torch.relu(self.conv1(x)))
        # x = self.pool(torch.relu(self.conv2(x)))
        # x = x.view(-1, 16 * 7 * 7)
        # x = torch.relu(self.fc1(x))
        # return x

        layer_outputs = []
        
        # Conv1 layer
        conv1_out = self.conv1(x)
        layer_outputs.append(conv1_out)
        
        # ReLU after Conv1
        relu1_out = torch.relu(conv1_out)
        layer_outputs.append(relu1_out)
        
        # Pool1 layer
        pool1_out = self.pool(relu1_out)
        layer_outputs.append(pool1_out)
        
        # Conv2 layer
        conv2_out = self.conv2(pool1_out)
        layer_outputs.append(conv2_out)
        
        # ReLU after Conv2
        relu2_out = torch.relu(conv2_out)
        layer_outputs.append(relu2_out)
        
        # Pool2 layer
        pool2_out = self.pool(relu2_out)
        layer_outputs.append(pool2_out)
        
        # Flatten
        flattened = pool2_out.view(-1, 16 * 8 * 8)
        layer_outputs.append(flattened)
        
        # Fully connected layer
        fc1_out = self.fc1(flattened)
        layer_outputs.append(fc1_out)
        
        # Final ReLU
        final_out = torch.relu(fc1_out)
        layer_outputs.append(final_out)
        
        return layer_outputs

class TinyCNN_Head(nn.Module):
    def __init__(self):
        super(TinyCNN_Head, self).__init__()
        self.fc1 = torch.nn.Linear(32,32)
        self.fc2 = torch.nn.Linear(32,10)

    def forward(self, x):
        layer_outputs = []

        fc1_layer = self.fc1(x)
        layer_outputs.append(fc1_layer)

        fc1_relu_layer = torch.relu(fc1_layer)
        layer_outputs.append(fc1_relu_layer)

        fc2_layer = self.fc2(fc1_relu_layer)
        layer_outputs.append(fc2_layer)
        
        return layer_outputs

    
class Resnet18(nn.Module):
    def __init__(self, num_classes, pretrained=False):
        super(TinyCNN, self).__init__()
        if pretrained:
            self.model = resnet18(weights="DEFAULT")
        else:
            self.model = resnet18(weights=None)

        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.model(x)
    
class WrapperModelTrainHead(nn.Module):
    def __init__(self, body, head):
        super().__init__()
        self.body = body
        self.head = head
    
    def forward(self, x):
        with torch.no_grad():
            embed = self.body(x)

        head = self.head(embed[-1])

        return embed + head
    
class CustomUNET(nn.Module):
    def __init__(self):
        super(CustomUNET, self).__init__()
        
        # Encoder
        self.enc1 = self.conv_block(3, 32)
        self.enc2 = self.conv_block(32, 64)
        self.enc3 = self.conv_block(64, 128)
        self.enc4 = self.conv_block(128, 256)
        self.enc5 = self.conv_block(256, 512)
        
        # Decoder
        self.dec4 = self.conv_block(768, 256)
        self.dec3 = self.conv_block(384, 128)
        self.dec2 = self.conv_block(192, 64)
        self.dec1 = self.conv_block(96, 32)
        
        self.final = nn.Conv2d(32, 3, kernel_size=1)
        
        self.pool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
    def conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        e5 = self.enc5(self.pool(e4))  # This is the 2x2x512 latent space
        
        # Decoder
        d4 = self.dec4(torch.cat([self.upsample(e5), e4], dim=1))
        d3 = self.dec3(torch.cat([self.upsample(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.upsample(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.upsample(d2), e1], dim=1))
        
        return self.final(d1)

class ProjNet(nn.Module):
    def __init__(self, size):
        super(ProjNet, self).__init__()
        self.size = size
        self.net = nn.Linear(self.size, self.size)

    def forward(self, input):
        out = self.net(input)
        return out / (torch.sqrt(torch.sum((out) ** 2, dim=1, keepdim=True)) + 1e-20)