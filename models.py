import torch
import torch.nn as nn
import logging
import torch.nn.functional as F
from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152


logger = logging.getLogger(__name__)

## TODO: make sure this can work for resnet or any other model

class DynamicCNN(nn.Module):
    def __init__(self, input_shape:tuple, num_filters:list=[32, 32], kernel_size:list=[3], stride:list=[1], padding:list=[1],
                 mlp_layer_sizes:list=[128, 64], num_classes:int=10):
        super(DynamicCNN, self).__init__()

        logger.debug(f"""
            Building model with parameters:
            \n\tInput Size: {input_shape}
            \n\tFilters: {num_filters}
            \n\tKernels: {kernel_size}
            \n\tMLP Layers: {mlp_layer_sizes}
            \n\tOutput Classes: {num_classes}
        """)

        self.input_shape = input_shape
        self.freeze_head = False
        self.freeze_body = False
        
        # Determine the number of convolutional layers based on the longest list
        self.num_conv_layers = max(len(num_filters), len(kernel_size), len(stride), len(padding))
        
        # Adjust lists to be the same length
        self.num_filters = num_filters + [num_filters[-1]] * (self.num_conv_layers - len(num_filters))
        self.kernel_size = kernel_size + [kernel_size[-1]] * (self.num_conv_layers - len(kernel_size))
        self.stride = stride + [stride[-1]] * (self.num_conv_layers - len(stride))
        self.padding = padding + [padding[-1]] * (self.num_conv_layers - len(padding))
        
        # Ensure all parameters are positive integers
        for param_list in [self.num_filters, self.kernel_size, self.stride, self.padding]:
            for param in param_list:
                assert isinstance(param, int) and param > 0, "All parameters must be positive integers"
        
        # Ensure all MLP layer sizes are positive integers
        for size in mlp_layer_sizes:
            assert isinstance(size, int) and size > 0, "All MLP layer sizes must be positive integers"

        self.conv_layers = nn.ModuleList()

        # Assuming input shape is (batch_size, channels, height, width)
        in_channels = input_shape[1]
        height = input_shape[2]
        width = input_shape[3]
        
        for i in range(self.num_conv_layers):
            if i == 0:
                layer = nn.Sequential(
                    nn.Conv2d(in_channels, self.num_filters[i], kernel_size=self.kernel_size[i], stride=self.stride[i], padding=self.padding[i]),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2)
                )
            else:
                layer = nn.Sequential(
                    nn.Conv2d(self.num_filters[i-1], self.num_filters[i], kernel_size=self.kernel_size[i], stride=self.stride[i], padding=self.padding[i]),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2)
                )
            
            self.conv_layers.append(layer)
            
            # Update dimensions after each layer
            height = (height + 2 * self.padding[i] - self.kernel_size[i]) // self.stride[i] + 1
            height = height // 2  # MaxPool2d
            width = (width + 2 * self.padding[i] - self.kernel_size[i]) // self.stride[i] + 1
            width = width // 2  # MaxPool2d
        
        # Store the final output shape before flattening
        self.output_shape = (self.num_filters[-1], height, width)

        self.mlp_layers = nn.ModuleList()
        
        # Initialize MLP layers
        for i in range(len(mlp_layer_sizes)):
            if i == 0:
                layer = nn.Linear(self.output_shape[0] * self.output_shape[1] * self.output_shape[2], mlp_layer_sizes[i])
            else:
                layer = nn.Linear(mlp_layer_sizes[i-1], mlp_layer_sizes[i])
            self.mlp_layers.append(layer)
        
        # Output layer
        self.output_layer = nn.Linear(mlp_layer_sizes[-1], num_classes)
        self.num_classes = num_classes

    def set_freeze_head(self, freeze: bool):
        self.freeze_head = freeze
        for param in self.output_layer.parameters():
            param.requires_grad = not freeze
        if freeze and self.freeze_body:
            logger.error("Cannot freeze both head and body simultaneously.")
            raise ValueError("Cannot freeze both head and body simultaneously.")

    def set_freeze_body(self, freeze: bool):
        self.freeze_body = freeze
        for layer in self.conv_layers + self.mlp_layers:
            for param in layer.parameters():
                param.requires_grad = not freeze
        if freeze and self.freeze_head:
            logger.error("Cannot freeze both head and body simultaneously.")
            raise ValueError("Cannot freeze both head and body simultaneously.")

    def get_body_output_size(self):
        return self.mlp_layers[-1].out_features
    
    def get_num_layers(self):
        conv_layers = self.num_conv_layers * 3
        mlp_layers = len(self.mlp_layers) * 2 - 1
        total_layers = conv_layers + mlp_layers + 1

        return total_layers
    
    def reset_parameters(self):
        for layer in self.conv_layers:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        for layer in self.mlp_layers:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        if hasattr(self.output_layer, 'reset_parameters'):
            self.output_layer.reset_parameters()


    def forward(self, x):
        layer_outputs = []
        
        for i, layer in enumerate(self.conv_layers):
            # Conv layer
            x = layer[0](x)
            layer_outputs.append(x)
            
            # ReLU layer
            x = layer[1](x)
            layer_outputs.append(x)
            
            # MaxPool layer
            x = layer[2](x)
            layer_outputs.append(x)
        
        # Flatten
        x = x.view(-1, self.output_shape[0] * self.output_shape[1] * self.output_shape[2])

        for i, mlp_layer in enumerate(self.mlp_layers):
            x = mlp_layer(x)
            layer_outputs.append(x)
            
            if i < len(self.mlp_layers) - 1:
                # Apply ReLU activation for hidden layers
                x = torch.relu(x)
                layer_outputs.append(x)

        if self.freeze_body:
            layer_outputs = [x.detach()]  # Only keep the final MLP output, detached
        
        # Output layer
        if not self.freeze_head:
            output = self.output_layer(x)
            layer_outputs.append(output)
        
        return layer_outputs
    
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet9(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet9, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        
        self.layer1 = ResidualBlock(64, 64)
        self.layer2 = ResidualBlock(64, 64)
        self.layer3 = ResidualBlock(64, 64)
        
        self.layer4 = ResidualBlock(64, 128, stride=2)
        self.layer5 = ResidualBlock(128, 128)
        
        self.layer6 = ResidualBlock(128, 256, stride=2)
        self.layer7 = ResidualBlock(256, 256)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        
        out = self.layer4(out)
        out = self.layer5(out)
        
        out = self.layer6(out)
        out = self.layer7(out)
        
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        
        return out

    
class DynamicResNet(nn.Module):
    def __init__(self, resnet_type='resnet9', num_classes=10):
        super(DynamicResNet, self).__init__()
        
        resnet_models = {
            'resnet9': ResNet9,
            'resnet18': resnet18,
            'resnet34': resnet34,
            'resnet50': resnet50,
            'resnet101': resnet101,
            'resnet152': resnet152
        }
        
        if resnet_type not in resnet_models:
            raise ValueError(f"Invalid ResNet type. Choose from {list(resnet_models.keys())}")
        
        if resnet_type != 'resnet9':
            self.model = resnet_models[resnet_type](pretrained=False)
        else:
            self.model = resnet_models[resnet_type]()

        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        
        self.freeze_head = False
        self.freeze_body = False

        self.body_output_size = self.model.fc.in_features

    def set_freeze_head(self, freeze: bool):
        self.freeze_head = freeze
        for param in self.model.fc.parameters():
            param.requires_grad = not freeze
        if freeze and self.freeze_body:
            logger.error("Cannot freeze both head and body simultaneously.")
            raise ValueError("Cannot freeze both head and body simultaneously.")

    def set_freeze_body(self, freeze: bool):
        self.freeze_body = freeze
        for name, param in self.model.named_parameters():
            if "fc" not in name:  # Freeze all layers except the final fully connected layer
                param.requires_grad = not freeze
        if freeze and self.freeze_head:
            logger.error("Cannot freeze both head and body simultaneously.")
            raise ValueError("Cannot freeze both head and body simultaneously.")
        
    def get_body_output_size(self):
        return self.body_output_size

    def forward(self, x):
        layer_outputs = []

        def hook_fn(module, input, output):
            if self.freeze_body:
                layer_outputs.append(output.detach())
            else:
                layer_outputs.append(output)

        hooks = []
        for name, module in self.model.named_children():
            if name != 'fc':  # Don't add hook to the final layer
                hooks.append(module.register_forward_hook(hook_fn))

        # Forward pass
        output = self.model(x)

        # Remove hooks
        for hook in hooks:
            hook.remove()

        if not self.freeze_head:
            layer_outputs.append(output)

        return layer_outputs
    
    def get_num_layers(self):
        return len(list(self.model.children()))

    
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