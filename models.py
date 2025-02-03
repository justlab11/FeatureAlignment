import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)

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
        layer_outputs.append(x)
        
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
    
class WrapperModelTrainHead(nn.Module):
    def __init__(self, body, head):
        super().__init__()
        self.body = body
        self.head = head
    
    def forward(self, x):
        with torch.no_grad():
            embed = self.body(x)

        head = [self.head(embed[-1])]

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