import torch
import torch.nn as nn
import logging
import torch.nn.functional as F
from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152


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

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=1,  # Modified: Always stride 1
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, bias=False) # Modified: Always stride 1


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
    ):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
    ):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(
        self,
        block,
        layers,
        num_classes=10,
        zero_init_residual=False,
        groups=1,
        width_per_group=64,
        replace_stride_with_dilation=None,
        norm_layer=None,
    ):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                "or a 3-element tuple, got {}".format(replace_stride_with_dilation)
            )
        self.groups = groups
        self.base_width = width_per_group

        # CIFAR10: kernel_size 7 -> 3, stride 2 -> 1, padding 3->1
        self.conv1 = nn.Conv2d(
            3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False
        )
        # END

        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        # Modified: Remove max pooling
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1) #Modified: Remove stride
        self.layer2 = self._make_layer(
            block, 128, layers[1], stride=1, dilate=replace_stride_with_dilation[0] #Modified: Remove stride
        )
        self.layer3 = self._make_layer(
            block, 256, layers[2], stride=1, dilate=replace_stride_with_dilation[1] #Modified: Remove stride
        )
        self.layer4 = self._make_layer(
            block, 512, layers[3], stride=1, dilate=replace_stride_with_dilation[2] #Modified: Remove stride
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False): #Modified: Remove stride
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer,
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # Modified: Remove max pooling
        # x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)

        return x

    
class DynamicResNet(nn.Module):
    def __init__(self, resnet_type='resnet9', num_classes=10):
        super(DynamicResNet, self).__init__()
        
        resnet_models = {
            'resnet9': ResNet,
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
            self.model = resnet_models[resnet_type](BasicBlock, [1,1,1,1])

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
    
    def reset_parameters(self):
        for module in self.model.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear, nn.ConvTranspose2d)):
                nn.init.xavier_normal_(module.weight)

                if module.bias is not None:
                    nn.init.zeros_(module.bias)

            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def reset_head_parameters(self):
        if hasattr(self.model, 'fc'):
            nn.init.xavier_normal_(self.model.fc.weight)
            if self.model.fc.bias is not None:
                nn.init.zeros_(self.model.fc.bias)

    def forward(self, x):
        layer_outputs = [x]

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

    
class SmallCustomUNET(nn.Module):
    def __init__(self, base_channels=32, noise_channels=8):
        super(SmallCustomUNET, self).__init__()
        
        self.base_channels = base_channels
        self.noise_channels = noise_channels

        # Encoder
        self.enc1 = self.conv_block(3, base_channels)
        self.enc2 = self.conv_block(base_channels, base_channels * 2)
        self.enc3 = self.conv_block(base_channels * 2, base_channels * 4)
        self.enc4 = self.conv_block(base_channels * 4, base_channels * 8)
        self.enc5 = self.conv_block(base_channels * 8, base_channels * 16)

        # Latent space normalization
        #self.latent_norm = nn.LayerNorm([base_channels * 16, 2, 2])  # For latent space (2x2 for 32x32 input)
        self.latent_norm = nn.Identity()
        # Noise addition
        self.noise_conv = nn.Conv1d(noise_channels, base_channels * 16, kernel_size=1)  # Transform noise to match latent space channels
        self.noise_weight = nn.Parameter(torch.tensor(0.05))
        self.latent_weight = nn.Parameter(torch.tensor(0.95))
        
        # Decoder
        self.dec4 = self.conv_block(base_channels * (16 + 8), base_channels * 8)
        self.dec3 = self.conv_block(base_channels * (8 + 4), base_channels * 4)
        self.dec2 = self.conv_block(base_channels * (4 + 2), base_channels * 2)
        self.dec1 = self.conv_block(base_channels * (2 + 1), base_channels)
        
        # Final output layer
        self.final = nn.Conv2d(base_channels, 3, kernel_size=1)
        
        # Pooling and Upsampling layers
        self.pool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def conv_block(self, in_ch, out_ch):
        """A helper function to create a convolutional block."""
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
    
    def get_normalized_weights(self):
        """Normalize noise and latent weights for blending."""
        sum_weights = self.noise_weight.abs() + self.latent_weight.abs() +1e-8
        return (
            self.noise_weight.abs() / sum_weights,
            self.latent_weight.abs() / sum_weights,
        )

    def forward(self, x):
        layer_outputs = []

        # Encoder
        e1 = self.enc1(x)  
        layer_outputs.append(e1)

        e2 = self.enc2(self.pool(e1))  
        layer_outputs.append(e2)

        e3 = self.enc3(self.pool(e2))  
        layer_outputs.append(e3)

        e4 = self.enc4(self.pool(e3))  
        layer_outputs.append(e4)

        e5 = self.enc5(self.pool(e4))  
        layer_outputs.append(e5)  # Latent representation

        # Latent space normalization and noise addition
        e5_normalized = self.latent_norm(e5)

        noise = torch.randn(e5.size(0), self.noise_channels, e5.size(2) * e5.size(3), device=e5.device)  
        
        # Transform noise to match latent space dimensions
        noise_transformed = self.noise_conv(noise).view_as(e5)

        norm_noise_weight, norm_latent_weight = self.get_normalized_weights()

        e5_blended = e5_normalized * norm_latent_weight + noise_transformed * norm_noise_weight
        e5_blended = self.latent_norm(e5_blended)  # Re-normalize after blending

        # Decoder
        d4 = self.dec4(torch.cat([self.upsample(e5_blended), e4], dim=1))
        layer_outputs.append(d4)

        d3 = self.dec3(torch.cat([self.upsample(d4), e3], dim=1))
        layer_outputs.append(d3)

        d2 = self.dec2(torch.cat([self.upsample(d3), e2], dim=1))
        layer_outputs.append(d2)

        d1 = self.dec1(torch.cat([self.upsample(d2), e1], dim=1))
        layer_outputs.append(d1)

        final_output = self.final(d1)
        layer_outputs.append(final_output)

        return layer_outputs

class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

class SmallAttentionUNET(nn.Module):
    def __init__(self, base_channels=32, noise_channels=8):
        super(SmallAttentionUNET, self).__init__()
        
        self.base_channels = base_channels
        self.noise_channels = noise_channels

        # Encoder
        self.enc1 = self.conv_block(3, base_channels)
        self.enc2 = self.conv_block(base_channels, base_channels * 2)
        self.enc3 = self.conv_block(base_channels * 2, base_channels * 4)
        self.enc4 = self.conv_block(base_channels * 4, base_channels * 8)
        self.enc5 = self.conv_block(base_channels * 8, base_channels * 16)

        # Latent space normalization
        self.latent_norm = nn.LayerNorm([base_channels * 16, 2, 2])  # For latent space (2x2 for 32x32 input)

        # Noise addition
        self.noise_conv = nn.Conv1d(noise_channels, base_channels * 16, kernel_size=1)  # Transform noise to match latent space channels
        self.noise_weight = nn.Parameter(torch.tensor(0.05))
        self.latent_weight = nn.Parameter(torch.tensor(1.0))
        
        # Attention Gates
        self.attention4 = AttentionGate(F_g=base_channels*16, F_l=base_channels*8, F_int=base_channels*8)
        self.attention3 = AttentionGate(F_g=base_channels*8, F_l=base_channels*4, F_int=base_channels*4)
        self.attention2 = AttentionGate(F_g=base_channels*4, F_l=base_channels*2, F_int=base_channels*2)
        self.attention1 = AttentionGate(F_g=base_channels*2, F_l=base_channels, F_int=base_channels)

        # Decoder
        self.dec4 = self.conv_block(base_channels * (16 + 8), base_channels * 8)
        self.dec3 = self.conv_block(base_channels * (8 + 4), base_channels * 4)
        self.dec2 = self.conv_block(base_channels * (4 + 2), base_channels * 2)
        self.dec1 = self.conv_block(base_channels * (2 + 1), base_channels)
        
        # Final output layer
        self.final = nn.Conv2d(base_channels, 3, kernel_size=1)
        
        # Pooling and Upsampling layers
        self.pool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
    
    def get_normalized_weights(self):
        sum_weights = self.noise_weight.abs() + self.latent_weight.abs()
        return (
            self.noise_weight.abs() / sum_weights,
            self.latent_weight.abs() / sum_weights,
        )

    def forward(self, x):
        layer_outputs = []

        # Encoder
        e1 = self.enc1(x)  
        layer_outputs.append(e1)

        e2 = self.enc2(self.pool(e1))  
        layer_outputs.append(e2)

        e3 = self.enc3(self.pool(e2))  
        layer_outputs.append(e3)

        e4 = self.enc4(self.pool(e3))  
        layer_outputs.append(e4)

        e5 = self.enc5(self.pool(e4))  
        layer_outputs.append(e5)  # Latent representation

        # Latent space normalization and noise addition
        e5_normalized = self.latent_norm(e5)

        noise = torch.randn(e5.size(0), self.noise_channels, e5.size(2) * e5.size(3), device=e5.device)  
        
        # Transform noise to match latent space dimensions
        noise_transformed = self.noise_conv(noise).view_as(e5)

        norm_noise_weight, norm_latent_weight = self.get_normalized_weights()

        e5_blended = e5_normalized * norm_latent_weight + noise_transformed * norm_noise_weight
        e5_blended = self.latent_norm(e5_blended)  # Re-normalize after blending

        # Decoder with Attention
        d4 = self.dec4(torch.cat([self.upsample(e5_blended), self.attention4(g=self.upsample(e5_blended), x=e4)], dim=1))
        layer_outputs.append(d4)

        d3 = self.dec3(torch.cat([self.upsample(d4), self.attention3(g=self.upsample(d4), x=e3)], dim=1))
        layer_outputs.append(d3)

        d2 = self.dec2(torch.cat([self.upsample(d3), self.attention2(g=self.upsample(d3), x=e2)], dim=1))
        layer_outputs.append(d2)

        d1 = self.dec1(torch.cat([self.upsample(d2), self.attention1(g=self.upsample(d2), x=e1)], dim=1))
        layer_outputs.append(d1)

        final_output = self.final(d1)
        layer_outputs.append(final_output)

        return layer_outputs

class LargeCustomUNET(nn.Module):
    """
    Expects images of size 224x224
    """
    def __init__(self, base_channels=32, noise_channels=8):
        super(LargeCustomUNET, self).__init__()
        
        self.base_channels = base_channels
        self.noise_channels = noise_channels

        # Encoder
        self.enc1 = self.conv_block(3, base_channels)
        self.enc2 = self.conv_block(base_channels, base_channels * 2)
        self.enc3 = self.conv_block(base_channels * 2, base_channels * 4)
        self.enc4 = self.conv_block(base_channels * 4, base_channels * 8)
        self.enc5 = self.conv_block(base_channels * 8, base_channels * 16)

        # Latent space normalization
        self.latent_norm = nn.LayerNorm([base_channels * 16, 8, 8])  # Adjusted for latent space size (7x7 for 224x224 input)

        # Noise addition
        self.noise_conv = nn.Conv2d(noise_channels, base_channels * 16, kernel_size=1)  # Transform noise to match latent space channels
        self.noise_weight = nn.Parameter(torch.tensor(0.05))
        self.latent_weight = nn.Parameter(torch.tensor(1.0))
        
        # Decoder
        self.dec4 = self.conv_block(base_channels * (16 + 8), base_channels * 8)
        self.dec3 = self.conv_block(base_channels * (8 + 4), base_channels * 4)
        self.dec2 = self.conv_block(base_channels * (4 + 2), base_channels * 2)
        self.dec1 = self.conv_block(base_channels * (2 + 1), base_channels)
        
        # Final output layer
        self.final = nn.Conv2d(base_channels, 3, kernel_size=1)
        
        # Pooling and Upsampling layers
        self.pool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
    
    def get_normalized_weights(self):
        sum_weights = self.noise_weight.abs() + self.latent_weight.abs()
        return (
            self.noise_weight.abs() / sum_weights,
            self.latent_weight.abs() / sum_weights,
        )

    def forward(self, x):
        layer_outputs = [x]

        # Encoder
        e1 = self.enc1(x)  
        layer_outputs.append(e1)

        e2 = self.enc2(self.pool(e1))  
        layer_outputs.append(e2)

        e3 = self.enc3(self.pool(e2))  
        layer_outputs.append(e3)

        e4 = self.enc4(self.pool(e3))  
        layer_outputs.append(e4)

        e5 = self.enc5(self.pool(e4))
        layer_outputs.append(e5)  

        e5_normalized = self.latent_norm(e5)

        noise = torch.randn(e5.size(0), self.noise_channels, e5.size(2), e5.size(3), device=e5.device)
        noise_transformed = self.noise_conv(noise).view_as(e5)

        norm_noise_weight, norm_latent_weight = self.get_normalized_weights()
        
        e5_blended = e5_normalized * norm_latent_weight + noise_transformed * norm_noise_weight
        e5_blended = self.latent_norm(e5_blended)

        # Decoder
        d4 = self.dec4(torch.cat([self.upsample(e5_blended), e4], dim=1))
        layer_outputs.append(d4)

        d3 = self.dec3(torch.cat([self.upsample(d4), e3], dim=1))
        layer_outputs.append(d3)

        d2 = self.dec2(torch.cat([self.upsample(d3), e2], dim=1))
        layer_outputs.append(d2)

        d1 = self.dec1(torch.cat([self.upsample(d2), e1], dim=1))
        layer_outputs.append(d1)

        final_output = self.final(d1)
        layer_outputs.append(final_output)

        return layer_outputs
    
class LargeAttentionUNET(nn.Module):
    """
    Expects images of size 224x224
    """
    def __init__(self, base_channels=32, noise_channels=8):
        super(LargeAttentionUNET, self).__init__()
        
        self.base_channels = base_channels
        self.noise_channels = noise_channels

        # Encoder
        self.enc1 = self.conv_block(3, base_channels)
        self.enc2 = self.conv_block(base_channels, base_channels * 2)
        self.enc3 = self.conv_block(base_channels * 2, base_channels * 4)
        self.enc4 = self.conv_block(base_channels * 4, base_channels * 8)
        self.enc5 = self.conv_block(base_channels * 8, base_channels * 16)

        # Latent space normalization
        self.latent_norm = nn.LayerNorm([base_channels * 16, 7, 7])  # Adjusted for latent space size (7x7 for 224x224 input)

        # Noise addition
        self.noise_conv = nn.Conv2d(noise_channels, base_channels * 16, kernel_size=1)  # Transform noise to match latent space channels
        self.noise_weight = nn.Parameter(torch.tensor(0.05))
        self.latent_weight = nn.Parameter(torch.tensor(1.0))
        
        # Attention Gates
        self.attention4 = AttentionGate(F_g=base_channels*16, F_l=base_channels*8, F_int=base_channels*8)
        self.attention3 = AttentionGate(F_g=base_channels*8, F_l=base_channels*4, F_int=base_channels*4)
        self.attention2 = AttentionGate(F_g=base_channels*4, F_l=base_channels*2, F_int=base_channels*2)
        self.attention1 = AttentionGate(F_g=base_channels*2, F_l=base_channels, F_int=base_channels)

        # Decoder
        self.dec4 = self.conv_block(base_channels * (16 + 8), base_channels * 8)
        self.dec3 = self.conv_block(base_channels * (8 + 4), base_channels * 4)
        self.dec2 = self.conv_block(base_channels * (4 + 2), base_channels * 2)
        self.dec1 = self.conv_block(base_channels * (2 + 1), base_channels)
        
        # Final output layer
        self.final = nn.Conv2d(base_channels, 3, kernel_size=1)
        
        # Pooling and Upsampling layers
        self.pool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
    
    def get_normalized_weights(self):
        sum_weights = self.noise_weight.abs() + self.latent_weight.abs()
        return (
            self.noise_weight.abs() / sum_weights,
            self.latent_weight.abs() / sum_weights,
        )

    def forward(self, x):
        layer_outputs = []

        # Encoder
        e1 = self.enc1(x)  
        layer_outputs.append(e1)

        e2 = self.enc2(self.pool(e1))  
        layer_outputs.append(e2)

        e3 = self.enc3(self.pool(e2))  
        layer_outputs.append(e3)

        e4 = self.enc4(self.pool(e3))  
        layer_outputs.append(e4)

        e5 = self.enc5(self.pool(e4))  
        
        # Latent space normalization and noise addition
        batch_size, channels, height, width = e5.shape

        e5_normalized = e5.view(batch_size, channels, -1)  # Flatten spatial dimensions
        e5_normalized = nn.LayerNorm(e5_normalized.size()[1:])(e5_normalized)  
        e5_normalized = e5_normalized.view(batch_size, channels, height, width)  # Reshape back

        noise = torch.randn(batch_size, self.noise_channels, height, width, device=e5.device)  
        
        # Transform noise to match latent space dimensions
        noise_transformed = self.noise_conv(noise)

        norm_noise_weight, norm_latent_weight = self.get_normalized_weights()

        e5_blended = e5_normalized * norm_latent_weight + noise_transformed * norm_noise_weight
        e5_blended = e5_blended.view_as(e5)  # Ensure shape consistency if needed

        layer_outputs.append(e5_blended)

        # Decoder with Attention
        d4 = self.dec4(torch.cat([self.upsample(e5_blended), self.attention4(g=self.upsample(e5_blended), x=e4)], dim=1))
        layer_outputs.append(d4)

        d3 = self.dec3(torch.cat([self.upsample(d4), self.attention3(g=self.upsample(d4), x=e3)], dim=1))
        layer_outputs.append(d3)

        d2 = self.dec2(torch.cat([self.upsample(d3), self.attention2(g=self.upsample(d3), x=e2)], dim=1))
        layer_outputs.append(d2)

        d1 = self.dec1(torch.cat([self.upsample(d2), self.attention1(g=self.upsample(d2), x=e1)], dim=1))
        layer_outputs.append(d1)

        final_output = self.final(d1)
        layer_outputs.append(final_output)

        return layer_outputs

class CustomAutoencoder(nn.Module):
    def __init__(self):
        super(CustomAutoencoder, self).__init__()
        
        # Encoder
        self.enc1 = self.conv_block(3, 32)
        self.enc2 = self.conv_block(32, 64)
        self.enc3 = self.conv_block(64, 128)
        self.enc4 = self.conv_block(128, 256)
        self.enc5 = self.conv_block(256, 512)

        # Latent space normalization
        #self.latent_norm = nn.LayerNorm([512, 2, 2])
        self.latent_norm = nn.Identity()
        # Decoder
        self.dec5 = self.conv_block(512, 256)
        self.dec4 = self.conv_block(256, 128)
        self.dec3 = self.conv_block(128, 64)
        self.dec2 = self.conv_block(64, 32)
        
        # Final output layer
        self.final = nn.Conv2d(32, 3, kernel_size=1)

        # Pooling and upsampling layers
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
        layer_outputs = []

        # Encoder
        e1 = self.enc1(x)
        layer_outputs.append(e1)

        e2 = self.enc2(self.pool(e1))
        layer_outputs.append(e2)

        e3 = self.enc3(self.pool(e2))
        layer_outputs.append(e3)

        e4 = self.enc4(self.pool(e3))
        layer_outputs.append(e4)

        e5 = self.enc5(self.pool(e4))  # This is the latent space (2x2x512)
        layer_outputs.append(e5)

        # Normalize latent space
        e5_normalized = self.latent_norm(e5)

        # Decoder
        d5 = self.dec5(self.upsample(e5_normalized))
        layer_outputs.append(d5)

        d4 = self.dec4(self.upsample(d5))
        layer_outputs.append(d4)

        d3 = self.dec3(self.upsample(d4))
        layer_outputs.append(d3)

        d2 = self.dec2(self.upsample(d3))
        layer_outputs.append(d2)

        final = self.final(d2)  # Final output (reconstructed image)
        layer_outputs.append(final)

        return layer_outputs

class ProjNet(nn.Module):
    def __init__(self, size):
        super(ProjNet, self).__init__()
        self.size = size
        self.net = nn.Linear(self.size, self.size)

    def forward(self, input):
        out = self.net(input)
        return out / (torch.sqrt(torch.sum((out) ** 2, dim=1, keepdim=True)) + 1e-20)
