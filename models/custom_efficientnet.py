"""Custom EfficientNet implementation for compatibility with V1 models."""

import torch
import torch.nn as nn
import math

class MBConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, expand_ratio, se_ratio=0.25):
        super(MBConvBlock, self).__init__()
        self.use_residual = (stride == 1) and (in_channels == out_channels)
        
        # Expansion phase
        expanded_channels = in_channels * expand_ratio
        
        # Pointwise expansion
        if expand_ratio != 1:
            self.expand_conv = nn.Conv2d(in_channels, expanded_channels, 1, bias=False)
            self.bn0 = nn.BatchNorm2d(expanded_channels)
            self.swish0 = nn.SiLU(inplace=True)
        
        # Depthwise convolution
        self.depthwise_conv = nn.Conv2d(
            expanded_channels, expanded_channels, kernel_size,
            stride=stride, padding=kernel_size//2, groups=expanded_channels, bias=False
        )
        self.bn1 = nn.BatchNorm2d(expanded_channels)
        self.swish1 = nn.SiLU(inplace=True)
        
        # Squeeze-and-Excitation
        se_channels = max(1, int(in_channels * se_ratio))
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(expanded_channels, se_channels, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(se_channels, expanded_channels, 1),
            nn.Sigmoid()
        )
        
        # Output phase
        self.project_conv = nn.Conv2d(expanded_channels, out_channels, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
    
    def forward(self, x):
        identity = x
        
        # Expansion phase
        if hasattr(self, 'expand_conv'):
            x = self.expand_conv(x)
            x = self.bn0(x)
            x = self.swish0(x)
        
        # Depthwise convolution
        x = self.depthwise_conv(x)
        x = self.bn1(x)
        x = self.swish1(x)
        
        # Squeeze-and-Excitation
        x_se = self.se(x)
        x = x * x_se
        
        # Output phase
        x = self.project_conv(x)
        x = self.bn2(x)
        
        # Skip connection
        if self.use_residual:
            x = x + identity
            
        return x

class EfficientNet(nn.Module):
    def __init__(self, num_classes=1000, width_coeff=1.0, depth_coeff=1.0, dropout_rate=0.2):
        super(EfficientNet, self).__init__()
        
        # Stem
        out_channels = self._round_filters(32, width_coeff)
        self.stem_conv = nn.Conv2d(3, out_channels, 3, stride=2, padding=1, bias=False)
        self.bn0 = nn.BatchNorm2d(out_channels)
        self.swish = nn.SiLU(inplace=True)
        
        # Build blocks
        self.blocks = nn.ModuleList()
        
        # Block 0
        in_channels = out_channels
        out_channels = self._round_filters(16, width_coeff)
        num_repeats = self._round_repeats(1, depth_coeff)
        self.blocks.append(self._make_block(in_channels, out_channels, 3, 1, 1, num_repeats))
        in_channels = out_channels
        
        # Block 1
        out_channels = self._round_filters(24, width_coeff)
        num_repeats = self._round_repeats(2, depth_coeff)
        self.blocks.append(self._make_block(in_channels, out_channels, 3, 2, 6, num_repeats))
        in_channels = out_channels
        
        # Block 2
        out_channels = self._round_filters(40, width_coeff)
        num_repeats = self._round_repeats(2, depth_coeff)
        self.blocks.append(self._make_block(in_channels, out_channels, 5, 2, 6, num_repeats))
        in_channels = out_channels
        
        # Block 3
        out_channels = self._round_filters(80, width_coeff)
        num_repeats = self._round_repeats(3, depth_coeff)
        self.blocks.append(self._make_block(in_channels, out_channels, 3, 2, 6, num_repeats))
        in_channels = out_channels
        
        # Block 4
        out_channels = self._round_filters(112, width_coeff)
        num_repeats = self._round_repeats(3, depth_coeff)
        self.blocks.append(self._make_block(in_channels, out_channels, 5, 1, 6, num_repeats))
        in_channels = out_channels
        
        # Block 5
        out_channels = self._round_filters(192, width_coeff)
        num_repeats = self._round_repeats(4, depth_coeff)
        self.blocks.append(self._make_block(in_channels, out_channels, 5, 2, 6, num_repeats))
        in_channels = out_channels
        
        # Block 6
        out_channels = self._round_filters(320, width_coeff)
        num_repeats = self._round_repeats(1, depth_coeff)
        self.blocks.append(self._make_block(in_channels, out_channels, 3, 1, 6, num_repeats))
        in_channels = out_channels
        
        # Head
        out_channels = self._round_filters(1280, width_coeff)
        self.head_conv = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        # Classifier
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(out_channels, num_classes)
        
        # Initialize weights
        self._initialize_weights()
    
    def _make_block(self, in_channels, out_channels, kernel_size, stride, expand_ratio, num_repeats):
        layers = [MBConvBlock(in_channels, out_channels, kernel_size, stride, expand_ratio)]
        for _ in range(1, num_repeats):
            layers.append(MBConvBlock(out_channels, out_channels, kernel_size, 1, expand_ratio))
        return nn.Sequential(*layers)
    
    def _round_filters(self, filters, width_coeff):
        if width_coeff == 1.0:
            return filters
        divisor = 8
        filters = filters * width_coeff
        new_filters = max(divisor, int(filters + divisor / 2) // divisor * divisor)
        if new_filters < 0.9 * filters:
            new_filters += divisor
        return int(new_filters)
    
    def _round_repeats(self, repeats, depth_coeff):
        return int(math.ceil(depth_coeff * repeats))
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        # Stem
        x = self.stem_conv(x)
        x = self.bn0(x)
        x = self.swish(x)
        
        # Blocks
        for block in self.blocks:
            x = block(x)
        
        # Head
        x = self.head_conv(x)
        x = self.bn1(x)
        x = self.swish(x)
        
        # Pooling and classifier
        x = x.mean([2, 3])  # Global average pooling
        x = self.dropout(x)
        x = self.fc(x)
        
        return x

    @classmethod
    def from_name(cls, model_name, **kwargs):
        """Create EfficientNet model from name."""
        if model_name == 'efficientnet-b0':
            return cls(width_coeff=1.0, depth_coeff=1.0, **kwargs)
        elif model_name == 'efficientnet-b1':
            return cls(width_coeff=1.0, depth_coeff=1.1, **kwargs)
        elif model_name == 'efficientnet-b2':
            return cls(width_coeff=1.1, depth_coeff=1.2, **kwargs)
        elif model_name == 'efficientnet-b3':
            return cls(width_coeff=1.2, depth_coeff=1.4, **kwargs)
        else:
            raise ValueError(f"Unknown model name: {model_name}")
