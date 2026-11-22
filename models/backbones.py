# backbone.py
import torch
from torch import nn
import torchvision

class Backbone(nn.Module):
    AVAILABLE_MODELS = {
        # Existing ResNet models
        "resnet18": torchvision.models.resnet18,
        "resnet34": torchvision.models.resnet34,
        "resnet50": torchvision.models.resnet50,
        "resnet101": torchvision.models.resnet101,
        "resnet152": torchvision.models.resnet152,
        "resnext50": torchvision.models.resnext50_32x4d,
        # New DINOv2 models
        "dinov2_vits14": lambda pretrained: torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14', pretrained=pretrained),
        "dinov2_vitb14": lambda pretrained: torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14', pretrained=pretrained),
        "dinov2_vitl14": lambda pretrained: torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14', pretrained=pretrained),
        "dinov2_vitg14": lambda pretrained: torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14', pretrained=pretrained),
        "dinov2_vitg14_reg": lambda pretrained: torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14_reg', pretrained=pretrained),
    }

    def __init__(
        self,
        backbone_name="resnet50",
        pretrained=True,
        unfreeze_n_blocks=1,
        crop_last_block=True,
    ):
        super().__init__()

        self.backbone_name = backbone_name
        self.pretrained = pretrained
        self.unfreeze_n_blocks = unfreeze_n_blocks
        self.crop_last_block = crop_last_block

        if backbone_name not in self.AVAILABLE_MODELS:
            raise ValueError(f"Backbone {backbone_name} is not recognized!" 
                             f"Supported backbones are: {list(self.AVAILABLE_MODELS.keys())}")

        # Handle DINOv2 backbones
        if backbone_name.startswith("dinov2"):
            self.net = self.AVAILABLE_MODELS[backbone_name](pretrained=pretrained)
            self.out_channels = self.net.embed_dim
            self._setup_dinov2_trainability()
        else:
            # Load ResNet model
            weights = "IMAGENET1K_V1" if pretrained else None
            resnet = self.AVAILABLE_MODELS[backbone_name](weights=weights)
            
            # Create backbone with only the necessary layers
            self.net = nn.Sequential(
                resnet.conv1,
                resnet.bn1,
                resnet.relu,
                resnet.maxpool,
                resnet.layer1,
                resnet.layer2,
                resnet.layer3,
                *([] if crop_last_block else [resnet.layer4]),
            )
            self._setup_resnet_trainability()
            
            # Output channels
            if backbone_name in ["resnet18", "resnet34"]:
                self.out_channels = resnet.layer3[-1].conv2.out_channels
            else:
                self.out_channels = resnet.layer3[-1].conv3.out_channels

    def _setup_resnet_trainability(self):
        """Configure trainable layers for ResNet backbones"""
        if not self.pretrained:
            if self.unfreeze_n_blocks > 0:
                print("Warning: unfreeze_n_blocks ignored when pretrained=False")
            return

        nb_layers = len(self.net)
        assert 0 <= self.unfreeze_n_blocks <= nb_layers, \
            f"unfreeze_n_blocks must be between 0 and {nb_layers}"

        # Freeze layers except last 'unfreeze_n_blocks' blocks
        freeze_layers = nb_layers - self.unfreeze_n_blocks
        for i, layer in enumerate(self.net):
            for param in layer.parameters():
                param.requires_grad = (i >= freeze_layers)

    def _setup_dinov2_trainability(self):
        """Configure trainable layers for DINOv2 backbones"""
        if not self.pretrained:
            if self.unfreeze_n_blocks > 0:
                print("Warning: unfreeze_n_blocks ignored when pretrained=False")
            return

        total_blocks = len(self.net.blocks)
        assert 0 <= self.unfreeze_n_blocks <= total_blocks, \
            f"unfreeze_n_blocks must be between 0 and {total_blocks} for DINOv2"

        # Freeze first N blocks
        freeze_blocks = total_blocks - self.unfreeze_n_blocks
        for i, block in enumerate(self.net.blocks):
            for param in block.parameters():
                param.requires_grad = (i >= freeze_blocks)

        # Always freeze patch embedding and normalization layers
        for param in self.net.patch_embed.parameters():
            param.requires_grad = False
        for param in self.net.norm.parameters():
            param.requires_grad = False

    def forward(self, x):
        if self.backbone_name.startswith("dinov2"):
            return self._forward_dinov2(x)
        return self.net(x)

    def _forward_dinov2(self, x):
        """Forward pass for DINOv2 backbones"""
        # Get patch size and ensure input dimensions are compatible
        patch_size = self.net.patch_embed.patch_size[0]
        H, W = x.shape[2], x.shape[3]
        assert H % patch_size == 0 and W % patch_size == 0, \
            f"Input size {H}x{W} must be divisible by patch size {patch_size}"

        # Forward through backbone
        features = self.net.forward_features(x)
        tokens = features["x_norm_patchtokens"]
        
        # Reshape tokens to spatial feature map
        num_patches = tokens.shape[1]
        grid_size = int(num_patches**0.5)  # Assuming square input
        tokens = tokens.permute(0, 2, 1).reshape(
            tokens.size(0), 
            tokens.size(2), 
            grid_size, 
            grid_size
        )
        return tokens