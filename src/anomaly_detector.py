import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.models as models
import numpy as np
import cv2
import os

class PatchCoreAnomalyDetector(nn.Module):
    """
    Unsupervised PatchCore Anomaly Detector.
    Extracts deep features using a pre-trained CNN backbone and generates 
    pixel-level anomaly score heatmaps using distance matching against a reference memory bank.
    """
    def __init__(self, backbone_name: str = "resnet18", device: str = None):
        super().__init__()
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Anomaly Engine] Loading pre-trained backbone '{backbone_name}' on device: '{self.device}'")
        
        # Load pre-trained ResNet backbone
        if backbone_name == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT
            self.backbone = models.resnet18(weights=weights)
        else:
            weights = models.ResNet50_Weights.DEFAULT
            self.backbone = models.resnet50(weights=weights)
            
        self.backbone.eval()
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        self.backbone.to(self.device)
        self.features = []

        # Hook to capture mid-level feature maps (layer2 and layer3)
        self.backbone.layer2[-1].register_forward_hook(self._hook)
        self.backbone.layer3[-1].register_forward_hook(self._hook)

        # Standard Preprocessing Transform for ImageNet Backbones
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.memory_bank = None

    def _hook(self, module, input, output):
        self.features.append(output)

    def extract_patch_features(self, img_tensor: torch.Tensor) -> torch.Tensor:
        """Extracts and concatenates patch feature maps from layer2 and layer3."""
        self.features = []
        with torch.no_grad():
            _ = self.backbone(img_tensor)

        fmap1 = self.features[0] # layer2 output
        fmap2 = self.features[1] # layer3 output

        # Resize layer3 features to match layer2 spatial dimensions
        fmap2_resized = F.interpolate(fmap2, size=fmap1.shape[-2:], mode="bilinear", align_corners=False)
        
        # Concatenate along channel dimension
        patch_features = torch.cat([fmap1, fmap2_resized], dim=1) # [B, C1+C2, H, W]
        return patch_features

    def fit_memory_bank(self, reference_images: list):
        """Builds normal feature memory bank from a set of reference/good cropped ROIs."""
        print(f"[Anomaly Engine] Building memory bank from {len(reference_images)} reference images...")
        bank = []
        for img in reference_images:
            tensor = self.transform(img).unsqueeze(0).to(self.device)
            fmaps = self.extract_patch_features(tensor) # [1, C, H, W]
            
            # Reshape into [H*W, C] feature vectors
            c = fmaps.shape[1]
            patches = fmaps.squeeze(0).permute(1, 2, 0).reshape(-1, c)
            bank.append(patches)

        self.memory_bank = torch.cat(bank, dim=0)
        print(f"[Anomaly Engine] Memory Bank built with shape: {self.memory_bank.shape}")

    def predict_anomaly(self, img: np.ndarray) -> tuple:
        """
        Calculates pixel-level anomaly heatmap and scalar score for an input ROI image.
        """
        orig_h, orig_w = img.shape[:2]
        tensor = self.transform(img).unsqueeze(0).to(self.device)
        
        fmaps = self.extract_patch_features(tensor) # [1, C, H, W]
        _, c, h, w = fmaps.shape
        test_patches = fmaps.squeeze(0).permute(1, 2, 0).reshape(-1, c) # [H*W, C]

        if self.memory_bank is not None:
            # Distance matching against reference memory bank
            dists = torch.cdist(test_patches, self.memory_bank, p=2.0)
            min_dists, _ = torch.min(dists, dim=1)
        else:
            # Self-reference mode (distance relative to feature norms)
            min_dists = torch.norm(test_patches, dim=1)

        # Reshape score back to spatial grid [1, 1, H, W]
        score_map = min_dists.reshape(1, 1, h, w)

        # Upsample anomaly map back to original image resolution
        anomaly_map = F.interpolate(score_map, size=(orig_h, orig_w), mode="bilinear", align_corners=False)
        anomaly_map = anomaly_map.squeeze().cpu().numpy()

        # Normalize score map to 0 - 255 for visualization
        anomaly_map_norm = ((anomaly_map - anomaly_map.min()) / (anomaly_map.max() - anomaly_map.min() + 1e-8) * 255).astype(np.uint8)
        heatmap = cv2.applyColorMap(anomaly_map_norm, cv2.COLORMAP_JET)

        anomaly_score = float(np.max(anomaly_map))
        return anomaly_score, anomaly_map_norm, heatmap
