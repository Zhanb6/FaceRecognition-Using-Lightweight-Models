import torch
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
import timm

print("Downloading MobileFaceNet backbone weights (MobileNetV2)...")
try:
    mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
    print("MobileNetV2 weights downloaded successfully!")
except Exception as e:
    print(f"Error downloading MobileNetV2: {e}")

print("\nDownloading EfficientNet-Lite0 backbone weights...")
try:
    timm.create_model('efficientnet_lite0', pretrained=True)
    print("EfficientNet-Lite0 weights downloaded successfully!")
except Exception as e:
    print(f"Error downloading EfficientNet-Lite0: {e}")

print("\nAll downloads finished.")
