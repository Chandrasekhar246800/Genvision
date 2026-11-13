import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

from attention_unet import AttentionUNet

# Device
device = torch.device("cpu")

# Load Model
model = AttentionUNet().to(device)

model.load_state_dict(
    torch.load(
        "models/attention_unet.pth",
        map_location=device
    )
)

model.eval()

# Image Transform
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# Load Cloudy Image
image = Image.open(
    "dataset/cloudy/0.png"
).convert("RGB")

input_tensor = transform(image).unsqueeze(0).to(device)

# Prediction
with torch.no_grad():
    output = model(input_tensor)

output = output.squeeze(0).cpu()

output = transforms.ToPILImage()(output)

# Save Result
output.save(
    "outputs/attention_prediction.png"
)

print("=" * 50)
print("Prediction Saved Successfully!")
print("Location : outputs/attention_prediction.png")
print("=" * 50)

# Display Result
plt.figure(figsize=(5,5))
plt.imshow(output)
plt.axis("off")
plt.title("Attention U-Net Prediction")
plt.show()