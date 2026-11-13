import torch
from PIL import Image
from torchvision import transforms

from unet import UNet

device = torch.device("cpu")

model = UNet().to(device)

model.load_state_dict(
    torch.load(
        "models/unet.pth",
        map_location=device
    )
)

model.eval()

image_path = "dataset/cloudy/0.png"

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

img = Image.open(image_path).convert("RGB")

input_tensor = transform(img).unsqueeze(0)

with torch.no_grad():
    output = model(input_tensor)

output = output.squeeze(0)

# Important
output = torch.clamp(output, 0, 1)

output_img = transforms.ToPILImage()(output)

output_img.save(
    "outputs/prediction.png"
)

print("Saved prediction!")