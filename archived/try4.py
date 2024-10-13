import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

# Load the trained model
model = torch.load(
    "C:/Users/hassa/OneDrive - Umich/Desktop/2024 HACKATHON/16_net_G.pth",
    map_location=torch.device("cpu"),
)
model.eval()  # Set the model to evaluation mode

# Define preprocessing transformations (resize to match your training data)
transform = transforms.Compose(
    [
        transforms.Resize((256, 256)),  # Resize to the appropriate size
        transforms.ToTensor(),  # Convert to tensor
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # Normalize
    ]
)

# Load and preprocess your input image
img = Image.open(
    "C:/Users/hassa/OneDrive - Umich/Desktop/2024 HACKATHON/foggyimage_person.jpg"
).convert("RGB")
img_tensor = transform(img).unsqueeze(0)  # Add batch dimension

# Perform inference
with torch.no_grad():
    output = model(img_tensor)

# Post-process and display the result
output = output.squeeze(0).detach().cpu()  # Remove batch dimension
output = (output * 0.5 + 0.5).clamp(0, 1)  # De-normalize the output to [0, 1]

# Convert tensor to image
output_img = transforms.ToPILImage()(output)
output_img.save("output_image.jpg")  # Save the output image

# Optionally, display the result
plt.imshow(output_img)
plt.axis("off")
plt.show()
