import torch
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt
import json

# Load class names from JSON file (saved during training)
with open('class_names.json', 'r') as f:
    class_names = json.load(f)

# Define the transformations (same as during training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load the saved model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the model with the same architecture used during saving
model = models.resnet18(weights='IMAGENET1K_V1')
num_ftrs = model.fc.in_features

# Modify the final fully connected layer to match training architecture
model.fc = torch.nn.Sequential(
    torch.nn.Dropout(p=0.5),  # Match the dropout used during training
    torch.nn.Linear(num_ftrs, len(class_names))  # Output features should match the number of classes
)

# Load the state dictionary with weights_only=True
state_dict = torch.load('best_fruits_model.pth', map_location=device)
model.load_state_dict(state_dict)

model = model.to(device)
model.eval()

# Function to classify an image
def classify_image(image_path):
    image = Image.open(image_path)
    image_tensor = transform(image)
    image_tensor = image_tensor.unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)
        predicted_class = predicted.item()
    
    return class_names[predicted_class], image

# Main function to run classification
if __name__ == "__main__":
    image_path = r'C:\Users\palwa\Desktop\fruitsclassify\food-360\test\Pear 2\r2_80_100.jpg'  # Replace with the path to your image
    result, image = classify_image(image_path)
    
    # Display the image with prediction
    plt.imshow(image)
    plt.title(f'The image is classified as: {result}')
    plt.axis('off')  # Hide axes
    plt.show()
