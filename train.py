import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import json  # Import json to save class names

def main():
    # Define Image Transformations with Data Augmentation
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load dataset
    train_dir = r"C:\Users\palwa\Desktop\fruitsclassify\food-360\train"
    val_dir = r"C:\Users\palwa\Desktop\fruitsclassify\food-360\test"

    train_data = datasets.ImageFolder(train_dir, transform=transform)
    val_data = datasets.ImageFolder(val_dir, transform=transform)

    # Save class names
    class_names = train_data.classes
    with open('class_names.json', 'w') as f:
        json.dump(class_names, f)

    # DataLoader with optimized settings
    train_loader = DataLoader(train_data, batch_size=16, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_data, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)

    # Load Pre-trained ResNet18 Model
    print("Loading pre-trained ResNet18 model...")
    model = models.resnet18(weights='DEFAULT')

    # Unfreeze all layers so that we fine-tune the entire model
    for param in model.parameters():
        param.requires_grad = True  # Unfreezing all layers

    # Replace the final fully connected layer and add dropout
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.5),  # Dropout layer with 50% probability
        nn.Linear(num_ftrs, len(train_data.classes))
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Define Loss Function and Optimizer with Weight Decay (L2 Regularization)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)  # Added weight decay (L2 regularization)

    # Optional: Define a learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    print("Training setup complete")

    # Training Loop with Debugging Information
    num_epochs = 10
    best_accuracy = 0.0

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs} started...")
        
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            if batch_idx % 10 == 0:  # Print every 10 batches
                print(f"Processing batch {batch_idx}/{len(train_loader)}")

            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        print(f"Train Loss: {epoch_loss:.4f}, Train Accuracy: {epoch_acc:.2f}%")

        # Validation phase
        print("Validation phase started...")
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0.0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_acc = 100 * val_correct / val_total
        val_loss /= len(val_loader)
        print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.2f}%')

        # Save the model if validation accuracy improves
        if val_acc > best_accuracy:
            best_accuracy = val_acc
            torch.save(model.state_dict(), 'best_fruits_model.pth')
            print(f"New best model saved with accuracy: {best_accuracy:.2f}%")

        # Step the scheduler
        scheduler.step()

    print(f'Training complete. Best accuracy: {best_accuracy:.2f}%')

if __name__ == '__main__':
    main()
