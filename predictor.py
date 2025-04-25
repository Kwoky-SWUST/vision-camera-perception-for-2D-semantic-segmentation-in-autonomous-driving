import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import cv2
import torchvision.transforms as transforms

# Define a simple MLP network with reduced complexity
class SimpleMLP(nn.Module):
    def __init__(self, input_dim=384, num_classes=20):  # Assuming 20 classes
        super(SimpleMLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 128),  # Reduced number of units
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        # Flatten the spatial dimensions
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1)  # (B, C, H, W) -> (B, H, W, C)
        x = x.view(B, -1, C)  # (B, H*W, C)
        x = self.mlp(x)       # (B, H*W, num_classes)
        x = x.view(B, H, W, -1)  # (B, H, W, num_classes)
        x = x.permute(0, 3, 1, 2)  # (B, num_classes, H, W)
        return x

# Custom transform to resize and pad features and labels to the same size
class ResizeAndPadFeatures:
    def __init__(self, target_size=(224, 224)):  # Smaller target size
        self.target_size = target_size
        
    def __call__(self, features, labels):
        # Resize features to target size
        features = cv2.resize(features, (self.target_size[1], self.target_size[0]), interpolation=cv2.INTER_LINEAR)
        
        # Pad features to target size if needed
        features = np.pad(features, ((0, self.target_size[0] - features.shape[0]), 
                                     (0, self.target_size[1] - features.shape[1]),
                                     (0, 0)), mode='constant')
        
        # Resize labels to target size
        labels = cv2.resize(labels.astype(np.uint8), (self.target_size[1], self.target_size[0]), interpolation=cv2.INTER_NEAREST)
        
        # Pad labels to target size if needed
        labels = np.pad(labels, ((0, self.target_size[0] - labels.shape[0]), 
                                 (0, self.target_size[1] - labels.shape[1])), mode='constant', constant_values=0)
        
        return features, labels

# Custom dataset class with resize and pad
class SemanticKittiDataset(Dataset):
    def __init__(self, data_root, sequences, transform=None):
        self.data_root = data_root
        self.sequences = sequences
        self.transform = transform
        self.scan_ids = []
        
        # Collect all scan IDs from the specified sequences
        for seq in sequences:
            seq_path = os.path.join(data_root, seq, 'velodyne')
            self.scan_ids.extend([(seq, f.split('.')[0]) for f in os.listdir(seq_path) if f.endswith('.bin')])
    
    def __len__(self):
        return len(self.scan_ids)
    
    def __getitem__(self, idx):
        seq, scan_id = self.scan_ids[idx]
        seq_path = os.path.join(self.data_root, seq)
        
        # Load DINO features
        dino_features_path = os.path.join(seq_path, 'dino_features', f'{scan_id}.npy')
        dino_features = np.load(dino_features_path).astype(np.float32)
        
        # Load pseudo-labels
        pseudo_labels_path = os.path.join(seq_path, 'pseudo_labels', f'{scan_id}.png')
        pseudo_labels = cv2.imread(pseudo_labels_path, cv2.IMREAD_GRAYSCALE).astype(np.long)  # Convert to long integer
        
        # Normalize features
        dino_features = dino_features / np.linalg.norm(dino_features, axis=-1, keepdims=True)
        
        # Apply transform if specified
        if self.transform:
            dino_features, pseudo_labels = self.transform(dino_features, pseudo_labels)
        
        # Convert to tensor
        dino_features = torch.from_numpy(dino_features).permute(2, 0, 1)  # (C, H, W)
        pseudo_labels = torch.from_numpy(pseudo_labels).long()  # Ensure labels are long integers
        
        return dino_features, pseudo_labels

# Custom collate function to handle different sizes
def custom_collate(batch):
    # Find the maximum height and width in the batch
    max_h = max(item[0].shape[1] for item in batch)
    max_w = max(item[0].shape[2] for item in batch)
    
    # Pad each item to the maximum size
    padded_batch = []
    for features, labels in batch:
        # Pad features
        padded_features = torch.zeros((features.shape[0], max_h, max_w), dtype=features.dtype)
        padded_features[:, :features.shape[1], :features.shape[2]] = features
        
        # Pad labels
        padded_labels = torch.zeros((max_h, max_w), dtype=torch.long)  # Ensure labels are long integers
        padded_labels[:labels.shape[0], :labels.shape[1]] = labels
        
        padded_batch.append((padded_features, padded_labels))
    
    # Stack the padded items
    features = torch.stack([item[0] for item in padded_batch])
    labels = torch.stack([item[1] for item in padded_batch])
    
    return features, labels

# Function to calculate IoU
def calculate_iou(predicted, labels, num_classes):
    # Create confusion matrix
    confusion_matrix = torch.zeros((num_classes, num_classes), device=predicted.device)
    for t, p in zip(labels.view(-1), predicted.view(-1)):
        confusion_matrix[t.long(), p.long()] += 1

    # Calculate IoU for each class
    intersection = torch.diag(confusion_matrix)
    union = confusion_matrix.sum(dim=0) + confusion_matrix.sum(dim=1) - intersection
    iou = intersection.float() / (union.float() + 1e-6)
    return iou.mean().item()  # Return mIoU

# Training function
def train_model(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    total_correct = 0
    total_pixels = 0
    
    for batch_idx, (features, labels) in enumerate(train_loader):
        features, labels = features.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(features)  # (B, num_classes, H, W)
        loss = criterion(outputs, labels)  # (B, num_classes, H, W) and (B, H, W)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # Calculate accuracy
        _, predicted = torch.max(outputs, 1)
        total_correct += (predicted == labels).sum().item()
        total_pixels += labels.numel()
    
    accuracy = total_correct / total_pixels
    return total_loss / len(train_loader), accuracy

# Testing function
def test_model(model, test_loader, criterion, device, num_classes):
    model.eval()
    total_loss = 0
    total_correct = 0
    total_pixels = 0
    iou_sum = 0
    
    with torch.no_grad():
        for batch_idx, (features, labels) in enumerate(test_loader):
            features, labels = features.to(device), labels.to(device)
            
            outputs = model(features)  # (B, num_classes, H, W)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            # Calculate accuracy
            _, predicted = torch.max(outputs, 1)
            total_correct += (predicted == labels).sum().item()
            total_pixels += labels.numel()
            
            # Calculate IoU
            iou = calculate_iou(predicted, labels, num_classes)
            iou_sum += iou
    
    accuracy = total_correct / total_pixels
    miou = iou_sum / len(test_loader)
    return total_loss / len(test_loader), accuracy, miou

# Visualization function
def visualize_predictions(model, data_root, sequence, scan_id, device, target_size=(224, 224)):
    seq_path = os.path.join(data_root, sequence)
    
    # Load DINO features
    dino_features_path = os.path.join(seq_path, 'dino_features', f'{scan_id}.npy')
    dino_features = np.load(dino_features_path).astype(np.float32)
    
    # Load original image
    image_path = os.path.join(seq_path, 'image_2', f'{scan_id}.png')
    image = cv2.imread(image_path)
    
    # Load pseudo-labels
    pseudo_labels_path = os.path.join(seq_path, 'pseudo_labels', f'{scan_id}.png')
    pseudo_labels = cv2.imread(pseudo_labels_path, cv2.IMREAD_GRAYSCALE).astype(np.long)
    
    # Normalize features
    dino_features = dino_features / np.linalg.norm(dino_features, axis=-1, keepdims=True)
    
    # Resize features and pseudo-labels to target size
    dino_features_resized = cv2.resize(dino_features, (target_size[1], target_size[0]), interpolation=cv2.INTER_LINEAR)
    pseudo_labels_resized = cv2.resize(pseudo_labels.astype(np.uint8), (target_size[1], target_size[0]), interpolation=cv2.INTER_NEAREST)
    
    # Convert to tensor
    dino_features_tensor = torch.from_numpy(dino_features_resized).permute(2, 0, 1).unsqueeze(0).to(device)  # (1, C, H, W)
    
    # Make prediction
    model.eval()
    with torch.no_grad():
        outputs = model(dino_features_tensor)  # (1, num_classes, H, W)
        _, predicted = torch.max(outputs, 1)
        predicted = predicted.squeeze(0).cpu().numpy()
    
    # Resize the original image to match the target size
    image_resized = cv2.resize(image, (target_size[1], target_size[0]), interpolation=cv2.INTER_LINEAR)
    
    # Visualize results
    cmap = create_semkitti_label_colormap()
    predicted_vis = cmap[predicted]
    pseudo_labels_vis = cmap[pseudo_labels_resized]
    
    # Overlay predictions on the resized original image
    overlay = cv2.addWeighted(image_resized, 0.5, predicted_vis, 0.5, 0)
    
    return image_resized, pseudo_labels_vis, overlay

def create_semkitti_label_colormap(): 
    """Creates a label colormap used in SEMANTICKITTI segmentation benchmark.

    Returns:
        A colormap for visualizing segmentation results in BGR format.
    """
    colormap = np.zeros((256, 3), dtype=np.uint8)
    colormap[0] = [0, 0, 0]
    colormap[1] = [245, 150, 100]       # "car"
    colormap[2] = [245, 230, 100]       # "bicycle"
    colormap[3] = [150, 60, 30]         # "motorcycle"
    colormap[4] = [180, 30, 80]         # "truck"
    colormap[5] = [255, 0, 0]           # "other-vehicle"
    colormap[6] = [30, 30, 255]         # "person"
    colormap[7] = [200, 40, 255]        # "bicyclist"
    colormap[8] = [90, 30, 150]         # "motorcyclist"
    colormap[9] = [255, 0, 255]         # "road"
    colormap[10] = [255, 150, 255]      # "parking"
    colormap[11] = [75, 0, 75]          # "sidewalk"
    colormap[12] = [75, 0, 175]         # "other-ground"
    colormap[13] = [0, 200, 255]        # "building"
    colormap[14] = [50, 120, 255]       # "fence"
    colormap[15] = [0, 175, 0]          # "vegetation"
    colormap[16] = [0, 60, 135]         # "trunk"
    colormap[17] = [80, 240, 150]       # "terrain"
    colormap[18] = [150, 240, 255]      # "pole"
    colormap[19] = [0, 0, 255]          # "traffic-sign"
    return colormap

if __name__ == "__main__":
    # Configuration
    data_root = "data_semantickitti"
    train_sequences = ["00", "01", "02", "03", "04", "05", "06", "07", "09", "10"]
    test_sequence = "08"
    batch_size = 2  # Reduced batch size
    num_epochs = 10
    learning_rate = 0.001
    num_classes = 20  # Update this based on the actual number of classes in your dataset
    
    # Define target size for resizing and padding
    target_size = (224, 224)  # Smaller target size
    
    # Define transform
    transform = ResizeAndPadFeatures(target_size=target_size)
    
    # Initialize model, loss function, and optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleMLP(input_dim=384, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Create datasets and dataloaders
    train_dataset = SemanticKittiDataset(data_root, train_sequences, transform=transform)
    test_dataset = SemanticKittiDataset(data_root, [test_sequence], transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate)
    
    # Train the model
    for epoch in range(num_epochs):
        train_loss, train_acc = train_model(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc, test_miou = test_model(model, test_loader, criterion, device, num_classes)
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}")
        print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}, Test mIoU: {test_miou:.4f}")
    
    # Visualize predictions
    # Visualization for training image
    train_seq = "00"
    train_scan_id = "000000"
    train_image, train_pseudo_labels, train_overlay = visualize_predictions(model, data_root, train_seq, train_scan_id, device, target_size)
    
    # Visualization for testing image
    test_seq = "08"
    test_scan_id = "000000"  # Assuming this scan exists in the test sequence
    test_image, test_pseudo_labels, test_overlay = visualize_predictions(model, data_root, test_seq, test_scan_id, device, target_size)
    
    # Save visualizations
    cv2.imwrite("train_image.png", train_image)
    cv2.imwrite("train_pseudo_labels.png", train_pseudo_labels)
    cv2.imwrite("train_overlay.png", train_overlay)
    cv2.imwrite("test_image.png", test_image)
    cv2.imwrite("test_pseudo_labels.png", test_pseudo_labels)
    cv2.imwrite("test_overlay.png", test_overlay)
    
    # Save the model
    torch.save(model.state_dict(), "semantic_segmentation_mlp.pth")