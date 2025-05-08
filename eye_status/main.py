import os
import traceback
from model import train_model
from utils import EyeDataset, infer_openness
from torch.utils.data import DataLoader
from torchvision import transforms

if __name__ == '__main__':
    # Step 1: Load pre-extracted eye images and labels
    print("\n===== Step 1: Loading eye images and labels =====")
    eye_image_dir = 'eye_images'
    label_file = 'labels.csv'
    if not os.path.exists(eye_image_dir):
        print(f"Error: Eye image directory {eye_image_dir} does not exist")
        exit(1)
    if not os.path.exists(label_file):
        print(f"Error: Label file {label_file} does not exist")
        exit(1)

    # Step 2: Train the model
    print("\n===== Step 2: Training the model =====")
    try:
        transform = transforms.Compose([
            transforms.Resize((48, 48)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        dataset = EyeDataset(eye_image_dir, label_file, transform=transform)
        print(f"Successfully loaded {len(dataset)} eye images")
        dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0)
        model = train_model(dataloader, epochs=10)
    except Exception as e:
        print(f"Training failed: {str(e)}")
        print("Detailed error message:")
        traceback.print_exc()
        exit(1)

    # Step 3: Predict eye openness for images in test/
    print("\n===== Step 3: Predicting eye openness for test images =====")
    test_dir = 'test'
    if os.path.exists(test_dir):
        for file in os.listdir(test_dir):
            if file.lower().endswith(('.jpg', '.png')):
                path = os.path.join(test_dir, file)
                try:
                    score = infer_openness(model, path)
                    print(f"Image {file} eye openness: {score:.1f}%")
                except Exception as e:
                    print(f"Inference failed for image {file}: {str(e)}")
    else:
        print(f"Test directory not found: {test_dir}")