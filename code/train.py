import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.functional as TF
import random
import numpy as np

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
from predict import postprocess_mask, save_mask
from LISA import LISA


os.environ["CUDA_VISIBLE_DEVICES"] = "3"

class FaceParsingDataset(Dataset):
    def __init__(self, root_dir, mode=None, transform=None):
        self.root_dir = root_dir
        self.mode = mode
        self.transform = transform
        if self.mode == 'train':
            self.images_dir = os.path.join(root_dir, 'train_image')
            self.masks_dir = os.path.join(root_dir, 'train_mask')
        elif self.mode == 'val':
            self.images_dir = os.path.join(root_dir, 'val_image')
            self.masks_dir = os.path.join(root_dir, 'val_mask')
        else:
            self.images_dir = os.path.join(root_dir, 'test_image')
            self.masks_dir = os.path.join(root_dir, 'test_mask')
        self.image_list = os.listdir(self.images_dir)
        self.class_values = list(range(19))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_name = self.image_list[idx]
        img_path = os.path.join(self.images_dir, img_name)
        mask_path = os.path.join(self.masks_dir, img_name.replace('.jpg', '.png'))

        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path)
        mask = np.array(mask).astype('long')

        masks = [(mask == v) for v in self.class_values]
        one_hot_mask = np.stack(masks, axis=0)
        
        mask = torch.from_numpy(mask)
        one_hot_mask = torch.from_numpy(one_hot_mask).float()
 
        if self.transform:
            image = self.transform(image)

        return image, mask, one_hot_mask

def dice_loss(input, target):
    input = input.contiguous().view(input.size()[0], -1)
    target = target.contiguous().view(target.size()[0], -1).float()
    a = torch.sum(input * target)           # |X⋂Y|
    b = torch.sum(input*input)    # |X|
    c = torch.sum(target*target)  # |Y|
    d = (2 * a) / (b + c + 0.0001)
    return 1-d


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0

    pbar = tqdm(total=len(loader), desc="Training", leave=False)
    for batch in loader:
        images, masks, one_hot_masks = batch
        images, masks, one_hot_masks = images.to(device), masks.to(device), one_hot_masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)        

        loss1 = criterion(outputs, masks)
        loss2 = dice_loss(outputs, one_hot_masks)

        loss = loss1 + loss2
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        pbar.update(1)
        pbar.set_postfix({'batch_loss': f'{loss.item():.4f}'})


    pbar.close()

    return total_loss / len(loader)

def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0

    pbar = tqdm(total=len(loader), desc="Validating", leave=False)
    with torch.no_grad():
        for batch in loader:
            images, masks, one_hot_masks = batch
            images, masks, one_hot_masks = images.to(device), masks.to(device), one_hot_masks.to(device)

            outputs = model(images)
            loss1 = criterion(outputs, masks)
            loss2 = dice_loss(outputs, one_hot_masks)

            loss = loss1 + loss2
            total_loss += loss.item()

            pbar.update(1)
            pbar.set_postfix({'batch_loss': f'{loss.item():.4f}'})


    pbar.close()

    return total_loss / len(loader)


def main():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = FaceParsingDataset(root_dir='', mode='train', transform=transform)
    val_dataset = FaceParsingDataset(root_dir='', mode='val', transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=32, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, num_workers=4)

    device = torch.device('cuda:0')
    model = LISA().to(device)
    #model.load_state_dict(torch.load('', map_location=device))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)


    num_epochs = 100
    best_val_loss = float('inf')
    patience = 20
    patience_counter = 0
    gradient_accumulation_steps = 1

    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate(model, val_loader, criterion, device)

 
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

        scheduler.step(val_loss)


        if epoch > 10:
            if epoch % 5 == 0:
                torch.save(model.state_dict(), ''+str(epoch)+'_'+str(val_loss)+'.pth')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), '')
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(best_val_loss)
            print("Early stopping triggered")
            break

if __name__ == "__main__":
    main()