import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
import numpy as np

# === Charset ===
CHARS = "0123456789/:~ "
CHAR2IDX = {c: i for i, c in enumerate(CHARS)}

# === Dataset ===
class OCRDataset(Dataset):
    def __init__(self, path="data/train"):
        self.files = [f for f in os.listdir(path) if f.endswith(".png")]
        self.path = path
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    def __len__(self): return len(self.files)
    def __getitem__(self, idx):
        img_name = self.files[idx]
        label_path = os.path.join(self.path, img_name.replace(".png", ".txt"))
        with open(label_path) as f:
            label = f.read().strip()
        img = cv2.imread(os.path.join(self.path, img_name), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (256, 64))
        img = self.transform(img)
        target = torch.LongTensor([CHAR2IDX[c] for c in label if c in CHAR2IDX])
        return img, target

# === Model ===
class CRNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2,2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2,2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d((2,1))
        )
        self.rnn = nn.LSTM(128*8, 128, num_layers=2, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(128*2, num_classes + 1)  # +1 for CTC blank

    def forward(self, x):
        x = self.cnn(x)
        b, c, h, w = x.size()
        x = x.permute(0, 3, 1, 2).contiguous()
        x = x.view(b, w, c*h)
        x, _ = self.rnn(x)
        x = self.fc(x)
        return x.log_softmax(2)

def train_model(epochs=10, batch_size=16, data_path="data/train"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CRNN(num_classes=len(CHARS))
    model.to(device)

    criterion = nn.CTCLoss(blank=len(CHARS))
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loader = DataLoader(OCRDataset(data_path), batch_size=batch_size, shuffle=True, num_workers=2, collate_fn=lambda b: zip(*b))

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for imgs, labels in tqdm(loader, desc=f"Epoch {epoch+1}"):
            imgs = torch.stack(list(imgs)).to(device)
            lbls = [l for l in labels]
            lbl_concat = torch.cat(lbls)
            lbl_lens = torch.IntTensor([len(l) for l in lbls])
            preds = model(imgs)
            input_lens = torch.IntTensor([preds.size(1)] * imgs.size(0))
            loss = criterion(preds.permute(1, 0, 2), lbl_concat, input_lens, lbl_lens)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Loss: {total_loss / len(loader):.4f}")

    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/baitoscan_crnn.pth")
    print("✅ Model saved to checkpoints/baitoscan_crnn.pth")
    return model

if __name__ == "__main__":
    train_model()
