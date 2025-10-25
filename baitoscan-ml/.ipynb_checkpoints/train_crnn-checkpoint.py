import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np, cv2, os, random
from tqdm import tqdm
from dataset_gen import CHARS, CHAR2IDX

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
        label = open(label_path).read().strip()
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
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2,2),  # 32x128x32
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2,2),  # 64x64x16
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d((2,1)), # 128x32x16
        )
        self.rnn = nn.LSTM(128*32, 128, num_layers=2, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(128*2, num_classes+1)  # +1 for CTC blank

    def forward(self, x):
        x = self.cnn(x)        # (B, C, H, W)
        b, c, h, w = x.size()
        x = x.permute(0, 3, 1, 2).contiguous()  # (B, W, C, H)
        x = x.view(b, w, c*h)  # sequence along width
        x, _ = self.rnn(x)
        x = self.fc(x)
        return x.log_softmax(2)

# === Training ===
device = "cuda" if torch.cuda.is_available() else "cpu"
model = CRNN(num_classes=len(CHARS))
model.to(device)

criterion = nn.CTCLoss(blank=len(CHARS))
optimizer = optim.Adam(model.parameters(), lr=1e-3)

def collate_batch(batch):
    imgs, labels = zip(*batch)
    return imgs, labels

train_loader = DataLoader(
    OCRDataset(), batch_size=16, shuffle=True,
    num_workers=0, collate_fn=collate_batch
)
for epoch in range(10):
    model.train()
    total_loss = 0
    for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        imgs = torch.stack(list(imgs)).to(device)
        lbls = [l for l in labels]
        lbl_concat = torch.cat(lbls)
        lbl_lens = torch.IntTensor([len(l) for l in lbls])
        preds = model(imgs)
        input_lens = torch.IntTensor([preds.size(1)] * imgs.size(0))
        loss = criterion(preds.permute(1,0,2), lbl_concat, input_lens, lbl_lens)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Loss: {total_loss/len(train_loader):.4f}")

torch.save(model.state_dict(), "baitoscan_crnn.pth")
