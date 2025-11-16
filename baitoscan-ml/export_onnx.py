import torch
import onnx
from train_crnn import CRNN, CHARS

def export_model(ckpt_path="checkpoints/baitoscan_crnn_1.pth", out_path="baitoscan/public/models/baitoscan-crnn_1.onnx"):
    model = CRNN(num_classes=len(CHARS))
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    model.eval()

    dummy = torch.randn(1, 1, 50, 330)
    torch.onnx.export(
        model,
        dummy,
        out_path,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "N", 3: "W"}},
        opset_version=13
    )
    print(f"✅ Exported ONNX model to {out_path}")

if __name__ == "__main__":
    export_model()
