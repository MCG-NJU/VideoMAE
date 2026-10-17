import torch
from modeling_pretrain import pretrain_videomae_base_patch16_224  # مدل pretrain VideoMAE

CKPT_PATH = "pretrained/k400_vitb_e800_pretrain.pth"


def load_videomae_from_k400():
    print("==== Load VideoMAE (K400 pretrain) on CPU ====")
    print(f"Loading checkpoint from: {CKPT_PATH}")

    # 1) ساختن مدلِ MAE مطابق مدل K400
    model = pretrain_videomae_base_patch16_224()

    # 2) لود کردن checkpoint
    ckpt = torch.load(CKPT_PATH, map_location="cpu")

    # بعضی چک‌پوینت‌ها داخل key به اسم 'model' ذخیره شدن
    state_dict = ckpt.get("model", ckpt)

    msg = model.load_state_dict(state_dict, strict=False)

    print("=> Checkpoint loaded.")
    print("   Missing keys   :", len(msg.missing_keys))
    print("   Unexpected keys:", len(msg.unexpected_keys))

    return model


def main():
    model = load_videomae_from_k400()

    # فقط برای تست: تعداد پارامترها را چاپ کن
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {n_params/1e6:.2f} M")

    # هیچ forward واقعی نمی‌زنیم، فقط مطمئن می‌شویم لود بدون خطا بوده
    print("Model loaded successfully and ready to use.")


if __name__ == "__main__":
    main()
