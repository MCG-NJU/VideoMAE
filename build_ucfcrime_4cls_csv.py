import os

# ======= تنظیم مسیرها (اینا رو با مسیرهای خودت چک کن) =======

# ریشه ویدیوها (جایی که پوشه‌های Arson, Explosion, ... هستند)
VIDEO_ROOT = r"D:\final_project\datasets\UCF_Crime"

# جایی که train_001.txt و test_001.txt هستند
SPLIT_ROOT = r"D:\final_project\datasets\UCF_Crime\annotations\Action_Regnition_splits"

# جایی که می‌خوای train.csv و val.csv ذخیره بشن
OUT_DIR = r"D:\final_project\VideoMAE\codes\VideoMAE\list_ucfcrime_4cls"


# ======= نگاشت پوشه‌ها به ۴ کلاس نهایی =======
# عددها رو هرطور خواستی ثابت نگه دار، فقط همیشه یکسان استفاده کن

CLASS_MAP = {
    "Normal_Videos_event": 0,  # NORMAL

    "Robbery": 1,              # ROBBERY

    "Explosion": 2,            # FIRE/EXPLOSION (Explosion + Arson)
    "Arson": 2,

    "Assault": 3,              # VIOLENCE (Assault + Fighting)
    "Fighting": 3,
}


def build_split(split_txt_name, out_csv_name):
    """یک فایل train_001.txt یا test_001.txt را می‌گیرد
    و فقط کلاس‌های مورد نظر را به CSV تبدیل می‌کند.
    """
    in_path = os.path.join(SPLIT_ROOT, split_txt_name)
    out_path = os.path.join(OUT_DIR, out_csv_name)

    os.makedirs(OUT_DIR, exist_ok=True)

    total_lines = 0
    used_lines = 0

    with open(in_path, "r") as f_in, open(out_path, "w", encoding="utf-8") as f_out:
        first_line = True
        for line in f_in:
            line = line.strip()
            if not line:
                continue

            # خط اول معمولا فقط "mp4" است، ردش می‌کنیم
            if first_line:
                first_line = False
                # اگر دوست داری چک کنی:
                # assert line == "mp4"
                continue

            total_lines += 1

            # مثال: "Explosion/Explosion001_x264.mp4"
            rel_path = line.replace("\\", "/")
            cls_name = rel_path.split("/")[0]

            # اگر کلاس توی ۴ کلاس ما نیست، نادیده بگیر
            if cls_name not in CLASS_MAP:
                continue

            label = CLASS_MAP[cls_name]

            # ساختن مسیر کامل روی دیسک
            abs_path = os.path.join(VIDEO_ROOT, *rel_path.split("/"))
            abs_path = os.path.normpath(abs_path)

            # نوشتن به فرمت VideoMAE: "abs_path label"
            f_out.write(f"{abs_path} {label}\n")
            used_lines += 1

    print(f"[{split_txt_name}] total={total_lines}, used={used_lines}, saved to {out_path}")


if __name__ == "__main__":
    # از fold اول استفاده می‌کنیم: train_001 به عنوان train، test_001 به عنوان val
    build_split("train_001.txt", "train.csv")
    build_split("test_001.txt", "val.csv")
    print("Done. You can now use list_ucfcrime_4cls/train.csv and val.csv in VideoMAE.")
