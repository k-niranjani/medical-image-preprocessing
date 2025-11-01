# 🩺 Medical Image Preprocessing Pipeline

A modular preprocessing toolkit for medical imaging datasets.  
Designed for reproducible, plug-and-play use with minimal setup.

---

## 📂 Project Structure

```text
MEDICAL-IMAGE-PREPROCESSING/
│
├── transforms/                         # Individual preprocessing modules
│   ├── crop_dark_borders.py
│   ├── circle_crop.py
│   ├── clahe.py
│   ├── resize.py
│   ├── unsharp_mask.py
│   └── NEW_TRANSFORM_GUIDE.md          # How to add your own transform
│
├── pipeline_utils.py                   # Handles run organization and naming
├── utils.py                            # Common helper utilities (I/O, visualization)
├── MEDICAL_IMAGE_PREPROCESSING.ipynb   # Main notebook for preview & batch execution
├── requirements.txt                    # Dependencies
├── .gitignore                          # Ignore build artifacts and cache
└── LICENSE                             # Open-source license
```

---

## ⚙️ Installation

```bash
git clone https://github.com/seratonini/medical-image-preprocessing.git
cd medical-image-preprocessing
pip install -r requirements.txt
```

---

## 🧩 Usage

Open `MEDICAL_IMAGE_PREPROCESSING.ipynb` in Jupyter or VS Code:

1. Add **train/test image folder paths** under the input section.  
2. Choose one or more transforms (e.g. `clahe`, `resize`, `circle_crop`).  
3. Set a **RUN_NAME** (e.g. `11012025`) — the date in `MMDDYYYY` format.  
4. Run cells to:
   - Visualize previews for each transform.
   - Apply and save results to new folders.

---

## 🧰 Adding a New Transform

Follow the step-by-step guide in  
[`transforms/NEW_TRANSFORM_GUIDE.md`](transforms/NEW_TRANSFORM_GUIDE.md).  
All new transforms become available automatically once imported in  
`transforms/__init__.py`.


## 🖋️ License

MIT License © 2025 Niranjani K.
