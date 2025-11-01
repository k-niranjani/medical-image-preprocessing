# 🩺 Eye Preprocessing Pipeline

A modular preprocessing toolkit for fundus and medical imaging datasets.  
Designed for reproducible, plug-and-play use with minimal setup.

---

## 📂 Project Structure

```text
EYE-PREPROCESSING/
│
├── transforms/                # Individual preprocessing modules
│   ├── crop_dark_borders.py
│   ├── circle_crop.py
│   ├── clahe.py
│   ├── resize.py
│   ├── unsharp_mask.py
│   └── NEW_TRANSFORM_GUIDE.md   # How to add your own transform
│
├── pipeline_utils.py          # Handles run organization and naming
├── utils.py                   # Common helper utilities (I/O, visualization)
├── preprocessing_main.ipynb   # Main notebook for preview & batch execution
├── requirements.txt           # Dependencies
├── .gitignore                 # Ignore build artifacts and cache
└── LICENSE                    # Open-source license
```

---

## ⚙️ Installation

```bash
git clone https://github.com/<yourusername>/eye-preprocessing.git
cd eye-preprocessing
pip install -r requirements.txt
```

---

## 🧩 Usage

Open `preprocessing_main.ipynb` in Jupyter or VS Code:

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

---

## 📦 Requirements

```text
numpy
pandas
opencv-python
matplotlib
scikit-image
tqdm
Pillow
```

(Optional: add `tensorflow`, `keras`, or `imgaug` if you plan to extend.)

---

## 🖋️ License

MIT License © 2025 Niranjani
