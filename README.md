
# Optic Disc & Cup Segmentation using U-Net (RIM-ONE-DL Dataset)

This project implements a deep learning-based solution to **segment the optic disc and optic cup** from retinal fundus images using the **U-Net architecture**. The goal is to assist in early detection of **glaucoma** by analyzing the **cup-to-disc ratio (CDR)**. The project was developed using the [RIM-ONE-DL dataset](https://figshare.com/articles/dataset/RIM-ONE_DL_Dataset/12627993) and follows standardized evaluation protocols.

---

## 📌 Objectives

- Segment optic disc and optic cup regions using U-Net.
- Measure segmentation performance using **Dice coefficient**.
- Generate overlay visualizations of predictions.
- Create a PDF report for visual + metric summary.

---

## 📂 Project Structure

```

├── optic\_segmentation .ipynb          # Main Jupyter notebook
├── train.py                           # Model training script
├── model.py                           # U-Net architecture
├── dataset.py                         # Data loader and transforms
├── evaluate.py                        # Calculates Dice coefficients
├── visualize\_and\_save.py              # Generates overlay predictions
├── visualize\_predictions.py           # Visual display of model output
├── generate\_report.py                 # Creates PDF with overlays and scores
├── move\_masks.py                      # Organizes mask files
├── check\_sample\_match.py              # Debugging tool for samples
├── debug\_dataset\_check.py             # Dataset consistency checker
├── requirements.txt                   # Python dependencies
├── README.md                          # This documentation
├── partitioned\_randomly/              # Train/test split images
├── RIM-ONE\_DL\_reference\_segmentations/ # Ground truth masks
├── optic\_disc\_cup\_overlays.pdf        # PDF of overlay + Dice results

````

---

## ⚙️ How to Run

### 1. Install Dependencies
```bash
pip install -r requirements.txt
````

### 2. Run Jupyter Notebook

```bash
jupyter notebook "optic_segmentation .ipynb"
```

### 3. Or Execute Scripts from Terminal

```bash
python train.py
python evaluate.py
python visualize_and_save.py
```

---

## 📊 Results – Dice Coefficients

The model achieved the following segmentation performance on the test set:

* **Average Dice Score (Optic Disc):** \~0.94
* **Average Dice Score (Optic Cup):** \~0.65

| Optic Disc Dice | Optic Cup Dice |
| --------------- | -------------- |
| 0.9584          | 0.8797         |
| 0.9512          | 0.5542         |
| 0.9347          | 0.6251         |
| 0.9652          | 0.8857         |
| 0.9575          | 0.7657         |
| 0.9467          | 0.9138         |
| 0.9474          | 0.9260         |

📎 [Click to view PDF report](optic_disc_cup_overlays.pdf)

---

## 📁 Dataset

* **Name:** RIM-ONE-DL
* **Source:** [figshare – RIM-ONE\_DL Dataset](https://figshare.com/articles/dataset/RIM-ONE_DL_Dataset/12627993)
* **Usage Note:** Data was partitioned randomly and evaluation was done on a hold-out test set using provided ground truth masks.

---

## ✍️ Credits

* **Author:** Harshini Akunuri
* **Affiliation:** Programming Challenge for USC Roski Eye Institute
* **Advisor:** Dr. Benjamin Y. Xu, MD, PhD – Associate Professor of Ophthalmology, Director of AI in Ophthalmology

---

## 📬 Contact

* 📧 [harshiniakunuri59@gmail.com](mailto:harshiniakunuri59@gmail.com)
* 🔗 [LinkedIn – Harshini Akunuri](https://www.linkedin.com/in/harshini-akunuri/)

---

## 🧠 Keywords

`U-Net`, `Glaucoma Detection`, `Optic Disc Segmentation`, `Optic Cup`, `Fundus Image`, `Medical Imaging`, `PyTorch`, `Dice Score`, `RIM-ONE-DL`


---

