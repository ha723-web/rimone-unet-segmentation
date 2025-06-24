# Optic Disc & Cup Segmentation with U-Net (RIM-ONE-DL)

This project performs semantic segmentation of the **optic disc** and **optic cup** from retinal fundus images using a **U-Net model**, trained on the [RIM-ONE-DL dataset](https://github.com/miag-ull/rim-one-dl?tab=readme-ov-file).

---

## Objective

- Identify and segment optic disc and cup regions from fundus images.
- Compute **Dice coefficients** for segmentation accuracy.
- Visualize overlays of predictions on test images.
- Export segmentation results to a downloadable PDF.


---

## How to Run

1. Install dependencies:

    pip install -r requirements.txt


2. Open the notebook:

    jupyter notebook optic_segmentation.ipynb
    

3. Run all cells in order:
    - Load and preprocess dataset
    - Load trained model
    - Predict on test images
    - Visualize overlays
    - Export and download results

---

## Metrics

- **Dice Coefficient (Disc)**: ~0.93  
- **Dice Coefficient (Cup)**: ~0.59

---

## Output

- Dice score report printed in notebook  
- PNG overlays in `overlays/`  
- PDF with all predictions: `output_images.pdf`  
- Inline PDF preview + download link in notebook   
## (After completion of the last cell you will seeing "Click to Download pdf link. So once you click that link, pdf will be downloaded in you local PC downloads")

---
 
## By Harshini Akunuri