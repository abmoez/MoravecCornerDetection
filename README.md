# MoravecCornerDetection

The **Moravec Corner Detection** algorithm is one of the earliest corner detection techniques. 
It detects interest points by evaluating how much a small window shifts in intensity in multiple directions. 

The algorithm slides a window across the image and calculates the intensity difference in **8 directions**.
If the **minimum difference** across all directions is **large**, the pixel is likely a corner.

### 📌 Steps:
1. Convert image to grayscale.
2. Slide a small window across the image.
3. Compare each window with shifted windows in 8 directions.
4. Compute the minimum of all shift differences as the corner score.
5. Apply thresholding and **non-maximum suppression** to localize corners.

