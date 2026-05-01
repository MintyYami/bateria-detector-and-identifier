# 🧬 Decoding Sepsis Pathogen Identity Through Fluorescent Images

## 📍 BioImage Hackathon — University of Warwick

---

## 🚀 Overview

This project was developed during the **BioImage Hackathon at the University of Warwick**, with the goal of transforming **small-scale fluorescent microscopy datasets** into **large, structured, machine-learning-ready data**.

We address a key bottleneck in biomedical imaging:

> **Limited image data → insufficient training data for machine learning**

### 💡 Solution

We built an automated pipeline that:
- Segments bacterial regions from fluorescent images
- Extracts morphological and intensity-based features
- Matches objects across imaging channels
- Converts raw images into structured tabular datasets

📊 **Result:**  
From **134 images → 14,337 rows** of trainable data

---

## 🧠 Key Idea

Instead of training ML models directly on images, we:
1. Extract biologically meaningful features  
2. Represent each bacterium as a data point  
3. Scale dataset size through feature-level expansion  

---

## 🔬 Pipeline

### 1. Image Processing
- Background subtraction
- Intensity normalization
- Thresholding (Otsu-based)
- Morphological cleaning

### 2. Segmentation
- Connected-component labeling
- Area-based filtering

### 3. Feature Extraction

Using region properties from `skimage.measure.regionprops`:

**Morphology:**
- `major_length`
- `minor_length`
- `length_ratio` (elongation)
- `eccentricity`
- `solidity`

**Intensity:**
- `mean_intensity`
- `max_intensity`

---

### 4. Multi-Channel Matching

Objects are matched between fluorescent channels using:
- Centroid proximity (KD-tree)
- Greedy nearest-neighbour matching

This enables:
- Detection of shared vs channel-specific bacteria
- Cross-channel feature comparison

---

### 5. Data Structuring

We convert wide channel-specific data into clean ML-ready format:
in_640 | in_488 | major_length | minor_length | length_ratio | eccentricity | solidity | mean_intensity | max_intensity


Each row = **one bacterium**

---

## 📦 Output

Final dataset:
- Fully numeric
- Label-ready
- Scalable for ML

### Example

| in_640 | in_488 | major_length | minor_length | length_ratio | eccentricity | solidity | mean_intensity | max_intensity |
|--------|--------|--------------|--------------|--------------|--------------|----------|----------------|---------------|
| True| True| 9.712993718118552| 3.801798442728979| 2.5548418304885265|0.9202148363264211|0.9333333333333333|2037.25,2566.0|
| True| False| 26.04119537045755|9.761132986938414|2.6678455645082786|0.9270918265019187|0.8949771689497716|2783.8469387755104,4540.0|

---

## 🧪 Technologies Used

- Python  
- NumPy  
- Pandas  
- scikit-image  
- SciPy (KDTree)  

---

## 📈 Impact

### ✅ Achievements
- Automated feature extraction from microscopy images  
- Increased dataset size by **~100x**  
- Eliminated manual annotation for feature engineering  

### 🔥 Why it matters
- Enables ML with **limited imaging data**  
- Reduces dependency on expert labelling  
- Speeds up pathogen identification workflows  

---

## 🔮 Future Development

### 1. Scaling Data from Small Datasets
- Apply pipeline to other microscopy datasets  
- Improve generalisation across imaging conditions  

### 2. Automated Bacteria Classification
- Train ML models on extracted features  
- Classify bacterial morphology and type  
- Move toward real-time diagnostic tools  

### 3. Improved Matching & Segmentation
- Smarter matching beyond centroid distance  
- Handle overlapping or clustered bacteria  

---

## 🧑‍🔬 Authors

Aishwarya Bhatta
Jiqiu Hu
Jialu Li
Mint Cheepchiewcharnchai

Developed during the **BioImage Hackathon**
University of Warwick  
