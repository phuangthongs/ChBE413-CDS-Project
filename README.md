# Chemical Data Science Project – Group 1

**UIUC ChBE 413 – Data Science for Chemists and Chemical Engineers**  

**Group Members:** Eric Giavedoni, Yong-Cheol Cho, Siri Phuangthong  

---

## Abstract

Polymer electrolytes are a class of materials widely studied for applications in energy storage, such as batteries and fuel cells, due to their ability to conduct ions while maintaining mechanical stability. Predicting their ionic conductivity, however, remains a significant challenge, as it depends on a complex interplay of polymer structure, ion–polymer interactions, and environmental conditions.

Molecular dynamics (MD) simulations have been employed to model polymer dynamics and estimate properties critical for energy storage applications. While useful, these simulations are often computationally expensive and limited in their ability to explore large chemical spaces efficiently. To overcome these limitations, this project leverages molecular descriptors to capture relevant chemical and structural information, thereby enhancing predictive capabilities.

---

## Project Goals

Our primary objective is to optimize the prediction of ionic conductivity in polymer electrolytes. To achieve this, we will:

1. Enrich the dataset features using RDKit-generated molecular descriptors and polymer physics knowledge.  
2. Build and refine machine learning models identified in previous literature as effective for predicting ionic conductivity.  
3. Perform hyperparameter tuning and model optimization to improve prediction accuracy.  
4. Identify key polymer features that influence ionic conductivity and evaluate which machine learning approaches are most suitable for predicting electronic properties comparable to molecular dynamics simulations.

---

## Repository Contents

Our repository includes:

- **Project Paper** – Full documentation of methodology, results, and analysis.  
- **Featurization Code** – Scripts for generating molecular descriptors and feature engineering.  
- **Random Forest Model** – Code for training and validating RF models for ionic conductivity prediction.  
- **Graph Convolutional Neural Network Model** – Code for training and validating GCNN models for ionic conductivity prediction.  


