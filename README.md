# MA_Mike

This repository contains the code, data, and presentation material for my Master's thesis on **Causal Modeling with Neural Networks and Individualized Treatment Effect Estimation **. The work includes multiple experiments using both simulated and real-world clinical trial data, specifically the International Stroke Trial (IST) dataset.

---

## 📁 Folder Structure

### 🔧 `code/`
Scripts used for simulations and data analysis.

#### Main Experiments:
- `TRAM_DAG_simulation.R`  
  *Experiment 1: TRAM-DAG (simulation)*

- `IST_data_processing.Rmd`  
  *Experiment 2: ITE on International Stroke Trial (IST)*

- `TRAM_DAG_ITE_Stroke_IST.R`  
  *Experiment 2: ITE on International Stroke Trial (IST)*

- `ITE_simulation.R`  
  *Experiment 3: ITE model robustness in RCTs (simulation)*

- `ITE_observational_simulation.R`  
  *Experiment 4: ITE estimation with TRAM-DAGs (simulation)*

- `ITE_RCT_simulation.R`  
  *Experiment 4: ITE estimation with TRAM-DAGs (simulation)*

#### Additional / Supporting Code:
- `ITE_simulation_CS.R`  
  *Appendix: Modeling interaction with complex shift*

- `tram_Fz_link_function_choice.R`  
  *Examples for different choices of the link function Fz on model fit*

- `analysis_of_scaling.Rmd`  
  *Analysis of the impact of scaling on parameter interpretation*

- `experiment_5_all_ordinal_dummy_encoded_real_data_linear.R`  
  *Not in thesis: Example of dummy encoding in TRAM-DAGs, uses `utils-dummy`*

- `trafo-viz.R`  
  *For plotting TRAM visualizations (code from Sick et. al (2025))*

##### `utils/`  
Subfolder containing helper functions used across the scripts above.

---

### 📂 `runs/`
Saved TRAM-DAG model runs, including:
- Neural Network parameters and training loss
- Model results

---

### 📂 `data/`

- `IST_model_data.RData`  
  *Preprocessed data for Experiment 2 (IST)*

- `IST_dataset_2011.csv`  
  *Raw data from the International Stroke Trial*  
  Source: [https://trialsjournal.biomedcentral.com/articles/10.1186/1745-6215-12-101#Sec8](https://trialsjournal.biomedcentral.com/articles/10.1186/1745-6215-12-101#Sec8)

- Additional example datasets (e.g., Teleconnections)  
  *Not used in the final thesis*

---

### 📂 `presentation_report/`

- `thesis/`  
  *Final version of the Master thesis*

- `final_presentation/`  
  *Slides for the Master exam*

- `intermediate_presentation/`  
  *Slides for the intermediate presentation*

- `literature/`  
  *Bibliography and reference material used in presentations*

- `report/`  
  *Old thesis template*

- `template_presentation/`  
  *Old presentation template*

---

## 📌 Notes

- This project was developed as part of Mike Krähenbühl's Master's thesis in Biostatistics at the University of Zurich (UZH).

---

