### Clinical Phenotype Classification/Prediction Algorithm

# Chest CT Report Multi-Label Classification Algorithm


## Overview
Multi-label classification model for chest CT reports to identify three diseases: Pneumonia, Cancer, and Pulmonary Embolism.
Fine-tuning code for BERT-based models including general domain and biomedical domain language models with their bilingual (Korean-English) variants.

## Pipeline
**1. Data Preprocessing & Labeling** (`preprocessing_labeling.py`)
   - Diagnosis code-based labeling
   - Text cleaning for CT reports
   - Rule-based labeling for multi-label annotation

**2. Model Development** (`train.py`)
   
   Pre-trained models used for fine-tuning are available from the repositories below.


   - **Base Models:**
     - Original versions: [KM-BERT](https://github.com/KU-RIAS/KM-BERT-Korean-Medical-BERT), [BioBERT](https://github.com/dmis-lab/biobert), [M-BERT](https://huggingface.co/google-bert/bert-base-multilingual-cased)
     - Bilingual(Korean-English) versions: [bi-KM-BERT, bi-BioBERT, bi-M-BERT](https://github.com/KU-RIAS/bi-medical-bert)
   
   
   - **Fine-tuning Process:**
     - Load pre-trained model and tokenizer
     - Add classification head for 3-label prediction
     - Train with binary cross-entropy loss for multi-label classification
     - Save model checkpoints and training history

## Requirements
### Environment
- Python 3.10
- CUDA  12.4 
- PyTorch 2.5.1

### Installation
```bash
pip install torch transformers pandas scikit-learn numpy tqdm
