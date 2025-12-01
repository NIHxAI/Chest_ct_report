"""
preprocessing_labeling.py
Chest CT Report Preprocessing and Labeling Pipeline
"""

import pandas as pd
import re
from tqdm import tqdm
from datetime import datetime, timedelta

# ==========================================
# Load Data
# ==========================================
df = pd.read_excel('chest_ct_data.xlsx')

# ==========================================
# 1. Diagnosis Code-based Labeling 
# ==========================================
# Check if diagnosis occurred within 1 month of report date
# PE_1m, Cancer_1m, PN_1m columns indicate if diagnosis was within 1 month

# Convert to string type for comparison
df['PE_1m'] = df['PE_1m'].astype(str)
df['Cancer_1m'] = df['Cancer_1m'].astype(str) 
df['PN_1m'] = df['PN_1m'].astype(str)

# Create labels based on 1-month window
# '포함' means the diagnosis was within 1 month
df['PE'] = df['PE_1m'].apply(lambda x: 1 if '포함' == x else 0)
df['Cancer'] = df['Cancer_1m'].apply(lambda x: 1 if '포함' == x else 0)
df['PN'] = df['PN_1m'].apply(lambda x: 1 if '포함' == x else 0)

# Create multi-label class combination
df['class'] = df['PE'].astype(str) + df['Cancer'].astype(str) + df['PN'].astype(str)

# ==========================================
# 2. Text Cleaning for CT Reports
# ==========================================
def preprocess_text(text):
    """
    Clean CT report text by removing special characters and normalizing whitespace
    Keep only Korean, English alphabets and spaces
    """
    # Remove all characters except Korean, English, and spaces
    text = re.sub(r"[^a-zA-Z가-힣\s]", "", text)
    # Replace multiple spaces with single space
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()

# Apply text preprocessing
df['text'] = df['EXRS_CTN'].apply(preprocess_text)  # EXRS_CTN is the report text column

# ==========================================
# 3. Rule-based Labeling for Multi-label Annotation
# ==========================================

# Define keywords for each disease
pneumonia_keywords = ["bronchopneumonia", "pneumonitis", "pneumonia"]
embolism_keywords = ['pulmonary thromboembolism', 'pulmonary embolism', 'pul embolism', 'pte', 'embolism']
cancer_keywords = ["mass", "ggn", "ggo", "lung", "lymph nodes", "lymph node", "ln", "lns", 
                   "lymphoma", "metastasis", "metastases", "mets", "recurrence", "metastaseis", 
                   "malignancy", "malignant", "pulmonary", "both", "lne", "mediastinal", 
                   "lesion", "pleural effusion", "pleural", "lungs", "carcinoma", "lesions", 
                   "ggn", "ggo", "psn", "gga", "lymphadenopathy", "recurr", "tumor"]

# Negation word lists
negation_words = ["no", "not", "without", "none"]
negation_phrases = ["no remarkable", "no significant", "no active", "no abnormal", 
                    "no abnormally", "no definite", "no definitely", "no demonstrable evidence", 
                    "no enlarged", "no evidence", "no focal", "no gross", "no indication", 
                    "no new", "no newly", "no other", "no pathologic", "no residual", 
                    "no significantly", "no visible"]

def check_keyword_with_negation(text, keywords):
    """
    Check if keywords exist without negation
    Returns 1 if keyword exists with negation, 0 otherwise
    """
    words = text.lower().split()
    word_positions = {i: word for i, word in enumerate(words)}
    
    for keyword in keywords:
        keyword_positions = []
        # Handle multi-word keywords
        if ' ' in keyword:
            keyword_words = keyword.split()
            for i in range(len(words) - len(keyword_words) + 1):
                if words[i:i+len(keyword_words)] == keyword_words:
                    keyword_positions.append(i)
        else:
            keyword_positions = [pos for pos, word in word_positions.items() if word == keyword]
        
        # Check for negation around keyword
        for keyword_pos in keyword_positions:
            left_bound_3 = max(0, keyword_pos - 3)  # Single negation words within 3 words
            left_bound_5 = max(0, keyword_pos - 5)  # Negation phrases within 5 words
            
            # Check single negation words
            single_neg_match = any(word_positions.get(i, "") in negation_words 
                                  for i in range(left_bound_3, keyword_pos))
            
            # Check negation phrases
            phrase_neg_match = any(phrase in " ".join(words[left_bound_5:keyword_pos]) 
                                 for phrase in negation_phrases)
            
            # If negation found, return 1 (indicating negative context)
            if single_neg_match or phrase_neg_match:
                return 1
    
    # Check if any keyword exists without negation
    for keyword in keywords:
        if keyword in text.lower():
            return 0  # Keyword exists without negation (positive)
    
    return 0  # No keyword found

# Apply rule-based labeling for Pneumonia
df["rule_pn"] = df["text"].apply(lambda x: 
    0 if check_keyword_with_negation(x, pneumonia_keywords) == 1 
    else (1 if any(word in x.lower() for word in pneumonia_keywords) else 0))

# Apply rule-based labeling for Pulmonary Embolism
df["rule_pe"] = df["text"].apply(lambda x: 
    0 if check_keyword_with_negation(x, embolism_keywords) == 1 
    else (1 if any(word in x.lower() for word in embolism_keywords) else 0))

# Apply rule-based labeling for Cancer (more complex rules)
# Initialize cancer label columns
for keyword in cancer_keywords:
    df[keyword] = 0

# Process cancer keywords with negation detection
for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing Cancer Keywords"):
    words = row["text"].split()
    word_positions = {i: word for i, word in enumerate(words)}
    
    for keyword in cancer_keywords:
        keyword_positions = [pos for pos, word in word_positions.items() if word == keyword]
        
        if keyword_positions:
            for keyword_pos in keyword_positions:
                left_bound_3 = max(0, keyword_pos - 3)
                left_bound_5 = max(0, keyword_pos - 5)
                
                single_neg_match = any(word_positions.get(i, "") in negation_words 
                                     for i in range(left_bound_3, keyword_pos))
                phrase_neg_match = any(phrase in " ".join(words[left_bound_5:keyword_pos]) 
                                     for phrase in negation_phrases)
                
                if single_neg_match or phrase_neg_match:
                    df.at[index, keyword] = 1

# Calculate cancer score based on keyword presence
df["keyword_sum"] = df[cancer_keywords].sum(axis=1)

# Initialize cancer label
df["label_ca"] = df["Cancer"]  # Start with diagnosis code-based label

# Apply exclusion rules for benign findings
df.loc[df["keyword_sum"] > 0, "label_ca"] = 0

# Benign nodule keywords
nodule_keywords = ["small nodule", "tiny nodule", "inflammatory nodule", 
                   "small centrilobular nodule", "benign"]
df["nodule_keywords"] = 0
df.loc[df["text"].str.contains("|".join(nodule_keywords), case=False, na=False), "nodule_keywords"] = 1
df.loc[df["nodule_keywords"] == 1, "label_ca"] = 0

# GGN/GGO keywords (often benign)
ggn_keywords = ["ggn", "ggo", "gga"]
df["ggn_keywords"] = 0
df.loc[df["text"].str.contains("|".join(ggn_keywords), case=False, na=False), "ggn_keywords"] = 1
df.loc[df["ggn_keywords"] == 1, "label_ca"] = 0

# Malignancy indicators
df["meta_malig_flag"] = 0
malignancy_terms = "meta|malig|mets|large mass|lung cancer|large tumor|large lymph nodes|large lymph node"
df.loc[(df["keyword_sum"] == 0) & 
       (df["text"].str.contains(malignancy_terms, case=False, na=False)), "meta_malig_flag"] = 1

# Check for mass growth patterns
for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Checking Mass Growth Patterns"):
    if row["keyword_sum"] == 0:
        words = row["text"].split()
        word_positions = {i: word for i, word in enumerate(words)}
        
        mass_positions = [pos for pos, word in word_positions.items() if word.startswith("mass")]
        
        for mass_pos in mass_positions:
            right_bound = min(len(words), mass_pos + 6)
            
            if any(words[i].startswith("증가") for i in range(mass_pos, right_bound)):
                df.at[index, "meta_malig_flag"] = 1
                break

# Apply malignancy flag to cancer label
df.loc[df["meta_malig_flag"] == 1, "label_ca"] = 1

# Final exclusion for clear negative cases
mask = (df['keyword_sum'] == 0) & (df['nodule_keywords'] == 0) & \
       (df['ggn_keywords'] == 0) & (df['meta_malig_flag'] == 0)
df.loc[mask, 'label_ca'] = 0

# ==========================================
# 4. Combine Rule-based and Diagnosis-based Labels
# ==========================================
# Final labels combining both approaches
df['final_pn'] = ((df['PN'] == 1) | (df['rule_pn'] == 1)).astype(int)
df['final_pe'] = ((df['PE'] == 1) | (df['rule_pe'] == 1)).astype(int)
df['final_ca'] = df['label_ca']

# Create final multi-label
df['final_labels'] = df['final_pn'].astype(str) + df['final_pe'].astype(str) + df['final_ca'].astype(str)

# ==========================================
# 5. Save Processed Data
# ==========================================
# Select relevant columns for output
output_columns = ['RPRT_DT', 'text', 'final_pn', 'final_pe', 'final_ca', 'final_labels']
df_output = df[output_columns]

# Save to CSV
df_output.to_csv('chest_ct_labeled_data.csv', index=False, encoding='utf-8-sig')
