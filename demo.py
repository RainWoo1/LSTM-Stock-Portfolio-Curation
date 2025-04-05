import streamlit as st
import torch
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn as nn
import os
from sklearn.preprocessing import StandardScaler

# ========== SETUP ==========
# pip install streamlit torch pandas matplotlib scipy scikit-learn
# streamlit run demo.py

# ========== MODEL ==========
class FourLayerLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_companies, embedding_dim=32, dropout=0.3):
        super(FourLayerLSTM, self).__init__()
        self.embedding = nn.Embedding(num_companies, embedding_dim)
        self.lstm1 = nn.LSTM(input_size + embedding_dim, hidden_size, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.lstm2 = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.dropout2 = nn.Dropout(dropout)
        self.lstm3 = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.dropout3 = nn.Dropout(dropout)
        self.lstm4 = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.dropout4 = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, company_ids):
        company_embed = self.embedding(company_ids)
        company_embed = company_embed.unsqueeze(1).expand(-1, x.size(1), -1)
        x = torch.cat([x, company_embed], dim=-1)
        x, _ = self.lstm1(x)
        x = self.dropout1(x)
        x, _ = self.lstm2(x)
        x = self.dropout2(x)
        x, _ = self.lstm3(x)
        x = self.dropout3(x)
        x, _ = self.lstm4(x)
        x = self.dropout4(x)
        x = x[:, -1, :]
        return self.fc(x)

# ========== LOAD THE MODEL ==========
@st.cache_resource
def load_model(model_path, device='cpu'):
    model = FourLayerLSTM(input_size=10, hidden_size=96, output_size=1, num_companies=547, embedding_dim=32, dropout=0.3)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

# ========== MODEL SETUP ==========
device = torch.device("cpu")
model_path = "demo.pth"
model = load_model(model_path, device=device)

# ========== LOAD COMPANY DATA FUNCTION ==========
def create_test_sequences_across_years(company, company_id, window_size=6):
    file_path = f"data_processing/outputs/by_ticker/{company}.csv"
    if not os.path.exists(file_path):
        print(f"Skipping {company} (No data found)")
        return None, None, None

    df = pd.read_csv(file_path)
    
    financial_cols = ["roic", "bvps", "fcf_me", "at_turnover", "ni_me"]
    time_cols = ["month_sin", "month_cos", "year_sin", "year_cos"]
    target_col = "stock_exret"
    
    df = df[financial_cols + [target_col, "year", "month"]]
    
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['year_sin'] = np.sin(2 * np.pi * (df['year'] - df['year'].min()) / (df['year'].max() - df['year'].min() + 1))
    df['year_cos'] = np.cos(2 * np.pi * (df['year'] - df['year'].min()) / (df['year'].max() - df['year'].min() + 1))
    
    df = df.sort_values(["year", "month"]).reset_index(drop=True)

    if len(df[df["year"] == 2023]) == 0:
        print(f"Skipping {company} (No 2023 data)")
        return None, None, None
    
    first_2023_idx = df[df["year"] == 2023].index[0]
    
    df = df.iloc[first_2023_idx - window_size:].reset_index(drop=True)
    
    for col in financial_cols:
        if df[col].skew() > 1:
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna(df[col].mean())
    
    scaler = StandardScaler()
    df[financial_cols] = scaler.fit_transform(df[financial_cols])
    
    sequences = []
    targets = []
    company_ids = []
    
    for i in range(window_size, len(df)):
        if df.iloc[i]["year"] != 2023:
            continue
            
        input_seq = df.iloc[i-window_size:i][financial_cols + time_cols + [target_col]].values
        target = df.iloc[i][target_col]
        
        sequences.append(input_seq)
        targets.append(target)
        company_ids.append(company_id)
    
    if len(sequences) == 0:
        return None, None, None
    
    X_tensor = torch.tensor(sequences, dtype=torch.float32)  # Shape: [n_sequences, 6, 10]
    y_tensor = torch.tensor(targets, dtype=torch.float32).unsqueeze(1)
    company_tensor = torch.tensor(company_ids, dtype=torch.long)
    
    return X_tensor, company_tensor, y_tensor


def evaluate_companies(model, company_list, company_to_id, device):
    results = {}
    
    for company in company_list:
        company_id = company_to_id[company]
        X_tensor, company_tensor, y_tensor = create_test_sequences_across_years(company, company_id)
        
        if X_tensor is None:
            print(f"Skipping {company} (Not enough data)")
            continue
            
        with torch.no_grad():
            predictions = model(X_tensor.to(device), company_tensor.to(device)).cpu().squeeze().numpy()
        
            
        actuals = y_tensor.squeeze().numpy()
        
        try:
            corr = np.corrcoef(actuals, predictions)[0, 1] if len(actuals) > 1 else float('nan')
        except Exception as e:
            pass
        
        st.write(f"📊 **{company} Correlation:** `{corr:.4f}`")
        
        results[company] = {
            "actual": actuals,
            "predicted": predictions,
            "correlation": corr
        }

    return results


# ========== STREAMLIT ==========
all_companies = []
st.title("📈 Stock Excess Return Prediction")

st.sidebar.header("Add Company Files")
uploaded_files = st.sidebar.file_uploader("Upload CSV files in the format of [TICKER].csv", type="csv", accept_multiple_files=True)

if uploaded_files:
    for uploaded_file in uploaded_files:
        company_name = uploaded_file.name.split('.')[0]
        df = pd.read_csv(uploaded_file)
        all_companies.append(company_name)

if len(all_companies) == 0:
    st.warning("No companies available. Please upload CSV files.")
    st.stop()

selected_companies = st.text_input("Enter Company Ticker (comma-separated) MAX 5")
selected_companies = selected_companies.upper().replace(' ', '').split(",")
# selected_companies = all_companies

if selected_companies == ['']:
    st.warning("Please select at least one company.")
    st.stop()
elif len(selected_companies) > 5:
    st.warning("Please select a maximum of 5 companies.")
    st.stop()

for company in selected_companies:
    if company not in all_companies:
        st.warning(f"{company} is not in the uploaded files.")
        st.stop()

num_companies_to_plot = len(selected_companies)
company_to_id = {company: i for i, company in enumerate(selected_companies)}

company_results = evaluate_companies(model, selected_companies, company_to_id, device)

# ========== PLOTTING ==========
st.subheader("📊 Actual vs Predicted Excess Return")
fig, axes = plt.subplots(1, num_companies_to_plot, figsize=(15, 5))

if num_companies_to_plot == 1:
    axes = [axes]

for i, (company, results) in enumerate(company_results.items()):
    ax = axes[i]
    ax.plot(results["actual"], label="Actual", marker="o", linestyle="-", color="blue")
    ax.plot(results["predicted"], label="Predicted", marker="x", linestyle="--", color="red")
    ax.set_xlabel("Time (Months)")
    ax.set_ylabel("Excess Return")
    ax.set_title(f"{company} - Actual vs. Predicted")
    ax.legend()
    ax.grid(True)

st.pyplot(fig)
