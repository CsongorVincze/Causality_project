import eurostat
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

print("--- Extracting & Plotting Hungary Arrays (1990-2020) ---")

def fetch_data(dataset_code):
    cache_file = f"{dataset_code}.csv"
    if os.path.exists(cache_file):
        return pd.read_csv(cache_file, low_memory=False)
    else:
        return eurostat.get_data_df(dataset_code)

def get_geo_col(df):
    candidates = [c for c in df.columns if 'geo' in c.lower()]
    return candidates[0] if candidates else None

def parse_dates_robust(index_obj):
    s = index_obj.astype(str).str.upper().str.strip()
    s = s.str.replace('Q1', '01', regex=False).str.replace('Q2', '04', regex=False).str.replace('Q3', '07', regex=False).str.replace('Q4', '10', regex=False)
    s = s.str.replace('M', '', regex=False).str.replace('-', '', regex=False).str.replace('_', '', regex=False).str.replace(' ', '', regex=False)
    return pd.to_datetime(s, format='%Y%m', errors='coerce')

def extract_series(dataset_code, geo_filter, unit_list, cat_col, cat_list, start_year, end_year):
    df = fetch_data(dataset_code)
    geo_col = get_geo_col(df)
    
    # Filters
    df = df[df[geo_col] == geo_filter]
    
    if 'unit' in df.columns:
        found = False
        for u in unit_list:
            if u in df['unit'].values:
                df = df[df['unit'] == u]
                found = True
                break
                
    if cat_col and cat_col in df.columns:
        found = False
        for c in cat_list:
            if c in df[cat_col].values:
                df = df[df[cat_col] == c]
                found = True
                break
    
    if df.empty: return None, "Empty after filters"

    # Time Cols
    time_cols = [col for col in df.columns if len(col) > 0 and col[0].isdigit()]
    if not time_cols: return None, "No time columns"
        
    df_series = df[time_cols].T
    df_series.index = parse_dates_robust(df_series.index)
    
    # Filter Year
    df_series = df_series[(df_series.index.year >= start_year) & (df_series.index.year <= end_year)]
    
    if df_series.empty: return None, "Empty after date filter"

    # Data
    s_data = df_series.iloc[:, 0]
    s_data = pd.to_numeric(s_data, errors='coerce').dropna()
    
    return s_data, "Success"

# Extract Series (keeping index for plotting)
hicp_series, msg_hicp = extract_series('prc_hicp_midx', 'HU', ['I15'], 'coicop', ['CP00'], 1990, 2020)
lci_series, msg_lci = extract_series('lc_lci_r2_q', 'HU', ['I15', 'I20'], 'nace_r2', ['B-S'], 1990, 2020)

# Get Arrays (Initial)
hicp_vals_initial = hicp_series.values if hicp_series is not None else None
lci_vals_initial = lci_series.values if lci_series is not None else None

# ALIGNMENT LOGIC
if hicp_series is not None and lci_series is not None:
    # Find common date range
    start_date = max(hicp_series.index.min(), lci_series.index.min())
    end_date = min(hicp_series.index.max(), lci_series.index.max())
    
    print(f"\n--- Aligning Series ---")
    print(f"Common Start Date: {start_date.date()}")
    print(f"Common End Date:   {end_date.date()}")
    
    # Slice both to this range
    hicp_series = hicp_series[(hicp_series.index >= start_date) & (hicp_series.index <= end_date)]
    lci_series = lci_series[(lci_series.index >= start_date) & (lci_series.index <= end_date)]
    
    # RESAMPLING: The user requested to "drop values between a quarter" rather than averaging.
    # Since LCI indices are Jan 1, Apr 1... (checking our parsing), and HICP has these + Feb, Mar...
    # We can simple intersect indices. This selects the Month corresponding to the Quarter start.
    print(f"Selecting HICP values matching LCI quarters (dropping intermediate months)...")
    # hicp_series = hicp_series.resample('QS').mean() # OLD METHOD
    
    # Ensure they are exactly the same size/index intersection again after resampling
    common_idx = hicp_series.index.intersection(lci_series.index)
    hicp_series = hicp_series.loc[common_idx]
    lci_series = lci_series.loc[common_idx]
    
    hicp_vals = hicp_series.values
    lci_vals = lci_series.values
else:
    hicp_vals = hicp_vals_initial
    lci_vals = lci_vals_initial

print("\n--- RESULTS (Aligned & Subset) ---")
if hicp_vals is not None:
    print(f"HICP Array Shape: {hicp_vals.shape}")
    print(f"First 10: {hicp_vals[:10]}")

if lci_vals is not None:
    print(f"LCI Array Shape: {lci_vals.shape}")
    print(f"First 10: {lci_vals[:10]}")

# Plotting
print("\n--- PLOTTING ---")
if hicp_series is not None and lci_series is not None:
    plt.figure(figsize=(12, 6))
    
    # Plot using the series index (dates) so they align correctly
    plt.plot(hicp_series.index, hicp_vals, label='HICP (First Month of Q)', color='blue', marker='x', linestyle='-')
    plt.plot(lci_series.index, lci_vals, label='LCI (Quarterly)', color='red', marker='o', linestyle='--')
    
    plt.title(f"Hungary HICP vs LCI ({start_date.year}-{end_date.year}) - Quarterly Aligned (Subset)")
    plt.xlabel("Date")
    plt.ylabel("Index")
    plt.grid(True)
    plt.legend()
    
    out_file = "hungary_aligned_plot.png"
    plt.savefig(out_file)
    print(f"Plot saved to: {out_file}")
else:
    print("Cannot plot: one or both series missing.")



print(lci_series.shape)