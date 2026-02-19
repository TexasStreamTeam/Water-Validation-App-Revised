# Water_Validation_App.py
# Streamlit app for automated Water Quality Data Validation (CRP/TST-style)

import io
import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Water Quality Data Validation App",
    layout="wide"
)

st.title("Water Quality Data Validation App")

# ---------------------------------------------------------------------------
# 1. CONFIG – COLUMN NAMES (edit here if your headers differ slightly)
# ---------------------------------------------------------------------------
COLUMN_MAP = {
    "site": ["Site ID", "Site ID: Site Name", "Site ID: Site Name ", "Site"],
    "sample_date": ["Sample Date", "Date"],
    "sample_time": ["Sample Time Final Format", "Sample Time", "Time"],
    "watershed": ["Watershed", "Watershed Name"],  # optional

    # CORE
    "sample_depth": ["Sample Depth (meters)", "Sample Depth (m)"],
    "total_depth": ["Total Depth (meters)", "Total Depth (m)"],
    "secchi": ["Secchi Disk Transparency - Average", "Secchi Transparency - Average"],
    "secchi_mod": ["Secchi Disk Modifier", "Secchi Modifier"],
    "tube": ["Transparency Tube (meters)", "Transparency Tube (m)"],
    "tube_mod": ["Transparency Tube Modifier", "Transparency Tube Qualifier"],
    "do_avg": ["Dissolved Oxygen (mg/L) Average", "Dissolved Oxygen (mg/L) avg"],
    "do_1": ["Dissolved Oxygen (mg/L) 1st titration"],
    "do_2": ["Dissolved Oxygen (mg/L) 2nd titration"],
    "air_temp": ["Air Temperature (° C)", "Air Temp (° C)"],
    "water_temp": ["Water Temperature (° C)", "Water Temp (° C)"],
    "ph": ["pH (standard units)", "pH"],
    "cond": ["Conductivity (?S/cm)", "Conductivity(µS/cm)", "Conductivity (uS/cm)"],
    "tds": ["Total Dissolved Solids (mg/L)", "TDS (mg/L)"],
    "salinity": ["Salinity (ppt)"],
    "flow_severity": ["Flow Severity", "Flow severity"],
    "rain_acc": ["Rainfall Accumulation", "Total Rainfall (inches)", "Total Rainfall"],
    "days_since_rain": ["Days Since Last Significant Rainfall"],

    # QC FLAGS (optional)
    "valid_flag": ["Validation", "Valid/Invalid", "Data Quality"],

    # E. COLI
    "ecoli_avg": ["E. Coli Average", "E. coli Average"],
    "ecoli_cfu1": ["Sample 1: Colony Forming Units per 100mL"],
    "ecoli_cfu2": ["Sample 2: Colony Forming Units per 100mL"],
    "ecoli_colonies1": ["Sample 1: Colonies Counted"],
    "ecoli_colonies2": ["Sample 2: Colonies Counted"],
    "ecoli_size1": ["Sample 1: Sample Size (mL)"],
    "ecoli_size2": ["Sample 2: Sample Size (mL)"],
    "ecoli_dil1": ["Sample 1: Dilution Factor (Manual)"],
    "ecoli_dil2": ["Sample 2: Dilution Factor (Manual)"],
    "ecoli_temp": ["Sample Temp (° C)", "Incubation Temperature (°C)"],
    "ecoli_hold": ["Sample Hold Time", "Incubation Period (hours)"],
    "ecoli_blank_qc": ["Field Blank QC", "No colony growth on Field Blank"],
    "ecoli_incubation_qc": [
        "Incubation time is between 24 hours",
        "Incubation Period QC"
    ],
    "ecoli_optimal_colony": ["Optimal colony number is achieved (<200)"],

    # ADVANCED
    "orthophosphate": ["Orthophosphate", "Phosphate (mg/L)"],
    "orthophosphate_f": ["Filtered (Orthophosphate)"],
    "nitrate_n": ["Nitrate-Nitrogen VALUE (ppm or mg/L)", "Nitrate-Nitrogen (mg/L)"],
    "nitrate_f": ["Filtered (Nitrate-Nitrogen)"],
    "nitrate": ["Nitrate"],
    "turbidity": ["Turbidity Result (JTU)", "Turbidity (NTU)", "Turbidity"],
    "cross_section": ["Waterbody Cross Section"],
    "water_depth": ["Water Depth"],
    "downstream_10ft": ["10-foot Downstream Measurement"],
    "discharge": ["Discharge Recorded", "Streamflow (ft2/sec)", "Discharge (cfs)"],

    # RIPARIAN (common fields)
    "bank_evaluated": ["Bank Evaluated", "Bank evaluated is completed"],
    "riparian_image": ["Image Submitted", "Image of site was submitted"],
}

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def find_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

def categorize_columns(df):
    cols = df.columns.tolist()
    core_cols, ecoli_cols, adv_cols, riparian_cols, general_cols = [], [], [], [], []

    core_keys = [
        "sample_depth", "total_depth", "secchi", "secchi_mod", "tube", "tube_mod",
        "do_avg", "do_1", "do_2", "air_temp", "water_temp", "ph", "cond", "tds",
        "salinity", "flow_severity", "rain_acc", "days_since_rain"
    ]
    ecoli_keys = [
        "ecoli_avg", "ecoli_cfu1", "ecoli_cfu2", "ecoli_colonies1",
        "ecoli_colonies2", "ecoli_size1", "ecoli_size2", "ecoli_dil1",
        "ecoli_dil2", "ecoli_temp", "ecoli_hold", "ecoli_blank_qc",
        "ecoli_incubation_qc", "ecoli_optimal_colony"
    ]
    adv_keys = [
        "orthophosphate", "orthophosphate_f", "nitrate_n", "nitrate_f",
        "nitrate", "turbidity", "cross_section", "water_depth",
        "downstream_10ft", "discharge"
    ]
    rip_keys = ["bank_evaluated", "riparian_image"]

    used_cols = set()

    def add_cols(keys, target_list):
        for key in keys:
            c = find_col(df, COLUMN_MAP.get(key, []))
            if c:
                target_list.append(c)
                used_cols.add(c)

    add_cols(core_keys, core_cols)
    add_cols(ecoli_keys, ecoli_cols)
    add_cols(adv_keys, adv_cols)
    add_cols(rip_keys, riparian_cols)

    for c in cols:
        if c not in used_cols:
            general_cols.append(c)

    return {
        "core": core_cols,
        "ecoli": ecoli_cols,
        "advanced": adv_cols,
        "riparian": riparian_cols,
        "general": general_cols,
    }

def parse_datetime(df):
    date_col = find_col(df, COLUMN_MAP["sample_date"])
    time_col = find_col(df, COLUMN_MAP["sample_time"])
    if date_col is None:
        return df, None, None
    df["_parsed_date"] = pd.to_datetime(df[date_col], errors="coerce")
    if time_col and df[time_col].notna().any():
        def _parse_t(x):
            if pd.isna(x):
                return None
            x = str(x).strip()
            for fmt in ["%H:%M", "%H:%M:%S", "%I:%M %p"]:
                try: return datetime.strptime(x, fmt).time()
                except: continue
            return None
        df["_parsed_time"] = df[time_col].apply(_parse_t)
    else:
        df["_parsed_time"] = None
    return df, date_col, time_col

def general_cleaning(df):
    df = df.copy()
    df = df.drop_duplicates().reset_index(drop=True)
    df, _, _ = parse_datetime(df)
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].replace(
            {"valid": "", "Valid": "", "VALID": "",
             "invalid": "", "Invalid": "", "INVALID": ""}
        )
    if "_parsed_time" in df.columns and df["_parsed_time"].notna().any():
        times = df["_parsed_time"].dropna().apply(lambda t: t.hour + t.minute / 60.0)
        median_hour = times.median() if len(times) > 0 else np.nan
        df["_sample_hour"] = df["_parsed_time"].apply(
            lambda t: t.hour + t.minute / 60.0 if pd.notna(t) else np.nan
        )
        df["QC_TimeOfDay_OK"] = np.abs(df["_sample_hour"] - median_hour) <= 4
    else:
        df["QC_TimeOfDay_OK"] = np.nan

    site_col = find_col(df, COLUMN_MAP["site"])
    sort_cols = []
    if site_col: sort_cols.append(site_col)
    if "_parsed_date" in df.columns: sort_cols.append("_parsed_date")
    if "_parsed_time" in df.columns: sort_cols.append("_parsed_time")
    if sort_cols: df = df.sort_values(sort_cols).reset_index(drop=True)
    return df

# -----------------------------
# Core cleaning (depth, DO, temp, pH, cond, TDS)
# -----------------------------
def clean_core(df):
    df = df.copy()
    flow_col = find_col(df, COLUMN_MAP["flow_severity"])
    sample_depth_col = find_col(df, COLUMN_MAP["sample_depth"])
    total_depth_col = find_col(df, COLUMN_MAP["total_depth"])
    secchi_col = find_col(df, COLUMN_MAP["secchi"])
    tube_col = find_col(df, COLUMN_MAP["tube"])
    tube_mod_col = find_col(df, COLUMN_MAP["tube_mod"])
    do_avg_col = find_col(df, COLUMN_MAP["do_avg"])
    do1_col = find_col(df, COLUMN_MAP["do_1"])
    do2_col = find_col(df, COLUMN_MAP["do_2"])
    air_col = find_col(df, COLUMN_MAP["air_temp"])
    water_col = find_col(df, COLUMN_MAP["water_temp"])
    ph_col = find_col(df, COLUMN_MAP["ph"])
    cond_col = find_col(df, COLUMN_MAP["cond"])
    tds_col = find_col(df, COLUMN_MAP["tds"])

    if total_depth_col:
        depth = pd.to_numeric(df[total_depth_col], errors="coerce")
        depth = depth.mask(depth >= 998, np.nan)
        if flow_col:
            flow = df[flow_col].astype(str).str.strip().str.lower()
            mask_zero_bad = (depth == 0) & (~flow.isin(["dry", "no water", "6"]))
            depth = depth.mask(mask_zero_bad, np.nan)
        df[total_depth_col] = depth

    if sample_depth_col and total_depth_col:
        sdepth = pd.to_numeric(df[sample_depth_col], errors="coerce")
        tdepth = pd.to_numeric(df[total_depth_col], errors="coerce")
        cond_03 = np.isclose(sdepth, 0.3, atol=0.05)
        cond_half = np.isclose(sdepth, 0.5 * tdepth, atol=0.05)
        df["QC_SampleDepth_OK"] = cond_03 | cond_half
        df.loc[(sdepth.notna()) & (~df["QC_SampleDepth_OK"]), "QC_SampleDepth_OK"] = False

    # Secchi, tube, DO, temp, pH, cond, TDS
    if secchi_col and total_depth_col:
        secchi = pd.to_numeric(df[secchi_col], errors="coerce")
        tdepth = pd.to_numeric(df[total_depth_col], errors="coerce")
        secchi = secchi.mask((secchi.notna()) & (tdepth.notna()) & (secchi > tdepth), np.nan)
        df[secchi_col] = secchi.round(2)

    if tube_col:
        tube = pd.to_numeric(df[tube_col], errors="coerce")
        over_mask = tube > 1.2
        tube = tube.mask(over_mask, np.nan)
        df[tube_col] = tube.round(2)
        if tube_mod_col:
            tube_mod = df[tube_mod_col].astype(str)
            tube_mod = tube_mod.mask(tube.isna() & over_mask, ">1.2m")
            df[tube_mod_col] = tube_mod

    if do_avg_col and do1_col and do2_col:
        do1 = pd.to_numeric(df[do1_col], errors="coerce")
        do2 = pd.to_numeric(df[do2_col], errors="coerce")
        diff = (do1 - do2).abs()
        df["QC_DO_dup_within_0.5"] = diff <= 0.5
        do_avg = (do1 + do2) / 2.0
        do_avg = do_avg.mask(diff > 0.5, np.nan)
        df[do1_col] = do1.round(1)
        df[do2_col] = do2.round(1)
        df[do_avg_col] = do_avg.round(1)

    for col in [air_col, water_col]:
        if col:
            temp = pd.to_numeric(df[col], errors="coerce")
            temp = temp.mask((temp < -5) | (temp > 50), np.nan)
            df[col] = temp.round(1)

    if ph_col:
        ph = pd.to_numeric(df[ph_col], errors="coerce")
        ph = ph.mask((ph < 2) | (ph > 12), np.nan)
        df[ph_col] = ph.round(1)

    if cond_col:
        cond = pd.to_numeric(df[cond_col], errors="coerce")
        cond = cond.mask(cond < 0, np.nan)
        df[cond_col] = cond.round(0 if cond.lt(100).all() else 2)

    if cond_col and tds_col:
        cond = pd.to_numeric(df[cond_col], errors="coerce")
        tds_calc = cond * 0.65
        df["TDS_Calc (mg/L)"] = tds_calc.round(1)
        tds = pd.to_numeric(df[tds_col], errors="coerce")
        tds = tds.fillna(tds_calc)
        df[tds_col] = tds.round(1)

    return df

# -----------------------------
# E. coli cleaning
# -----------------------------
def clean_ecoli(df):
    df = df.copy()
    ecoli_avg_col = find_col(df, COLUMN_MAP["ecoli_avg"])
    cfu1_col = find_col(df, COLUMN_MAP["ecoli_cfu1"])
    cfu2_col = find_col(df, COLUMN_MAP["ecoli_cfu2"])
    col1_col = find_col(df, COLUMN_MAP["ecoli_colonies1"])
    col2_col = find_col(df, COLUMN_MAP["ecoli_colonies2"])
    temp_col = find_col(df, COLUMN_MAP["ecoli_temp"])
    hold_col = find_col(df, COLUMN_MAP["ecoli_hold"])
    blank_qc_col = find_col(df, COLUMN_MAP["ecoli_blank_qc"])
    optimal_col = find_col(df, COLUMN_MAP["ecoli_optimal_colony"])

    if ecoli_avg_col:
        ecoli_avg = pd.to_numeric(df[ecoli_avg_col], errors="coerce")
        ecoli_avg = ecoli_avg.mask(ecoli_avg == 0, np.nan).round(0)
        df[ecoli_avg_col] = ecoli_avg

    for col in [cfu1_col, cfu2_col]:
        if col:
            cfu = pd.to_numeric(df[col], errors="coerce")
            df[col] = cfu.mask(cfu == 0, np.nan)

    for col in [col1_col, col2_col]:
        if col:
            colonies = pd.to_numeric(df[col], errors="coerce")
            bad = colonies >= 200
            df.loc[bad, col] = np.nan
            if ecoli_avg_col: df.loc[bad, ecoli_avg_col] = np.nan

    if temp_col:
        temp = pd.to_numeric(df[temp_col], errors="coerce")
        df["QC_Ecoli_Temp_30_36"] = (temp >= 30) & (temp <= 36)

    if hold_col:
        hold = pd.to_numeric(df[hold_col], errors="coerce")
        df["QC_Ecoli_Hold_28_31h"] = (hold >= 28) & (hold <= 31)

    if blank_qc_col:
        blank = df[blank_qc_col].astype(str).str.strip().str.lower()
        df["QC_Ecoli_Blank_OK"] = blank.isin(["yes", "true", "ok", "no growth", "none"])

    if optimal_col:
        df["QC_Ecoli_OptimalColonyFlag"] = df[optimal_col]

    return df

# -----------------------------
# Advanced cleaning
# -----------------------------
def clean_advanced(df):
    df = df.copy()
    turb_col = find_col(df, COLUMN_MAP["turbidity"])
    discharge_col = find_col(df, COLUMN_MAP["discharge"])

    if turb_col:
        turb = pd.to_numeric(df[turb_col], errors="coerce")
        df[turb_col] = turb.mask(turb < 0, np.nan)

    if discharge_col:
        q = pd.to_numeric(df[discharge_col], errors="coerce")
        q = q.mask(q < 0, np.nan)
        df.loc[q < 10, discharge_col] = q[q < 10].round(1)
        df.loc[q >= 10, discharge_col] = q[q >= 10].round(0)

    return df

# -----------------------------
# Riparian cleaning
# -----------------------------
def clean_riparian(df):
    df = df.copy()
    bank_col = find_col(df, COLUMN_MAP["bank_evaluated"])
    img_col = find_col(df, COLUMN_MAP["riparian_image"])

    if bank_col:
        df["QC_BankEvaluated_OK"] = df[bank_col].astype(str).str.lower().isin(["yes", "true", "ok"])
    if img_col:
        df["QC_Image_OK"] = df[img_col].astype(str).str.lower().isin(["yes", "true", "ok"])
    return df

# -----------------------------
# DSR Filter Functions
# -----------------------------
def dsr_quantity_summary(df, param_cols):
    site_col = find_col(df, COLUMN_MAP["site"])
    ws_col = find_col(df, COLUMN_MAP["watershed"])
    if site_col is None:
        return {}
    site_param_counts = []
    for site, group in df.groupby(site_col):
        for param in param_cols:
            if param in group.columns:
                n_events = group[param].count()
                site_param_counts.append({"site": site, "parameter": param, "n_events": n_events})
    site_param_counts = pd.DataFrame(site_param_counts)
    if ws_col:
        ws_counts = df.groupby(ws_col)[site_col].nunique().reset_index(name="n_sites")
    else:
        ws_counts = pd.DataFrame()
    return {"site_param_counts": site_param_counts, "watershed_site_counts": ws_counts}

def filter_dsr_ready(df, category_cols, min_events=10):
    df = df.copy()
    site_col = find_col(df, COLUMN_MAP["site"])
    ws_col = find_col(df, COLUMN_MAP["watershed"])

    if site_col is None:
        return df, pd.DataFrame(), pd.DataFrame()

    summary = dsr_quantity_summary(df, category_cols)
    param_counts = summary["site_param_counts"].copy()
    if param_counts.empty:
        return df, pd.DataFrame(), pd.DataFrame()

    param_counts = param_counts.rename(columns={"n_events": "n_valid"})
    param_counts["decision"] = np.where(param_counts["n_valid"] >= min_events, "KEEP", "EXCLUDE")
    param_counts["reason"] = np.where(param_counts["decision"]=="EXCLUDE",
                                      f"≤{min_events} valid numeric values at this site", "")
    exclusion_report = param_counts.copy()

    for _, row in exclusion_report[exclusion_report["decision"]=="EXCLUDE"].iterrows():
        site_val = row[site_col]
        param = row["parameter"]
        if param in df.columns:
            df.loc[df[site_col]==site_val, param] = np.nan

    if ws_col:
        ws_counts = df.groupby(ws_col)[site_col].nunique().reset_index(name="n_sites")
        good_ws = ws_counts[ws_counts["n_sites"]>=3][ws_col]
        df = df[df[ws_col].isin(good_ws)]

    param_cols_existing = [c for c in category_cols if c in df.columns]
    if param_cols_existing:
        df = df.dropna(subset=param_cols_existing, how="all")

    wide_count_table = param_counts.pivot_table(
        index=site_col,
        columns="parameter",
        values="n_valid",
        aggfunc="first",
        fill_value=0
    ).reset_index()

    return df.reset_index(drop=True), exclusion_report, wide_count_table

# -----------------------------
# 2. UPLOAD
# -----------------------------
uploaded_file = st.file_uploader("Upload Excel or CSV", type=["xlsx", "csv"])
if uploaded_file:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
    st.success(f"Loaded {uploaded_file.name} ({df.shape[0]} rows, {df.shape[1]} columns)")
else:
    st.stop()

# -----------------------------
# 3. CLEANING
# -----------------------------
df = general_cleaning(df)
cats = categorize_columns(df)
df = clean_core(df)
df = clean_ecoli(df)
df = clean_advanced(df)
df = clean_riparian(df)

# -----------------------------
# 4. TAB 8: DSR FILTER
# -----------------------------
st.header("DSR Filtering")
all_param_cols = cats["core"] + cats["ecoli"] + cats["advanced"]

apply_dsr_filter = st.checkbox("Apply DSR filter (≥10 events per site per parameter, ≥3 sites per watershed)", value=True)

if apply_dsr_filter:
    dsr_ready_df, exclusion_report, wide_counts = filter_dsr_ready(df, all_param_cols)
    summary_filtered = dsr_quantity_summary(dsr_ready_df, all_param_cols)
else:
    dsr_ready_df = df.copy()
    summary_filtered = dsr_quantity_summary(dsr_ready_df, all_param_cols)
    exclusion_report = pd.DataFrame()
    wide_counts = pd.DataFrame()

st.markdown("**Number of sites per watershed**")
st.dataframe(summary_filtered["watershed_site_counts"])

st.markdown("**Number of valid events per parameter per site**")
st.dataframe(summary_filtered["site_param_counts"])

st.markdown("**DSR-wide count table**")
st.dataframe(wide_counts)

st.markdown("**Exclusions**")
st.dataframe(exclusion_report)

# -----------------------------
# 5. DOWNLOAD CLEANED DATA
# -----------------------------
def to_excel_download(df):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="DSR_Ready")
        writer.save()
    processed_data = output.getvalue()
    return processed_data

st.download_button(
    label="Download DSR-ready data",
    data=to_excel_download(dsr_ready_df),
    file_name=f"DSR_Ready_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
)
