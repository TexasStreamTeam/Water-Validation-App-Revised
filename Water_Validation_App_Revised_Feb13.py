def filter_dsr_ready(df, category_cols=None, min_events=10):
    """
    DSR logic:
      1. Remove watersheds with ≤ 3 sites.
      2. For remaining sites, remove parameters with ≤ min_events values per site.
      3. Drop rows where all parameters are NaN.
      4. Return filtered df + exclusion report + wide count table.
    """

    df = df.copy()
    site_col = find_col(df, COLUMN_MAP["site"])
    watershed_col = find_col(df, COLUMN_MAP["watershed"])

    if not site_col:
        return df, pd.DataFrame(), pd.DataFrame()

    exclusion_records = []

    # --------------------------------------------------
    # RULE 1: Remove watersheds with ≤ 3 sites
    # --------------------------------------------------
    if watershed_col:
        site_counts = df.groupby(watershed_col)[site_col].nunique()
        bad_watersheds = site_counts[site_counts <= 3].index.tolist()

        for ws in bad_watersheds:
            exclusion_records.append({
                "Watershed": ws,
                "Site": "",
                "Parameter": "",
                "n_values": "",
                "decision": "EXCLUDE (≤3 sites in watershed)"
            })

        df = df[~df[watershed_col].isin(bad_watersheds)]

    # --------------------------------------------------
    # RULE 2: Remove parameters with ≤ min_events values per site
    # --------------------------------------------------
    if not category_cols:
        category_cols = []
    
    param_cols = [c for c in category_cols if c in df.columns]

    df["_site_norm"] = df[site_col].astype(str).str.strip()

    # Build a mapping of (site, parameter) pairs that fail the threshold
    params_to_exclude = set()

    for col in param_cols:
        counts = df.groupby("_site_norm")[col].apply(lambda x: x.notna().sum())
        bad_sites = counts[counts <= min_events].index.tolist()

        for site in bad_sites:
            params_to_exclude.add((site, col))
            watershed_value = ""
            if watershed_col:
                ws_data = df.loc[df["_site_norm"] == site, watershed_col]
                if len(ws_data) > 0:
                    watershed_value = ws_data.iloc[0]
            
            exclusion_records.append({
                "Watershed": watershed_value,
                "Site": site,
                "Parameter": col,
                "n_values": int(counts[site]),
                "decision": f"EXCLUDE (≤{min_events} values)"
            })

    # Apply exclusions: set matching (site, parameter) pairs to NaN
    for (site, col) in params_to_exclude:
        df.loc[df["_site_norm"] == site, col] = np.nan

    # --------------------------------------------------
    # RULE 3: Drop rows where ALL parameters are NaN
    # --------------------------------------------------
    if param_cols:
        df = df.dropna(subset=param_cols, how="all")

    df = df.drop(columns=["_site_norm"], errors="ignore")

    # --------------------------------------------------
    # Build outputs
    # --------------------------------------------------
    exclusion_report = pd.DataFrame(exclusion_records)

    # Wide table: site × parameter counts
    if param_cols:
        wide_counts = (
            df.groupby(site_col)[param_cols]
            .apply(lambda x: x.notna().sum())
            .reset_index()
        )
    else:
        wide_counts = pd.DataFrame()

    return df.reset_index(drop=True), exclusion_report, wide_counts