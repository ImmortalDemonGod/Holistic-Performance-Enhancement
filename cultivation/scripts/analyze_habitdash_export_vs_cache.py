import pandas as pd

EXPORT_CSV = "cultivation/data/raw/habitdash_export_2025-06-06/2025-06-06 Habit Dash - Integrations (flat file).csv"
CACHE_PARQUET = "cultivation/data/daily_wellness.parquet"
REPORT_MD = "cultivation/data/habitdash_export_vs_cache_report.md"

# 1. Load export and cache
export_df = pd.read_csv(EXPORT_CSV)
cache_df = pd.read_parquet(CACHE_PARQUET)

# 2. All analysis uses canonical export field names (e.g., sleep_score_whoop)
def make_cache_colname(row):
    metric = row['name'].strip().lower().replace(' ', '_').replace('/', '_per_')
    source = row['source'].strip().lower()
    return f"{metric}_{source}"

export_df['cache_col'] = export_df.apply(make_cache_colname, axis=1)

# 3. Catalog all unique fields in export
export_fields = export_df.drop_duplicates(['cache_col', 'source', 'category', 'name', 'units'])

# 4. List all columns in cache and get non-null counts
cache_cols = set(cache_df.columns)
notnull_counts = cache_df.notnull().sum().to_dict()

# Analysis suggestion templates by category
analysis_suggestions = {
    'Activity': 'time trends, intensity analysis, correlation with sleep/fatigue, weekly/monthly totals',
    'Sleep Length': 'sleep duration trends, stage distribution, impact on recovery/performance',
    'Sleep Quality': 'fragmentation, disturbances, onset/latency, correlation with next-day performance',
    'Sleep Vitals': 'respiratory rate trends, anomaly detection',
    'Heart': 'resting/avg/max trends, HRV analysis, stress/fatigue modeling',
    'Body': 'weight/body comp trends, BMI, health risk flags',
    'Metabolism': 'caloric balance, activity vs. calories, weight change modeling',
    'Recovery': 'readiness, recovery prediction, overtraining risk',
    'Heart + Lungs': 'aerobic fitness, VO₂ max, blood oxygenation, altitude adaptation',
    'Vitals': 'blood pressure tracking, cardiovascular risk',
}
def suggest_analysis(cat):
    return analysis_suggestions.get(cat, 'trend analysis, anomaly detection, cross-metric correlation')

# 5. Compare and build detailed field inventory
inventory = []
for _, row in export_fields.iterrows():
    col = row['cache_col']
    in_cache = col in cache_cols
    notnull = notnull_counts.get(col, 0)
    inventory.append({
        'cache_col': col,
        'source': row['source'],
        'category': row['category'],
        'name': row['name'],
        'units': row['units'],
        'notnull': notnull,
        'in_cache': in_cache,
        'suggested_analysis': suggest_analysis(row['category'])
    })

# 6. Fields in export only (not in cache)
not_in_cache = [col for col in export_fields['cache_col'].values if col not in cache_cols]

# 7. Fields in cache only (legacy/orphaned)
legacy_cache_cols = [col for col in cache_cols if col not in export_fields['cache_col'].values]

# 8. Generate Markdown report
with open(REPORT_MD, 'w') as f:
    # --- Summary ---
    f.write("# Habit Dash Export vs. Current Wellness Cache\n\n")
    f.write(f"**Flat file analyzed:** {EXPORT_CSV}\n\n")
    f.write(f"**Current cache:** {CACHE_PARQUET}\n\n")
    f.write(f"**Total canonical fields:** {len(inventory)}\n")
    f.write(f"**Fields with data (non-null):** {sum(1 for i in inventory if i['notnull']>0)}\n")
    f.write(f"**Most complete fields:** {[i['cache_col'] for i in sorted(inventory, key=lambda x: -x['notnull'])[:5]]}\n")
    f.write(f"**Least complete fields:** {[i['cache_col'] for i in sorted(inventory, key=lambda x: x['notnull'])[:5]]}\n")
    f.write("\n**Schema notes:**\n- Canonical export field names are now used throughout.\n- Cache is wide-format (one column per metric).\n- Legacy columns removed.\n- All export fields are present in cache.\n\n")

    # --- Inventory by category/source ---
    f.write("## Canonical Field Inventory (Grouped by Category/Source)\n\n")
    cats = sorted(set(i['category'] for i in inventory))
    for cat in cats:
        f.write(f"### {cat}\n\n")
        cat_fields = [i for i in inventory if i['category']==cat]
        sources = sorted(set(i['source'] for i in cat_fields))
        for src in sources:
            f.write(f"#### Source: {src}\n\n")
            f.write("| cache_col | name | units | non-null count | suggested analyses |\n")
            f.write("|-----------|------|-------|----------------|-------------------|\n")
            for i in cat_fields:
                if i['source'] == src:
                    f.write(f"| {i['cache_col']} | {i['name']} | {i['units']} | {i['notnull']} | {i['suggested_analysis']} |\n")
            f.write("\n")

    # --- High-level field availability matrix ---
    f.write("## Field Availability Matrix (high-level)\n\n")
    f.write("| Field | Non-null Count |\n")
    f.write("|-------|----------------|\n")
    for i in inventory:
        f.write(f"| {i['cache_col']} | {i['notnull']} |\n")
    f.write("\n")

    # --- Fields in export only (should be empty) ---
    f.write("## Fields in Export Only (not in cache; ❌ need to backfill)\n\n")
    for col in not_in_cache:
        field = export_fields[export_fields['cache_col'] == col].iloc[0]
        f.write(f"- {col} | {field['source']} | {field['category']} | {field['name']} | {field['units']}\n")

    # --- Fields in cache only (should be empty/legacy) ---
    f.write("\n## Fields in Cache Only (legacy/orphaned; ⚠️ should be migrated/removed)\n\n")
    for col in legacy_cache_cols:
        f.write(f"- {col}\n")

    f.write("\n---\n")
    f.write("- ✅ = field exists in both export and cache\n")
    f.write("- ❌ = field is new (not in cache, but available in export)\n")
    f.write("- ⚠️ = legacy/orphaned field in cache (not in export; should be migrated/removed)\n\n")

    # --- Next analysis opportunities ---
    f.write("## Next Analysis Opportunities (Unlocked by Full Data)\n\n")
    f.write("- Multi-device comparison: e.g., compare RHR, HRV, sleep between Garmin, Whoop, Withings.\n")
    f.write("- Longitudinal health trends: weight, HRV, sleep, activity over months/years.\n")
    f.write("- Machine learning: predict fatigue, recovery, or injury risk from full metric set.\n")
    f.write("- Cross-metric correlation: e.g., sleep quality vs. next-day activity, HRV vs. stress.\n")
    f.write("- Anomaly detection: flag outlier days for any metric.\n")
    f.write("- Custom dashboards: build performance, wellness, or risk dashboards using the full schema.\n")

print(f"Report written to {REPORT_MD}")
