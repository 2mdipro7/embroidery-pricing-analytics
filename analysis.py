import os, sys, re, json, warnings, textwrap, hashlib, time
sys.stdout.reconfigure(encoding='utf-8')
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
import seaborn as sns
from scipy import stats
from scipy.stats import chi2_contingency, mannwhitneyu, spearmanr, kruskal, f_oneway
import statsmodels.api as sm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.miscmodels.ordinal_model import OrderedModel
from openai import AzureOpenAI

try:
    import squarify
    HAS_SQUARIFY = True
except ImportError:
    HAS_SQUARIFY = False
try:
    import scikit_posthocs as sp
    HAS_POSTHOCS = True
except ImportError:
    HAS_POSTHOCS = False

# CONFIGURATION
warnings.filterwarnings('ignore')
sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams.update({
    'figure.dpi': 200,
    'savefig.dpi': 200,
    'savefig.bbox': 'tight',
    'font.family': 'sans-serif'
})

PALETTE = {
    'Pricing Blind Spots': '#e74c3c',
    'Partial Pricing System': '#f39c12',
    'Strong Pricing System': '#2ecc71'
}

AZURE_OPENAI_DEPLOYMENT = os.environ.get('AZURE_OPENAI_DEPLOYMENT', 'mehrabuse')
AZURE_OPENAI_API_VERSION = os.environ.get('AZURE_OPENAI_API_VERSION', '2024-12-01-preview')
AZURE_OPENAI_ENDPOINT = os.environ.get('AZURE_OPENAI_ENDPOINT', '')
AZURE_OPENAI_API_KEY = os.environ.get('AZURE_OPENAI_API_KEY', '')

BASE_DIR = Path('.')
RAW_DIAG_PATH = BASE_DIR / 'nnep_uncleaned_survey_simulation_package/raw/pricing_diagnostic_raw_export.csv'
RAW_BENCH_PATH = BASE_DIR / 'nnep_uncleaned_survey_simulation_package/raw/benchmark_survey_raw_export.csv'
OUT_DIR = BASE_DIR / 'outputs'
CLEAN_DIR = BASE_DIR / 'cleaned'

OUT_DIR.mkdir(exist_ok=True)
CLEAN_DIR.mkdir(exist_ok=True)

print("="*60)
print("PHASE 1: DATA LOADING & PROFILING")
print("="*60)

df_diag_raw = pd.read_csv(RAW_DIAG_PATH)
df_bench_raw = pd.read_csv(RAW_BENCH_PATH)

print(f"Diagnostic Raw Shape: {df_diag_raw.shape}")
print(f"Benchmark Raw Shape: {df_bench_raw.shape}")

print("\nDiagnostic Data Types:")
print(df_diag_raw.dtypes.value_counts())
print(f"Diagnostic Nulls: {df_diag_raw.isnull().sum().sum()}")

print("\nBenchmark Data Types:")
print(df_bench_raw.dtypes.value_counts())
print(f"Benchmark Nulls: {df_bench_raw.isnull().sum().sum()}")

print("\nDiagnostic simulated issues:")
print(df_diag_raw['_simulated_raw_issue'].value_counts())

print("\nBenchmark simulated issues:")
print(df_bench_raw['_simulated_raw_issue'].value_counts())

# Visual 1
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
issue_colors = {'clean': '#2ecc71', 'duplicate_complete': '#e74c3c', 'duplicate_partial': '#e74c3c', 'partial': '#f39c12', 'test': '#9b59b6'}
for ax, df, title in zip(axes, [df_diag_raw, df_bench_raw], ['Diagnostic Raw Issues', 'Benchmark Raw Issues']):
    counts = df['_simulated_raw_issue'].value_counts()
    bars = ax.bar(counts.index.astype(str), counts.values, color=[issue_colors.get(x, '#34495e') for x in counts.index])
    ax.set_title(title, fontweight='bold')
    ax.tick_params(axis='x', rotation=45)
    for bar in bars:
        ax.annotate(str(bar.get_height()), (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    ha='center', va='bottom')
plt.tight_layout()
plt.savefig(OUT_DIR / '01_data_quality_profile.png')
plt.close()
print("✅ Saved 01_data_quality_profile.png")

print("\n" + "="*60)
print("PHASE 2: DATA CLEANING & ENGINEERING")
print("="*60)

def clean_email(email_str):
    if pd.isna(email_str): return ""
    e = str(email_str).strip().lower()
    e = re.sub(r'\s+', '', e)
    e = re.sub(r'\+[^@]*@', '@', e)
    return e

def parse_years(text):
    if pd.isna(text): return np.nan
    text = str(text).lower()
    if 'since' in text:
        match = re.search(r'since\s+(\d{4})', text)
        if match: return max(0, 2026 - int(match.group(1)))
    if 'new' in text or 'less than 1' in text: return 0
    match = re.search(r'(\d+)', text)
    if match: return int(match.group(1))
    return np.nan

word_to_num = {'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6, 'single': 1}
def parse_machine_count(text):
    if pd.isna(text): return np.nan
    text = str(text).lower()
    for w, n in word_to_num.items():
        if w in text: return n
    match = re.search(r'(\d+)', text)
    if match: return int(match.group(1))
    return np.nan

def normalize_business_size(text):
    if pd.isna(text): return 'Unknown'
    t = str(text).lower()
    if any(x in t for x in ['solo', 'just me', 'one', 'owner']): return 'Solo owner/operator'
    if any(x in t for x in ['2', '3', '4', '5', 'small team']): return '2-5 employees'
    if any(x in t for x in ['6', '7', '8', '9', '10', 'mid']): return '6-10 employees'
    if any(x in t for x in ['11', '25', 'larger shop']): return '11-25 employees'
    if any(x in t for x in ['26', '30', 'large operation']): return '26+ employees'
    return 'Unknown'

def normalize_pricing_method(text):
    if pd.isna(text): return 'Unknown'
    t = str(text).lower()
    if any(x in t for x in ['by stitch', 'stitch rate', 'stitch count', '1000', '1k']): return 'Stitch-count based'
    if any(x in t for x in ['cost based', 'cost+margin', 'cost plus', 'costing worksheet', 'materials', 'calculator']): return 'Cost-plus'
    if any(x in t for x in ['hybrid', 'job costing', 'spreadsheet']): return 'Hybrid/job costing'
    if any(x in t for x in ['value based', 'premium pricing', 'worth']): return 'Value-based'
    if any(x in t for x in ['market rate', 'competitor', 'competitive', 'match local']): return 'Competitor-based'
    if any(x in t for x in ['gut feel', 'quote each', 'case by case', 'custom', 'depends']): return 'Informal/custom quote'
    return 'Unknown'

def normalize_revenue(text):
    if pd.isna(text): return 'Unknown'
    t = str(text).lower()
    if 'prefer' in t or 'n/a' in t: return 'Prefer not to say'
    if any(x in t for x in ['under 50', 'under $50']): return 'Under $50k'
    if any(x in t for x in ['50k-100k', '50k - 100k', '$50k-$100k']): return '$50k-$100k'
    if any(x in t for x in ['100k', '250k']): return '$100k-$250k'
    if any(x in t for x in ['250k', '500k']): return '$250k-$500k'
    if any(x in t for x in ['500k', '1m']): return '$500k-$1M'
    if any(x in t for x in ['1m+', '1 m', '$1m']): return '$1M+'
    return 'Unknown'

def normalize_review_freq(text):
    if pd.isna(text): return 'Unknown'
    t = str(text).lower()
    if any(x in t for x in ['every quote', 'each estimate']): return 'Before every quote'
    if any(x in t for x in ['quarter', '4x', 'every few months']): return 'Quarterly'
    if any(x in t for x in ['year', 'annual']): return 'Annually'
    if any(x in t for x in ['when costs change', 'supplies go up', 'vendors raise', 'as needed', 'constantly']): return 'When costs change'
    if any(x in t for x in ['not often', 'rarely', 'when i remember']): return 'Rarely/never'
    return 'Unknown'

def extract_numeric(text):
    if pd.isna(text): return np.nan
    t = str(text).lower().replace(',', '')
    match = re.search(r'(\d+\.?\d*)', t)
    if match: return float(match.group(1))
    return np.nan

def clean_diag(df_raw):
    df = df_raw.copy()
    steps = [('Raw Data', len(df))]
    
    df['Finished_bool'] = df['Finished'].astype(str).str.lower().isin(['true', 'y', 'complete', '1', '100'])
    df = df[df['Finished_bool']]
    steps.append(('Finished=True', len(df)))
    
    df = df[pd.to_numeric(df['Progress'], errors='coerce') == 100]
    steps.append(('Progress=100', len(df)))
    
    df = df[df['Status'] != 'Survey Preview']
    df = df[df['Distribution Channel'] != 'preview']
    steps.append(('Remove Test Rows', len(df)))
    
    df['canonical_email'] = df['Email'].apply(clean_email)
    df = df[df['canonical_email'] != ""]
    df['End Date'] = pd.to_datetime(df['End Date'], errors='coerce', utc=True)
    df = df.sort_values('End Date').drop_duplicates('canonical_email', keep='last')
    steps.append(('Deduplicate Emails', len(df)))
    
    df['business_name_clean'] = df['Business Name'].astype(str).str.strip().str.title()
    df['region'] = df['Region'].fillna('Unknown')
    df['business_size'] = df['Business Size'].apply(normalize_business_size)
    df['annual_revenue_band'] = df['Annual Revenue Band'].apply(normalize_revenue)
    df['years_in_operation'] = df['Years in Business'].apply(parse_years)
    
    def years_band(y):
        if pd.isna(y): return 'Unknown'
        if y < 2: return '<2'
        if y <= 5: return '2-5'
        if y <= 10: return '6-10'
        if y <= 20: return '11-20'
        return '21+'
    df['years_in_operation_band'] = df['years_in_operation'].apply(years_band)
    df['machine_count'] = df['Machine Count'].apply(parse_machine_count)
    df['primary_business_model'] = df['Primary Business Model'].fillna('Unknown')
    df['customer_mix'] = df['Customer Mix'].fillna('Unknown')
    df['current_pricing_method'] = df['Current Pricing Method'].apply(normalize_pricing_method)
    df['price_review_frequency'] = df['Price Review Frequency'].apply(normalize_review_freq)
    
    score_0_labels = ['No', 'Not really', 'I usually guess', 'No consistent process', "I don't track this", 'Not yet']
    score_1_labels = ['Sometimes', 'For larger jobs only', 'Partially', 'Rough estimate', 'Depends on the job', 'I have a basic spreadsheet']
    score_2_labels = ['Yes', 'Always', 'Yes, documented', 'Built into our quote process', 'We review and adjust', 'We use a calculator/spreadsheet']
    
    def score_q(ans):
        if pd.isna(ans): return 1
        ans = str(ans).strip()
        if ans in score_0_labels: return 0
        if ans in score_2_labels: return 2
        return 1
        
    q_cols = [c for c in df.columns if re.match(r'^Q\d+:', c)]
    # In case column names don't exactly match Q1: ..., use the first 10 matching ones
    if not q_cols:
        q_cols = [c for c in df.columns if c.startswith('Q')]
    for i, c in enumerate(q_cols[:10]):
        df[f'q{i+1}_score'] = df[c].apply(score_q)
        
    df['production_cost_score'] = df.get('q1_score', 0) + df.get('q2_score', 0)
    df['real_labor_score'] = df.get('q3_score', 0) + df.get('q4_score', 0)
    df['intended_profit_score'] = df.get('q5_score', 0) + df.get('q6_score', 0)
    df['capacity_pressure_score'] = df.get('q7_score', 0) + df.get('q8_score', 0)
    df['end_customer_value_score'] = df.get('q9_score', 0) + df.get('q10_score', 0)
    
    score_cols = [f'q{i}_score' for i in range(1, 11) if f'q{i}_score' in df.columns]
    df['diagnostic_total_score'] = df[score_cols].sum(axis=1)
    
    def get_tier(s):
        if s <= 7: return 'Pricing Blind Spots'
        if s <= 14: return 'Partial Pricing System'
        return 'Strong Pricing System'
    df['pricing_maturity_tier'] = df['diagnostic_total_score'].apply(get_tier)
    
    df['has_structured_pricing_system'] = df['diagnostic_total_score'] >= 15
    df['labor_underpriced_flag'] = df['real_labor_score'] <= 1
    df['profit_target_missing_flag'] = df['intended_profit_score'] <= 1
    df['capacity_pricing_gap_flag'] = df['capacity_pressure_score'] <= 1
    
    keep_cols = ['canonical_email', 'business_name_clean', 'region', 'business_size', 'annual_revenue_band', 
                 'years_in_operation', 'years_in_operation_band', 'machine_count', 'primary_business_model', 
                 'customer_mix', 'current_pricing_method', 'price_review_frequency', 
                 'production_cost_score', 'real_labor_score', 'intended_profit_score', 'capacity_pressure_score', 'end_customer_value_score', 
                 'diagnostic_total_score', 'pricing_maturity_tier', 'has_structured_pricing_system', 'labor_underpriced_flag', 
                 'profit_target_missing_flag', 'capacity_pricing_gap_flag', 'Open Comment', '_simulated_raw_issue', 'Response ID'] + score_cols
    missing = [c for c in keep_cols if c not in df.columns]
    for c in missing: df[c] = np.nan
    return df[keep_cols].copy(), steps

df_diag_clean, diag_funnel = clean_diag(df_diag_raw)

audit_records = []
for i, r in df_diag_raw.iterrows():
    if r['Response ID'] not in df_diag_clean['Response ID'].values:
        audit_records.append({
            'audit_id': f"AUD-D-{i}",
            'dataset': 'Pricing Diagnostic',
            'raw_response_id': r['Response ID'],
            'issue_type': str(r.get('_simulated_raw_issue', 'unknown')),
            'cleaning_action': 'Excluded',
            'rationale': 'Filtered during cleaning'
        })

def clean_bench(df_raw):
    df = df_raw.copy()
    
    df['Finished_bool'] = df['Finished'].astype(str).str.lower().isin(['true', 'y', 'complete', '1', '100'])
    df = df[df['Finished_bool']]
    df = df[pd.to_numeric(df['Progress'], errors='coerce') == 100]
    if 'Distribution Channel' in df.columns:
        df = df[df['Distribution Channel'] != 'preview']
    if 'Status' in df.columns:
        df = df[df['Status'] != 'Survey Preview']
    
    df['canonical_email'] = df['Email Address'].apply(clean_email)
    df = df[df['canonical_email'] != ""]
    df['Submit Timestamp'] = pd.to_datetime(df['Submit Timestamp'], errors='coerce', utc=True)
    df = df.sort_values('Submit Timestamp').drop_duplicates('canonical_email', keep='last')
    
    df['business_name_clean'] = df['Business Name'].astype(str).str.strip().str.title()
    df['region'] = df['Region'].fillna('Unknown')
    df['business_size'] = df['Business Size'].apply(normalize_business_size)
    df['annual_revenue_band'] = df['Annual Revenue Band'].apply(normalize_revenue)
    df['years_in_operation'] = df['Years in Business'].apply(parse_years)
    df['machine_count'] = df['Machine Count'].apply(parse_machine_count)
    df['primary_business_model'] = df['Primary Business Model'].fillna('Unknown')
    df['primary_pricing_method'] = df['How do you price most embroidery jobs?'].apply(normalize_pricing_method)
    
    df['hourly_rate_estimate_usd'] = df['Hourly rate used, if any'].apply(extract_numeric)
    df['stitch_rate_per_1000'] = df['Price per 1,000 stitches, if used'].apply(extract_numeric)
    
    def parse_min(text):
        if pd.isna(text): return np.nan, np.nan
        t = str(text).lower()
        q, c = np.nan, np.nan
        if 'piece' in t or re.search(r'^\d+$', t):
            q = extract_numeric(t)
        if '$' in t or 'charge' in t:
            c = extract_numeric(t)
        return q, c
    
    mins = df['Minimum order or minimum charge'].apply(parse_min).apply(pd.Series)
    df['minimum_order_quantity'] = mins[0]
    df['minimum_job_charge_usd'] = mins[1]
    
    def parse_setup(text):
        if pd.isna(text): return False, 'No'
        t = str(text).lower()
        if 'no' in t and 'setup' in t: return False, 'No'
        if 'no' in t: return False, 'No'
        if any(x in t for x in ['rarely', 'depends', 'sometimes']): return True, 'Sometimes/depends'
        return True, 'Always/standard'
    
    setups = df['Do you charge a setup fee?'].apply(parse_setup).apply(pd.Series)
    df['charges_setup_fee'] = setups[0]
    df['setup_fee_policy'] = setups[1]
    
    def parse_rush(text):
        if pd.isna(text): return False, 'No'
        t = str(text).lower()
        if 'no' in t and 'rush' in t: return False, 'No'
        if 'no' in t: return False, 'No'
        if any(x in t for x in ['only for very fast', 'not usually', 'case by case', "we should but don't"]): return False, 'Case-by-case'
        return True, 'Published percentage/flat fee'
        
    rushes = df['Do you charge a rush fee?'].apply(parse_rush).apply(pd.Series)
    df['charges_rush_fee'] = rushes[0]
    df['rush_fee_policy'] = rushes[1]
    
    def norm_discount(text):
        if pd.isna(text): return 'Unknown'
        t = str(text).lower()
        if any(x in t for x in ['rarely', 'not often', 'occasionally', 'hold price', 'strategic']): return 'Rarely'
        if 'sometimes' in t or 'depends' in t or 'schools' in t: return 'Sometimes'
        if 'frequently' in t or 'often' in t or 'a lot' in t or 'more than i should' in t: return 'Frequently'
        if 'no ' in t or 'never' in t: return 'Never'
        return 'Unknown'
    df['discounting_frequency'] = df['How often do you discount quoted prices?'].apply(norm_discount)
    
    df['price_review_frequency'] = df['How often do you review or update prices?'].apply(normalize_review_freq)
    
    def parse_conf(text):
        if pd.isna(text): return np.nan, 'Unknown'
        t = str(text).lower()
        val = extract_numeric(t)
        score = np.nan
        if pd.notna(val) and val <= 5:
            score = int(val)
        else:
            if 'very confident' in t or 'extremely' in t: score = 5
            elif 'not confident' in t or 'not very' in t or 'not' in t: score = 1
            elif 'slightly' in t: score = 2
            elif 'somewhat' in t or 'pretty' in t or 'middle' in t: score = 3
            elif 'confident' in t: score = 4
        
        level_map = {1: 'Not confident', 2: 'Slightly confident', 3: 'Somewhat confident', 4: 'Confident', 5: 'Very confident'}
        level = level_map.get(score, 'Unknown')
        return score, level
    confs = df['How confident are you in your pricing?'].apply(parse_conf).apply(pd.Series)
    df['confidence_score_1_5'] = confs[0]
    df['pricing_confidence_level'] = confs[1]
    
    def norm_prof(text):
        if pd.isna(text): return 'Unknown'
        t = str(text).lower()
        if 'not enough' in t or 'low' in t or 'thin' in t: return 'Struggling'
        if 'varies' in t or 'up and down' in t or 'depends' in t: return 'Variable'
        if 'okay' in t or 'acceptable' in t or 'fine' in t or 'good enough' in t: return 'Acceptable'
        if 'very profitable' in t or 'excellent' in t or 'very strong' in t: return 'Very strong'
        if 'good' in t or 'profitable' in t or 'healthy' in t: return 'Good'
        return 'Unknown'
    df['perceived_profitability'] = df['How would you describe profitability?'].apply(norm_prof)
    
    df['plans_price_increase_2026'] = df['Planning to raise prices in 2026?'].astype(str).str.lower().isin(['yes', 'probably'])
    
    keep_cols = ['canonical_email', 'business_name_clean', 'region', 'business_size', 'annual_revenue_band', 
                 'years_in_operation', 'machine_count', 'primary_business_model', 'primary_pricing_method', 
                 'hourly_rate_estimate_usd', 'stitch_rate_per_1000', 'minimum_order_quantity', 'minimum_job_charge_usd', 
                 'charges_setup_fee', 'setup_fee_policy', 'charges_rush_fee', 'rush_fee_policy', 'discounting_frequency', 
                 'price_review_frequency', 'confidence_score_1_5', 'pricing_confidence_level', 'perceived_profitability', 
                 'plans_price_increase_2026', 'Benchmark open comment', 'Response ID', '_simulated_raw_issue']
    missing = [c for c in keep_cols if c not in df.columns]
    for c in missing: df[c] = np.nan
    return df[keep_cols].copy()

df_bench_clean = clean_bench(df_bench_raw)

for i, r in df_bench_raw.iterrows():
    if r['Response ID'] not in df_bench_clean['Response ID'].values:
        audit_records.append({
            'audit_id': f"AUD-B-{i}",
            'dataset': 'Benchmark Survey',
            'raw_response_id': r['Response ID'],
            'issue_type': str(r.get('_simulated_raw_issue', 'unknown')),
            'cleaning_action': 'Excluded',
            'rationale': 'Filtered during cleaning'
        })
        
df_audit = pd.DataFrame(audit_records)

# Visual 2: Funnel Chart
fig, ax = plt.subplots(figsize=(10, 6))
labels = [x[0] for x in diag_funnel]
values = [x[1] for x in diag_funnel]
ax.barh(labels[::-1], values[::-1], color=sns.color_palette("Blues_d", len(values)))
for i, (v, l) in enumerate(zip(values[::-1], labels[::-1])):
    pct = (v / values[0]) * 100
    ax.text(v + 10, i, f"{v} ({pct:.1f}%)", va='center', fontweight='bold')
ax.set_title('Cleaning Funnel: Pricing Diagnostic', fontweight='bold')
plt.tight_layout()
plt.savefig(OUT_DIR / '02_cleaning_funnel.png')
plt.close()
print("✅ Saved 02_cleaning_funnel.png")

# Visual 3: Submission Heatmap
try:
    ts = pd.to_datetime(df_diag_raw['Recorded Date'], errors='coerce', utc=True).dropna()
    heatmap_data = pd.DataFrame({'day': ts.dt.day_name(), 'hour': ts.dt.hour})
    heatmap_pivot = heatmap_data.groupby(['day', 'hour']).size().unstack(fill_value=0)
    days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    heatmap_pivot = heatmap_pivot.reindex(days_order)
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(heatmap_pivot, cmap='YlOrRd', ax=ax)
    ax.set_title('Submission Patterns', fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUT_DIR / '03_submission_heatmap.png')
    plt.close()
    print("✅ Saved 03_submission_heatmap.png")
except Exception as e:
    print(f"Skipping heatmap due to date parsing error: {e}")

print("\n" + "="*60)
print("PHASE 3: SEGMENTATION")
print("="*60)

df_diag_clean['business_size_segment'] = df_diag_clean['business_size']

def map_rev_seg(r):
    if r in ['Under $50k', '$50k-$100k']: return 'Under $100k'
    if r in ['$100k-$250k', '$250k-$500k']: return '$100k-$500k'
    if r in ['$500k-$1M', '$1M+']: return '$500k+'
    return 'Unknown'
df_diag_clean['revenue_segment'] = df_diag_clean['annual_revenue_band'].apply(map_rev_seg)

def map_exp(y):
    if pd.isna(y): return 'Unknown'
    if y <= 3: return 'Newcomer (0-3 yrs)'
    if y <= 10: return 'Established (4-10)'
    if y <= 20: return 'Veteran (11-20)'
    return 'Legacy (21+)'
df_diag_clean['experience_segment'] = df_diag_clean['years_in_operation'].apply(map_exp)

df_diag_clean['pricing_method_segment'] = df_diag_clean['current_pricing_method']

def map_cust(c):
    if 'consumer' in str(c).lower(): return 'Consumer'
    if 'b2b' in str(c).lower() or 'corporate' in str(c).lower(): return 'B2B'
    return 'Mixed'
df_diag_clean['customer_focus_segment'] = df_diag_clean['customer_mix'].apply(map_cust)

driver_cols = ['production_cost_score', 'real_labor_score', 'intended_profit_score', 'capacity_pressure_score', 'end_customer_value_score']
df_diag_clean['weakest_driver_segment'] = df_diag_clean[driver_cols].idxmin(axis=1).str.replace('_score', '').str.replace('_', ' ').str.title()
df_diag_clean['region_segment'] = df_diag_clean['region']

# Visual 4
if HAS_SQUARIFY:
    counts = df_diag_clean.groupby(['business_size_segment', 'pricing_maturity_tier']).size().reset_index(name='count')
    counts = counts.sort_values('count', ascending=False)
    # Only label segments large enough to be readable
    min_label_count = counts['count'].sum() * 0.02  # 2% threshold
    counts['label'] = counts.apply(
        lambda r: (r['business_size_segment'] + '\n' + r['pricing_maturity_tier'] + '\n(' + str(r['count']) + ')')
                  if r['count'] >= min_label_count else '', axis=1)
    colors = [PALETTE.get(t, '#bdc3c7') for t in counts['pricing_maturity_tier']]
    fig, ax = plt.subplots(figsize=(16, 10))
    squarify.plot(sizes=counts['count'], label=counts['label'], color=colors, alpha=0.85, ax=ax,
                  text_kwargs={'fontsize': 9, 'fontweight': 'bold', 'wrap': True},
                  pad=True)
    plt.axis('off')
    plt.title('Business Size by Pricing Maturity', fontsize=18, fontweight='bold', pad=20)
    # Add legend
    legend_handles = [mpatches.Patch(color=PALETTE[k], label=k, alpha=0.85) for k in PALETTE]
    ax.legend(handles=legend_handles, loc='lower right', fontsize=11, framealpha=0.9, title='Maturity Tier')
    plt.tight_layout()
    plt.savefig(OUT_DIR / '04_business_landscape_treemap.png')
    plt.close()
    print("✅ Saved 04_business_landscape_treemap.png")
else:
    print("Skipping treemap (squarify not installed)")

# Visual 5
fig, ax = plt.subplots(figsize=(12, 6))
size_order = ['Solo owner/operator', '2-5 employees', '6-10 employees', '11-25 employees', '26+ employees']
sns.countplot(data=df_diag_clean, x='business_size_segment', hue='pricing_maturity_tier', order=size_order, palette=PALETTE, ax=ax)
ax.set_title('Maturity by Business Size', fontsize=16, fontweight='bold')
ax.set_xlabel('Business Size', fontsize=13)
ax.set_ylabel('Count', fontsize=13)
ax.tick_params(axis='x', rotation=30)
for p in ax.patches:
    h = int(p.get_height())
    if h > 0:  # Skip zero-height bars
        ax.annotate(str(h), (p.get_x() + p.get_width() / 2, p.get_height()),
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
plt.tight_layout()
plt.savefig(OUT_DIR / '05_maturity_by_size.png')
plt.close()
print("✅ Saved 05_maturity_by_size.png")

# Visual 6
rev_order = ['Under $100k', '$100k-$500k', '$500k+']
rev_data = df_diag_clean[df_diag_clean['revenue_segment'].isin(rev_order)].groupby(['revenue_segment', 'pricing_method_segment']).size().unstack(fill_value=0)
rev_pct = rev_data.div(rev_data.sum(axis=1), axis=0) * 100
rev_pct = rev_pct.reindex(rev_order)
fig, ax = plt.subplots(figsize=(14, 7))
rev_pct.plot(kind='barh', stacked=True, ax=ax, colormap='tab20')
# Suppress labels for segments < 5% to prevent overlap
for c in ax.containers:
    labels = [f'{v.get_width():.1f}%' if v.get_width() >= 5.0 else '' for v in c]
    ax.bar_label(c, labels=labels, label_type='center', fontsize=9)
ax.set_title('Pricing Method by Revenue', fontweight='bold')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(OUT_DIR / '06_revenue_by_method.png')
plt.close()
print("✅ Saved 06_revenue_by_method.png")

if 'Response ID' in df_diag_clean.columns: df_diag_clean = df_diag_clean.drop('Response ID', axis=1)
if 'Response ID' in df_bench_clean.columns: df_bench_clean = df_bench_clean.drop('Response ID', axis=1)
if '_simulated_raw_issue' in df_diag_clean.columns: df_diag_clean = df_diag_clean.drop('_simulated_raw_issue', axis=1)
if '_simulated_raw_issue' in df_bench_clean.columns: df_bench_clean = df_bench_clean.drop('_simulated_raw_issue', axis=1)

df_diag_clean.to_csv(CLEAN_DIR / 'pricing_diagnostic_cleaned.csv', index=False)
df_bench_clean.to_csv(CLEAN_DIR / 'benchmark_survey_cleaned.csv', index=False)
df_audit.to_csv(CLEAN_DIR / 'cleaning_audit_log.csv', index=False)

print("\n" + "="*60)
print(f"Cleaned Diagnostic: {df_diag_clean.shape}")
print(f"Cleaned Benchmark: {df_bench_clean.shape}")
print(f"Audit Log Records: {df_audit.shape[0]}")
print("="*60)

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 4 ── DATA MERGING & OVERLAP ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("PHASE 4 ── DATA MERGING & OVERLAP ANALYSIS")
print("=" * 80 + "\n")

# 4.1  Inner-join on canonical_email ─────────────────────────────────────────
df_overlap = pd.merge(
    df_diag_clean,
    df_bench_clean,
    on="canonical_email",
    how="inner",
    suffixes=("_diag", "_bench"),
)
print(f"  Overlap shape : {df_overlap.shape}  (target ≈ 150 rows)")

# 4.2  Diagnostic-only respondents ──────────────────────────────────────────
overlap_emails = set(df_overlap["canonical_email"])
df_diag_only = df_diag_clean[~df_diag_clean["canonical_email"].isin(overlap_emails)].copy()

mean_overlap = df_overlap["diagnostic_total_score"].mean()
mean_diag_only = df_diag_only["diagnostic_total_score"].mean()
print(f"  Mean diagnostic_total_score — Overlap : {mean_overlap:.2f}")
print(f"  Mean diagnostic_total_score — Diag-Only: {mean_diag_only:.2f}")

# ── Visual 7: Venn-style Overlap Diagram ───────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 7))
n_overlap = len(df_overlap)
n_diag_only = len(df_diag_clean) - n_overlap
n_bench_only = len(df_bench_clean) - n_overlap

circle_left = plt.Circle((-0.18, 0.0), 0.38, color="#4C72B0", alpha=0.45,
                          linewidth=2.5, edgecolor="#2a4a7f")
circle_right = plt.Circle((0.18, 0.0), 0.38, color="#DD8452", alpha=0.45,
                           linewidth=2.5, edgecolor="#a55a2a")
ax.add_patch(circle_left)
ax.add_patch(circle_right)

ax.text(-0.35, 0.0, f"Diagnostic\nOnly\n{n_diag_only}",
        ha="center", va="center", fontsize=15, fontweight="bold", color="#1a3050")
ax.text(0.35, 0.0, f"Benchmark\nOnly\n{n_bench_only}",
        ha="center", va="center", fontsize=15, fontweight="bold", color="#5a2a0a")
ax.text(0.0, 0.0, f"Overlap\n{n_overlap}",
        ha="center", va="center", fontsize=16, fontweight="bold", color="#2d0a3a")

ax.set_xlim(-0.75, 0.75)
ax.set_ylim(-0.55, 0.55)
ax.set_aspect("equal")
ax.axis("off")
ax.set_title("Survey Response Overlap — Diagnostic vs Benchmark",
             fontsize=16, fontweight="bold", pad=20)

fig.text(0.5, 0.04,
         f"Diagnostic: {len(df_diag_clean)}  |  Benchmark: {len(df_bench_clean)}  |  Both: {n_overlap}",
         ha="center", fontsize=11, color="#555555")

plt.tight_layout()
plt.savefig(OUT_DIR / "07_venn_overlap.png", dpi=200, bbox_inches="tight")
plt.close()
print("  ✅ Saved 07_venn_overlap.png")

# ── Visual 8: Overlap vs Non-Overlap Score Distributions ───────────────────
fig, ax = plt.subplots(figsize=(10, 7))

overlap_scores = df_overlap["diagnostic_total_score"].dropna()
diagonly_scores = df_diag_only["diagnostic_total_score"].dropna()

violin_data = [overlap_scores.values, diagonly_scores.values]
parts = ax.violinplot(violin_data, positions=[1, 2], showmeans=True,
                      showmedians=True, showextrema=False)

colors_v = ["#4C72B0", "#DD8452"]
for i, pc in enumerate(parts["bodies"]):
    pc.set_facecolor(colors_v[i])
    pc.set_alpha(0.65)
parts["cmeans"].set_color("#222222")
parts["cmedians"].set_color("#888888")

ax.set_xticks([1, 2])
ax.set_xticklabels(["Overlap Respondents", "Diagnostic-Only"], fontsize=13)
ax.set_ylabel("Diagnostic Total Score", fontsize=13)
ax.set_title("Score Distributions — Overlap vs Diagnostic-Only",
             fontsize=16, fontweight="bold")

ax.axhline(overlap_scores.mean(), color=colors_v[0], ls="--", alpha=0.5,
           label=f"Overlap mean = {overlap_scores.mean():.1f}")
ax.axhline(diagonly_scores.mean(), color=colors_v[1], ls="--", alpha=0.5,
           label=f"Diag-Only mean = {diagonly_scores.mean():.1f}")
ax.legend(fontsize=11)
sns.despine()
plt.tight_layout()
plt.savefig(OUT_DIR / "08_overlap_vs_nonoverlap_scores.png", dpi=200, bbox_inches="tight")
plt.close()
print("  ✅ Saved 08_overlap_vs_nonoverlap_scores.png")

# Save overlap dataset ──────────────────────────────────────────────────────
df_overlap.to_csv(CLEAN_DIR / "overlap_analysis_ready.csv", index=False)
print(f"  ✅ Saved overlap_analysis_ready.csv  ({len(df_overlap)} rows)")

# ── Helper: flexible column name resolution ────────────────────────────────
def _col(df, base, suffixes=('', '_diag', '_bench')):
    """Return the first matching column name found in df."""
    for s in suffixes:
        candidate = f"{base}{s}"
        if candidate in df.columns:
            return candidate
    return base  # fallback

# ── PHASE 4.5 ── IQR-BASED OUTLIER FILTERING (WINSORIZATION) ──────────────
print("\n" + "=" * 80)
print("PHASE 4.5 ── IQR OUTLIER FILTERING (1.5× IQR Winsorization)")
print("=" * 80)

def iqr_winsorize(series, label=""):
    """Cap extreme values at 1.5× IQR bounds (Winsorize), preserving sample size."""
    clean = series.dropna()
    if len(clean) < 4:
        return series, 0
    q1, q3 = clean.quantile(0.25), clean.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    n_below = (clean < lower).sum()
    n_above = (clean > upper).sum()
    n_capped = n_below + n_above
    capped = series.clip(lower=lower, upper=upper)
    if n_capped > 0:
        print(f"  {label:40s}  IQR=[{q1:.1f}, {q3:.1f}]  "
              f"Bounds=[{lower:.1f}, {upper:.1f}]  "
              f"Capped: {n_capped} ({n_capped/len(clean)*100:.1f}%)")
    else:
        print(f"  {label:40s}  IQR=[{q1:.1f}, {q3:.1f}]  No outliers detected")
    return capped, n_capped

# Columns to filter in overlap dataset
rate_cols_to_filter = [
    ('hourly_rate_estimate_usd', 'Hourly Rate (USD)'),
    ('stitch_rate_per_1000', 'Stitch Rate per 1,000'),
    ('minimum_job_charge_usd', 'Minimum Job Charge (USD)'),
]

# Save pre-filtering stats for comparison
pre_filter_stats = {}
total_capped = 0

for col_name, label in rate_cols_to_filter:
    actual_col = _col(df_overlap, col_name)
    if actual_col and actual_col in df_overlap.columns:
        pre_filter_stats[label] = {
            'mean': df_overlap[actual_col].mean(),
            'median': df_overlap[actual_col].median(),
            'std': df_overlap[actual_col].std(),
        }
        df_overlap[actual_col], n = iqr_winsorize(df_overlap[actual_col], label)
        total_capped += n

# Also apply to standalone benchmark dataset
for col_name, label in rate_cols_to_filter:
    actual_col = _col(df_bench_clean, col_name)
    if actual_col and actual_col in df_bench_clean.columns:
        df_bench_clean[actual_col], _ = iqr_winsorize(df_bench_clean[actual_col], f"(bench) {label}")

# Print before/after comparison
print(f"\n  Summary: {total_capped} total values capped across overlap dataset")
print(f"  Before → After filtering:")
for col_name, label in rate_cols_to_filter:
    actual_col = _col(df_overlap, col_name)
    if actual_col and actual_col in df_overlap.columns and label in pre_filter_stats:
        pre = pre_filter_stats[label]
        post_mean = df_overlap[actual_col].mean()
        post_median = df_overlap[actual_col].median()
        post_std = df_overlap[actual_col].std()
        print(f"    {label}:")
        print(f"      Mean:   ${pre['mean']:.2f} → ${post_mean:.2f}  (Δ = {post_mean - pre['mean']:+.2f})")
        print(f"      Median: ${pre['median']:.2f} → ${post_median:.2f}")
        print(f"      StdDev: ${pre['std']:.2f} → ${post_std:.2f}")

# Save filtered version
df_overlap.to_csv(CLEAN_DIR / "overlap_analysis_ready_filtered.csv", index=False)
print(f"  ✅ Saved overlap_analysis_ready_filtered.csv  ({len(df_overlap)} rows)")


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 5 ── NLP-POWERED ANALYSIS (AZURE OPENAI)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("PHASE 5 ── NLP-POWERED ANALYSIS (AZURE OPENAI)")
print("=" * 80 + "\n")

# 5a ── Collect open-text comments ──────────────────────────────────────────
diag_comments_raw = (
    df_diag_clean["Open Comment"]
    .dropna()
    .astype(str)
    .str.strip()
)
diag_comments_raw = diag_comments_raw[diag_comments_raw != ""]
diag_comments = diag_comments_raw.tolist()
diag_indices = diag_comments_raw.index.tolist()

bench_comments_raw = (
    df_bench_clean["Benchmark open comment"]
    .dropna()
    .astype(str)
    .str.strip()
)
bench_comments_raw = bench_comments_raw[bench_comments_raw != ""]
bench_comments = bench_comments_raw.tolist()
bench_indices = bench_comments_raw.index.tolist()

all_comments = (
    [(c, "diagnostic", idx) for c, idx in zip(diag_comments, diag_indices)]
    + [(c, "benchmark", idx) for c, idx in zip(bench_comments, bench_indices)]
)
print(f"  Total open-text comments: {len(all_comments)}"
      f"  (diag={len(diag_comments)}, bench={len(bench_comments)})")

# ── Keyword-based fallback classifiers ─────────────────────────────────────
THEME_KEYWORDS = {
    "Setup/Overhead Concerns": ["setup", "overhead", "hidden cost", "supplies"],
    "Rush Job Pricing": ["rush", "turnaround", "urgent", "express"],
    "Customer Price Sensitivity": ["price sensitive", "comparing price",
                                   "shop around", "sticker shock"],
    "Undercharging Fear": ["undercharg", "too low", "leaving money",
                           "raise price", "should charge more"],
    "Process Improvement Need": ["price sheet", "calculator", "system",
                                 "process", "spreadsheet", "formula"],
    "Customer Retention Worry": ["losing customer", "repeat business",
                                 "keep customer", "loyalty"],
    "Market/Competition Pressure": ["competitor", "market rate", "race to bottom",
                                    "lowball", "undercut"],
}


def classify_theme_keyword(text: str) -> str:
    """Keyword-based theme classification fallback."""
    lower = text.lower()
    for theme, keywords in THEME_KEYWORDS.items():
        if any(kw in lower for kw in keywords):
            return theme
    return "General Pricing Challenge"


def classify_sentiment_keyword(text: str) -> str:
    """Keyword-based sentiment classification fallback."""
    lower = text.lower()
    negative_words = ["trouble", "worry", "worries", "losing", "struggling",
                      "frustrat", "problem", "difficult", "hard", "fear",
                      "scared", "afraid", "anxious"]
    positive_words = ["better", "changed everything", "confident", "great",
                      "happy", "improved", "love", "success", "grow"]
    anxious_words = ["uncertain", "anxiety", "nervous", "unsure", "confused",
                     "overwhelm"]

    neg = sum(1 for w in negative_words if w in lower)
    pos = sum(1 for w in positive_words if w in lower)
    anx = sum(1 for w in anxious_words if w in lower)

    if anx >= 1 and neg >= 1:
        return "Anxious"
    if neg > pos:
        return "Negative"
    if pos > neg:
        return "Positive"
    return "Neutral"


# 5b ── Theme Extraction via Azure OpenAI ───────────────────────────────────
THEME_PROMPT_TEMPLATE = """You are an expert survey analyst. Classify each of the following survey comments from embroidery business owners into exactly ONE of these themes:
1. Setup/Overhead Concerns - comments about setup time, overhead costs, hidden costs
2. Rush Job Pricing - comments about rush orders, turnaround time pricing
3. Customer Price Sensitivity - comments about customers comparing prices, being price-sensitive
4. Undercharging Fear - comments about undercharging, leaving money on the table, needing to raise prices
5. Process Improvement Need - comments about needing better systems, price sheets, processes
6. Customer Retention Worry - comments about losing customers, repeat business concerns
7. Market/Competition Pressure - comments about competitors, market rates
8. General Pricing Challenge - general comments about pricing being difficult

Return a JSON array where each element has "index" (0-based position) and "theme" (the theme name from the list above).

Comments:
{comments}"""

SENTIMENT_PROMPT_TEMPLATE = """Classify the sentiment of each comment as one of: Positive, Neutral, Negative, Anxious.
Return a JSON array with "index" and "sentiment".

Comments:
{comments}"""

BATCH_SIZE = 40

# Initialize columns
df_diag_clean["comment_theme"] = np.nan
df_diag_clean["comment_sentiment"] = np.nan
df_bench_clean["comment_theme"] = np.nan
df_bench_clean["comment_sentiment"] = np.nan

use_api = bool(AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY)
client = None
if use_api:
    try:
        client = AzureOpenAI(
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_key=AZURE_OPENAI_API_KEY,
            api_version=AZURE_OPENAI_API_VERSION,
        )
        print("  Azure OpenAI client created ✅")
    except Exception as e:
        print(f"  ⚠️  Azure client init failed: {e} — using keyword fallback")
        client = None


def call_openai_batch(prompt_text: str, retries: int = 2) -> str | None:
    """Call Azure OpenAI with retries. Returns raw content or None."""
    if client is None:
        return None
    for attempt in range(retries + 1):
        try:
            resp = client.chat.completions.create(
                model=AZURE_OPENAI_DEPLOYMENT,
                messages=[{"role": "user", "content": prompt_text}],
                temperature=0.0,
                max_tokens=2000,
            )
            return resp.choices[0].message.content
        except Exception as e:
            print(f"    ⚠️  API attempt {attempt + 1} failed: {e}")
            if attempt < retries:
                time.sleep(2 ** attempt)
    return None


def parse_json_response(raw: str | None) -> list[dict] | None:
    """Attempt to parse JSON array from LLM response."""
    if raw is None:
        return None
    # Try to find JSON array in the response
    try:
        # Strip markdown code fences if present
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned)
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # Try to find array within the text
        match = re.search(r"\[.*\]", raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                return None
    return None


# ── Process themes ─────────────────────────────────────────────────────────
print("  Processing comment themes...")
all_themes: dict[tuple[str, int], str] = {}

for batch_start in range(0, len(all_comments), BATCH_SIZE):
    batch = all_comments[batch_start: batch_start + BATCH_SIZE]
    numbered = "\n".join(f"{i}. {c}" for i, (c, _, _) in enumerate(batch))
    prompt = THEME_PROMPT_TEMPLATE.format(comments=numbered)

    result = call_openai_batch(prompt)
    parsed = parse_json_response(result)

    for j, (comment_text, source, orig_idx) in enumerate(batch):
        theme = None
        if parsed:
            entry = next((e for e in parsed if e.get("index") == j), None)
            if entry:
                theme = entry.get("theme")
        if theme is None:
            theme = classify_theme_keyword(comment_text)
        all_themes[(source, orig_idx)] = theme

# Map themes back to DataFrames
df_diag_clean["comment_theme"] = pd.Series(dtype='object')
df_bench_clean["comment_theme"] = pd.Series(dtype='object')
for (source, orig_idx), theme in all_themes.items():
    if source == "diagnostic":
        df_diag_clean.loc[orig_idx, "comment_theme"] = theme
    else:
        df_bench_clean.loc[orig_idx, "comment_theme"] = theme

print(f"    Themes classified: {len(all_themes)}")

# ── Process sentiments ─────────────────────────────────────────────────────
print("  Processing comment sentiments...")
all_sentiments: dict[tuple[str, int], str] = {}

for batch_start in range(0, len(all_comments), BATCH_SIZE):
    batch = all_comments[batch_start: batch_start + BATCH_SIZE]
    numbered = "\n".join(f"{i}. {c}" for i, (c, _, _) in enumerate(batch))
    prompt = SENTIMENT_PROMPT_TEMPLATE.format(comments=numbered)

    result = call_openai_batch(prompt)
    parsed = parse_json_response(result)

    for j, (comment_text, source, orig_idx) in enumerate(batch):
        sentiment = None
        if parsed:
            entry = next((e for e in parsed if e.get("index") == j), None)
            if entry:
                sentiment = entry.get("sentiment")
        if sentiment is None:
            sentiment = classify_sentiment_keyword(comment_text)
        all_sentiments[(source, orig_idx)] = sentiment

df_diag_clean["comment_sentiment"] = pd.Series(dtype='object')
df_bench_clean["comment_sentiment"] = pd.Series(dtype='object')
for (source, orig_idx), sentiment in all_sentiments.items():
    if source == "diagnostic":
        df_diag_clean.loc[orig_idx, "comment_sentiment"] = sentiment
    else:
        df_bench_clean.loc[orig_idx, "comment_sentiment"] = sentiment

print(f"    Sentiments classified: {len(all_sentiments)}")

# 5d ── NLP Risk Micro-Segments ─────────────────────────────────────────────
def assign_nlp_risk(row):
    theme = row.get("comment_theme")
    sentiment = row.get("comment_sentiment")
    comment = row.get("Open Comment")

    if pd.isna(comment) or (isinstance(comment, str) and comment.strip() == ""):
        return "No Comment"
    if theme in ("Undercharging Fear", "Rush Job Pricing") and sentiment in ("Negative", "Anxious"):
        return "High Risk"
    if theme == "Process Improvement Need" and sentiment == "Neutral":
        return "Improvement Focused"
    return "Moderate"


df_diag_clean["nlp_risk_segment"] = df_diag_clean.apply(assign_nlp_risk, axis=1)
risk_counts = df_diag_clean["nlp_risk_segment"].value_counts()
print("  NLP Risk Segments:")
for seg, cnt in risk_counts.items():
    print(f"    {seg}: {cnt}")

# ── Visual 9: Comment Theme Frequencies ────────────────────────────────────
theme_counts = (
    pd.concat([
        df_diag_clean["comment_theme"].dropna(),
        df_bench_clean["comment_theme"].dropna(),
    ])
    .value_counts()
    .sort_values()
)

fig, ax = plt.subplots(figsize=(12, 7))
palette_9 = sns.color_palette("Set2", n_colors=len(theme_counts))
bars = ax.barh(theme_counts.index, theme_counts.values, color=palette_9,
               edgecolor="white", linewidth=0.8)

for bar, count in zip(bars, theme_counts.values):
    ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
            str(count), va="center", fontsize=11, fontweight="bold")

ax.set_xlabel("Number of Comments", fontsize=13)
ax.set_title("Open-Comment Theme Frequencies", fontsize=16, fontweight="bold")
sns.despine()
plt.tight_layout()
plt.savefig(OUT_DIR / "09_comment_themes.png", dpi=200, bbox_inches="tight")
plt.close()
print("  ✅ Saved 09_comment_themes.png")

# ── Visual 10: Themes vs Pricing Maturity Heatmap ─────────────────────────
ct_theme_maturity = pd.crosstab(
    df_diag_clean["comment_theme"].fillna("(no comment)"),
    df_diag_clean["pricing_maturity_tier"],
)

fig, ax = plt.subplots(figsize=(14, 8))
sns.heatmap(ct_theme_maturity, annot=True, fmt="d", cmap="YlOrRd",
            linewidths=0.5, linecolor="white", ax=ax, cbar_kws={"label": "Count"})
ax.set_title("Comment Themes × Pricing Maturity Tier",
             fontsize=16, fontweight="bold")
ax.set_xlabel("Pricing Maturity Tier", fontsize=13)
ax.set_ylabel("Comment Theme", fontsize=13)
plt.tight_layout()
plt.savefig(OUT_DIR / "10_themes_vs_maturity.png", dpi=200, bbox_inches="tight")
plt.close()
print("  ✅ Saved 10_themes_vs_maturity.png")

# ── Visual 11: Sentiment by Maturity (Stacked Bar) ────────────────────────
sentiment_order = ["Positive", "Neutral", "Negative", "Anxious"]
sentiment_colors = {"Positive": "#2ecc71", "Neutral": "#95a5a6",
                    "Negative": "#e67e22", "Anxious": "#e74c3c"}

ct_sent_mat = pd.crosstab(
    df_diag_clean["pricing_maturity_tier"],
    df_diag_clean["comment_sentiment"],
    normalize="index",
) * 100

# Reindex to ensure consistent column order
for s in sentiment_order:
    if s not in ct_sent_mat.columns:
        ct_sent_mat[s] = 0.0
ct_sent_mat = ct_sent_mat[sentiment_order]

fig, ax = plt.subplots(figsize=(12, 7))
bottom = np.zeros(len(ct_sent_mat))
for sent in sentiment_order:
    vals = ct_sent_mat[sent].values
    ax.bar(ct_sent_mat.index, vals, bottom=bottom,
           color=sentiment_colors[sent], label=sent, edgecolor="white", linewidth=0.5)
    bottom += vals

ax.set_ylabel("Percentage of Respondents (%)", fontsize=13)
ax.set_xlabel("Pricing Maturity Tier", fontsize=13)
ax.set_title("Comment Sentiment Distribution by Maturity Tier",
             fontsize=16, fontweight="bold")
ax.legend(title="Sentiment", fontsize=11, title_fontsize=12)
sns.despine()
plt.tight_layout()
plt.savefig(OUT_DIR / "11_sentiment_by_maturity.png", dpi=200, bbox_inches="tight")
plt.close()
print("  ✅ Saved 11_sentiment_by_maturity.png")

# ── Visual 12: Score vs Confidence coloured by NLP Risk ────────────────────
# Merge NLP risk onto overlap data
if "nlp_risk_segment" not in df_overlap.columns:
    # Map from df_diag_clean via canonical_email
    risk_map = df_diag_clean.set_index("canonical_email")["nlp_risk_segment"]
    df_overlap["nlp_risk_segment"] = df_overlap["canonical_email"].map(risk_map)

fig, ax = plt.subplots(figsize=(12, 8))
risk_palette = {"High Risk": "#e74c3c", "Moderate": "#f39c12",
                "Improvement Focused": "#3498db", "No Comment": "#bdc3c7"}
risk_segments = df_overlap["nlp_risk_segment"].dropna().unique()

for seg in risk_segments:
    mask = df_overlap["nlp_risk_segment"] == seg
    subset = df_overlap.loc[mask]
    jitter_x = np.random.normal(0, 0.25, size=len(subset))
    jitter_y = np.random.normal(0, 0.05, size=len(subset))
    ax.scatter(
        subset["diagnostic_total_score"] + jitter_x,
        subset["confidence_score_1_5"] + jitter_y,
        label=seg,
        color=risk_palette.get(seg, "#999999"),
        alpha=0.7,
        s=60,
        edgecolors="white",
        linewidth=0.5,
    )

ax.set_xlabel("Diagnostic Total Score", fontsize=13)
ax.set_ylabel("Confidence Score (1–5)", fontsize=13)
ax.set_title("Diagnostic Score vs Confidence — by NLP Risk Segment",
             fontsize=16, fontweight="bold")
ax.legend(title="NLP Risk Segment", fontsize=11, title_fontsize=12)
sns.despine()
plt.tight_layout()
plt.savefig(OUT_DIR / "12_score_vs_confidence_nlp.png", dpi=200, bbox_inches="tight")
plt.close()
print("  ✅ Saved 12_score_vs_confidence_nlp.png")


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 6 ── EXPLORATORY DATA ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("PHASE 6 ── EXPLORATORY DATA ANALYSIS")
print("=" * 80 + "\n")

# ── Visual 13: Diagnostic Total Score Distribution ─────────────────────────
fig, ax = plt.subplots(figsize=(12, 7))
scores = df_diag_clean["diagnostic_total_score"].dropna()

# Background shading for tiers
ax.axvspan(scores.min() - 0.5, 7, alpha=0.10, color="red", label="Blind Spots (0–6)")
ax.axvspan(7, 14, alpha=0.10, color="#f1c40f", label="Partial Pricing (7–13)")
ax.axvspan(14, scores.max() + 0.5, alpha=0.10, color="green", label="Strong Pricing (14–20)")

sns.histplot(scores, kde=True, bins=21, color="#2c3e50", alpha=0.65, ax=ax,
             edgecolor="white", linewidth=0.6)

ax.axvline(7, color="#c0392b", ls="--", lw=2, label="Tier cutoff = 7")
ax.axvline(14, color="#27ae60", ls="--", lw=2, label="Tier cutoff = 14")

ax.set_xlabel("Diagnostic Total Score", fontsize=13)
ax.set_ylabel("Count", fontsize=13)
ax.set_title("Distribution of Diagnostic Total Scores",
             fontsize=16, fontweight="bold")
ax.legend(fontsize=10, loc="upper right")
sns.despine()
plt.tight_layout()
plt.savefig(OUT_DIR / "13_score_distribution.png", dpi=200, bbox_inches="tight")
plt.close()
print("  ✅ Saved 13_score_distribution.png")

# ── Visual 14: Radar / Spider Chart by Maturity Tier ──────────────────────
driver_cols = [
    "production_cost_score", "real_labor_score", "intended_profit_score",
    "capacity_pressure_score", "end_customer_value_score",
]
driver_labels = [
    "Production Cost", "Real Labor", "Intended Profit",
    "Capacity Pressure", "End Customer Value",
]

tiers = sorted(df_diag_clean["pricing_maturity_tier"].dropna().unique())
tier_colors = {
    "Pricing Blind Spots": "#e74c3c",
    "Partial Pricing System": "#f39c12",
    "Strong Pricing System": "#2ecc71",
}
tier_markers = {
    "Pricing Blind Spots": "o",
    "Partial Pricing System": "s",
    "Strong Pricing System": "D",
}

n_axes = len(driver_cols)
angles = np.linspace(0, 2 * np.pi, n_axes, endpoint=False).tolist()
angles += angles[:1]  # close the polygon

fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

for tier in tiers:
    mask = df_diag_clean["pricing_maturity_tier"] == tier
    means = df_diag_clean.loc[mask, driver_cols].mean().values.tolist()
    means += means[:1]
    color = tier_colors.get(tier, "#888888")
    marker = tier_markers.get(tier, "o")
    ax.plot(angles, means, marker=marker, linestyle='-', label=tier, color=color,
            linewidth=2.2, markersize=7)
    ax.fill(angles, means, alpha=0.12, color=color)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(driver_labels, fontsize=12)
ax.set_ylim(0, 4.2)
ax.set_title("Mean Driver Sub-Scores by Pricing Maturity Tier",
             fontsize=16, fontweight="bold", pad=30)
ax.legend(loc="upper right", bbox_to_anchor=(1.30, 1.12), fontsize=12,
          framealpha=0.9, edgecolor='#cccccc')
plt.tight_layout()
plt.savefig(OUT_DIR / "14_radar_drivers.png", bbox_inches="tight")
plt.close()
print("  ✅ Saved 14_radar_drivers.png")

# ── Visual 15: Correlation Heatmap (q1 – q10) ─────────────────────────────
q_cols = [f"q{i}_score" for i in range(1, 11)]
corr_matrix = df_diag_clean[q_cols].corr()
mask_upper = np.triu(np.ones_like(corr_matrix, dtype=bool))

fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(corr_matrix, mask=mask_upper, annot=True, fmt=".2f",
            cmap="coolwarm", center=0, linewidths=0.5, linecolor="white",
            square=True, ax=ax, cbar_kws={"shrink": 0.8, "label": "Pearson r"})
ax.set_title("Question Score Correlation Matrix (q1 – q10)",
             fontsize=16, fontweight="bold")
plt.tight_layout()
plt.savefig(OUT_DIR / "15_correlation_heatmap.png", dpi=200, bbox_inches="tight")
plt.close()
print("  ✅ Saved 15_correlation_heatmap.png")

# ── Visual 16: Box Plots for Each Question Score ──────────────────────────
q_labels = [
    "Production Cost\n(Q1)", "Production Cost\n(Q2)",
    "Real Labor\n(Q3)", "Real Labor\n(Q4)",
    "Intended Profit\n(Q5)", "Intended Profit\n(Q6)",
    "Capacity Pressure\n(Q7)", "Capacity Pressure\n(Q8)",
    "End Cust Value\n(Q9)", "End Cust Value\n(Q10)",
]

fig, axes = plt.subplots(2, 5, figsize=(18, 8), sharey=True)
palette_16 = sns.color_palette("husl", 10)

for i, (col, label) in enumerate(zip(q_cols, q_labels)):
    row, col_idx = divmod(i, 5)
    ax = axes[row][col_idx]
    data = df_diag_clean[col].dropna()
    bp = ax.boxplot(data, patch_artist=True, widths=0.5,
                    boxprops=dict(facecolor=palette_16[i], alpha=0.7),
                    medianprops=dict(color="#2c3e50", linewidth=2))
    ax.set_title(label, fontsize=10, fontweight="bold")
    ax.set_xticks([])
    if col_idx == 0:
        ax.set_ylabel("Score", fontsize=11)

fig.suptitle("Question-Level Score Distributions",
             fontsize=16, fontweight="bold")
fig.subplots_adjust(top=0.90)
plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig(OUT_DIR / "16_question_boxplots.png", dpi=200, bbox_inches="tight")
plt.close()
print("  ✅ Saved 16_question_boxplots.png")

# ── Visual 17: Hourly Rate by Pricing Method (Violin) ─────────────────────
bench_rates = df_bench_clean.dropna(subset=["hourly_rate_estimate_usd"]).copy()

fig, ax = plt.subplots(figsize=(14, 7))
methods_ordered = (
    bench_rates.groupby("primary_pricing_method")["hourly_rate_estimate_usd"]
    .median()
    .sort_values(ascending=False)
    .index.tolist()
)

if len(methods_ordered) > 0:
    sns.violinplot(
        data=bench_rates, x="primary_pricing_method", y="hourly_rate_estimate_usd",
        order=methods_ordered, palette="Set2", inner="box", ax=ax, cut=0,
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=25, ha="right", fontsize=11)

ax.set_xlabel("Primary Pricing Method", fontsize=13)
ax.set_ylabel("Hourly Rate Estimate (USD)", fontsize=13)
ax.set_title("Hourly Rate Distribution by Pricing Method",
             fontsize=16, fontweight="bold")
sns.despine()
plt.tight_layout()
plt.savefig(OUT_DIR / "17_hourly_rate_violin.png", dpi=200, bbox_inches="tight")
plt.close()
print("  ✅ Saved 17_hourly_rate_violin.png")

# ── Visual 18: Hourly Rate vs Stitch Rate Scatter ─────────────────────────
bench_scatter = df_bench_clean.dropna(
    subset=["hourly_rate_estimate_usd", "stitch_rate_per_1000"]
).copy()

fig, ax = plt.subplots(figsize=(12, 8))
if len(bench_scatter) > 0:
    methods_in_scatter = bench_scatter["primary_pricing_method"].unique()
    palette_18 = dict(zip(methods_in_scatter,
                          sns.color_palette("tab10", len(methods_in_scatter))))
    for method in methods_in_scatter:
        subset = bench_scatter[bench_scatter["primary_pricing_method"] == method]
        ax.scatter(subset["hourly_rate_estimate_usd"],
                   subset["stitch_rate_per_1000"],
                   label=method, color=palette_18[method],
                   alpha=0.7, s=60, edgecolors="white", linewidth=0.5)

    # Overall regression line
    sns.regplot(data=bench_scatter, x="hourly_rate_estimate_usd",
                y="stitch_rate_per_1000", scatter=False,
                line_kws={"color": "#2c3e50", "lw": 2, "ls": "--"},
                ax=ax, ci=95)

ax.set_xlabel("Hourly Rate Estimate (USD)", fontsize=13)
ax.set_ylabel("Stitch Rate per 1,000", fontsize=13)
ax.set_title("Hourly Rate vs Stitch Rate — by Pricing Method",
             fontsize=16, fontweight="bold")
ax.legend(title="Pricing Method", fontsize=10, title_fontsize=11)
sns.despine()
plt.tight_layout()
plt.savefig(OUT_DIR / "18_rate_scatter.png", dpi=200, bbox_inches="tight")
plt.close()
print("  ✅ Saved 18_rate_scatter.png")

# ── Visual 19: Setup & Rush Fee Adoption by Business Size ─────────────────
fee_data = df_bench_clean.copy()
fee_grouped = fee_data.groupby("business_size").agg(
    setup_pct=("charges_setup_fee", lambda x: x.astype(bool).mean() * 100),
    rush_pct=("charges_rush_fee", lambda x: x.astype(bool).mean() * 100),
).reset_index()

# Sort by logical business size order
size_order_map = {'Solo owner/operator': 0, '2-5 employees': 1, '6-10 employees': 2, '11-25 employees': 3, '26+ employees': 4, 'Unknown': 5}
fee_grouped['sort_key'] = fee_grouped['business_size'].map(size_order_map).fillna(99)
fee_grouped = fee_grouped.sort_values('sort_key').drop('sort_key', axis=1)

fig, ax = plt.subplots(figsize=(12, 7))
x_pos = np.arange(len(fee_grouped))
bar_w = 0.35

bars1 = ax.bar(x_pos - bar_w / 2, fee_grouped["setup_pct"], bar_w,
               label="Setup Fee", color="#3498db", edgecolor="white")
bars2 = ax.bar(x_pos + bar_w / 2, fee_grouped["rush_pct"], bar_w,
               label="Rush Fee", color="#e67e22", edgecolor="white")

for bar in list(bars1) + list(bars2):
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2, h + 1, f"{h:.0f}%",
            ha="center", va="bottom", fontsize=10, fontweight="bold")

ax.set_xticks(x_pos)
ax.set_xticklabels(fee_grouped["business_size"], fontsize=11, rotation=15)
ax.set_ylabel("% of Respondents Charging Fee", fontsize=13)
ax.set_xlabel("Business Size", fontsize=13)
ax.set_title("Setup & Rush Fee Adoption by Business Size",
             fontsize=16, fontweight="bold")
ax.legend(fontsize=12)
ax.set_ylim(0, 110)
sns.despine()
plt.tight_layout()
plt.savefig(OUT_DIR / "19_fee_adoption.png", dpi=200, bbox_inches="tight")
plt.close()
print("  ✅ Saved 19_fee_adoption.png")

# ── Visual 20: Confidence vs Perceived Profitability ──────────────────────
ct_conf_prof = pd.crosstab(
    df_bench_clean["pricing_confidence_level"],
    df_bench_clean["perceived_profitability"],
)

fig, ax = plt.subplots(figsize=(12, 8))
sns.heatmap(ct_conf_prof, annot=True, fmt="d", cmap="Blues",
            linewidths=0.6, linecolor="white", ax=ax,
            cbar_kws={"label": "Respondent Count"})
ax.set_title("Pricing Confidence vs Perceived Profitability",
             fontsize=16, fontweight="bold")
ax.set_xlabel("Perceived Profitability", fontsize=13)
ax.set_ylabel("Pricing Confidence Level", fontsize=13)
plt.tight_layout()
plt.savefig(OUT_DIR / "20_confidence_vs_profit.png", dpi=200, bbox_inches="tight")
plt.close()
print("  ✅ Saved 20_confidence_vs_profit.png")

# ── Visual 21: Paired Dot / Slope Chart (Overlap) ─────────────────────────
fig, ax = plt.subplots(figsize=(12, 8))

# Use diag maturity tier — resolve column name (may have suffix)
tier_col = ("pricing_maturity_tier_diag"
            if "pricing_maturity_tier_diag" in df_overlap.columns
            else "pricing_maturity_tier")

slope_data = df_overlap.dropna(
    subset=["diagnostic_total_score", "confidence_score_1_5"]
).copy()
slope_data["confidence_scaled"] = slope_data["confidence_score_1_5"] * 4  # → 0-20

slope_colors = {
    "Pricing Blind Spots": "#e74c3c",
    "Partial Pricing System": "#f39c12",
    "Strong Pricing System": "#2ecc71",
}

for _, row in slope_data.iterrows():
    tier = row.get(tier_col, "Unknown")
    color = slope_colors.get(tier, "#aaaaaa")
    ax.plot(
        [0, 1],
        [row["diagnostic_total_score"], row["confidence_scaled"]],
        color=color, alpha=0.35, linewidth=1.2,
    )

# Add mean markers
for tier in slope_colors:
    mask = slope_data[tier_col] == tier
    if mask.any():
        mean_left = slope_data.loc[mask, "diagnostic_total_score"].mean()
        mean_right = slope_data.loc[mask, "confidence_scaled"].mean()
        ax.plot([0, 1], [mean_left, mean_right], color=slope_colors[tier],
                linewidth=3.5, alpha=0.9, label=f"{tier} mean")
        ax.scatter([0], [mean_left], color=slope_colors[tier], s=100, zorder=5)
        ax.scatter([1], [mean_right], color=slope_colors[tier], s=100, zorder=5)

ax.set_xticks([0, 1])
ax.set_xticklabels(["Diagnostic Total Score", "Confidence (scaled 0–20)"], fontsize=13)
ax.set_ylabel("Score", fontsize=13)
ax.set_title("Paired Comparison — Diagnostic Score vs Self-Rated Confidence",
             fontsize=16, fontweight="bold")
ax.legend(fontsize=11, loc="upper left")
ax.set_xlim(-0.15, 1.15)
sns.despine()
plt.tight_layout()
plt.savefig(OUT_DIR / "21_paired_dot_plot.png", dpi=200, bbox_inches="tight")
plt.close()
print("  ✅ Saved 21_paired_dot_plot.png")

# ── Visual 22: Bubble Chart — Region × Pricing Method ─────────────────────
bench_bubble = df_bench_clean.dropna(
    subset=["region", "primary_pricing_method"]
).copy()

bubble_agg = (
    bench_bubble.groupby(["region", "primary_pricing_method"])
    .agg(
        count=("canonical_email", "size"),
        mean_rate=("hourly_rate_estimate_usd", "mean"),
    )
    .reset_index()
)

# Encode categorical axes as numeric positions
regions_list = sorted(bubble_agg["region"].unique())
methods_list = sorted(bubble_agg["primary_pricing_method"].unique())
region_map = {r: i for i, r in enumerate(regions_list)}
method_map = {m: i for i, m in enumerate(methods_list)}
bubble_agg["x"] = bubble_agg["region"].map(region_map)
bubble_agg["y"] = bubble_agg["primary_pricing_method"].map(method_map)

fig, ax = plt.subplots(figsize=(14, 9))
sc = ax.scatter(
    bubble_agg["x"], bubble_agg["y"],
    s=bubble_agg["count"] * 80,
    c=bubble_agg["mean_rate"],
    cmap="RdYlGn", edgecolors="#333333", linewidth=0.8, alpha=0.85,
)

for _, row in bubble_agg.iterrows():
    ax.annotate(str(int(row["count"])),
                (row["x"], row["y"]),
                ha="center", va="center", fontsize=9, fontweight="bold")

ax.set_xticks(range(len(regions_list)))
ax.set_xticklabels(regions_list, rotation=30, ha="right", fontsize=11)
ax.set_yticks(range(len(methods_list)))
ax.set_yticklabels(methods_list, fontsize=11)
ax.set_xlabel("Region", fontsize=13)
ax.set_ylabel("Primary Pricing Method", fontsize=13)
ax.set_title("Respondent Distribution — Region × Pricing Method\n"
             "(bubble size = count, color = mean hourly rate USD)",
             fontsize=15, fontweight="bold")

cbar = plt.colorbar(sc, ax=ax, shrink=0.7, pad=0.02)
cbar.set_label("Mean Hourly Rate (USD)", fontsize=12)

plt.tight_layout()
plt.savefig(OUT_DIR / "22_bubble_chart.png", dpi=200, bbox_inches="tight")
plt.close()
print("  ✅ Saved 22_bubble_chart.png")

# ── Phase 6 complete ──────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("PHASES 4–6 COMPLETE")
print(f"  Charts saved to: {OUT_DIR.resolve()}")
print(f"  Overlap CSV    : {(CLEAN_DIR / 'overlap_analysis_ready.csv').resolve()}")
print("=" * 80)

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 7 — STATISTICAL HYPOTHESIS TESTING
# ═══════════════════════════════════════════════════════════════════════════════

print("\n")
print("=" * 72)
print("█" * 72)
print("██                                                              ██")
print("██       PHASE 7: STATISTICAL HYPOTHESIS TESTING                ██")
print("██       Six rigorous tests with effect sizes & post-hocs       ██")
print("██                                                              ██")
print("█" * 72)
print("=" * 72)
print()


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  TEST 1: One-Way ANOVA — Hourly Rate by Pricing Maturity Tier           ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
print("─" * 72)
print("  TEST 1: One-Way ANOVA — Hourly Rate by Pricing Maturity Tier")
print("─" * 72)

tier_col_ov = _col(df_overlap, 'pricing_maturity_tier')
rate_col_ov = _col(df_overlap, 'hourly_rate_estimate_usd')

anova_df = df_overlap[[tier_col_ov, rate_col_ov]].dropna()
anova_groups = [grp[rate_col_ov].values for _, grp in anova_df.groupby(tier_col_ov)]
anova_labels = [name for name, _ in anova_df.groupby(tier_col_ov)]

if len(anova_groups) >= 2 and all(len(g) >= 2 for g in anova_groups):
    # ── Normality Check (Shapiro-Wilk) ──
    print("\n  ➤ Normality Check (Shapiro-Wilk per group):")
    normality_ok = True
    for lbl, grp in zip(anova_labels, anova_groups):
        if len(grp) >= 3:
            w_stat, p_sw = stats.shapiro(grp)
            normal_flag = "✅ normal" if p_sw >= 0.05 else "⚠ non-normal"
            if p_sw < 0.05:
                normality_ok = False
            print(f"    {lbl:25s}  W={w_stat:.4f}  p={p_sw:.4f}  → {normal_flag}")
        else:
            print(f"    {lbl:25s}  (too few samples for Shapiro-Wilk)")

    F_stat, p_anova = f_oneway(*anova_groups)

    # Effect size η²
    grand_mean = anova_df[rate_col_ov].mean()
    ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in anova_groups)
    ss_total = ((anova_df[rate_col_ov] - grand_mean) ** 2).sum()
    eta_sq = ss_between / ss_total if ss_total > 0 else 0.0

    print(f"\n  Groups analysed : {dict(zip(anova_labels, [len(g) for g in anova_groups]))}")
    print(f"  F-statistic     : {F_stat:.4f}")
    print(f"  p-value         : {p_anova:.6f}  {'*** SIGNIFICANT' if p_anova < 0.05 else '(not significant)'}")
    print(f"  Effect size η²  : {eta_sq:.4f}  ({'large' if eta_sq > 0.14 else 'medium' if eta_sq > 0.06 else 'small'})")

    # Non-parametric backup if normality violated
    if not normality_ok:
        print("\n  ➤ Normality violated — running Kruskal-Wallis (non-parametric backup):")
        H_kw, p_kw_backup = stats.kruskal(*anova_groups)
        # Epsilon-squared effect size for Kruskal-Wallis
        n_total = sum(len(g) for g in anova_groups)
        eps_sq = (H_kw - len(anova_groups) + 1) / (n_total - len(anova_groups)) if n_total > len(anova_groups) else 0
        print(f"    H-statistic   : {H_kw:.4f}")
        print(f"    p-value       : {p_kw_backup:.6f}  {'*** SIGNIFICANT' if p_kw_backup < 0.05 else '(not significant)'}")
        print(f"    ε² effect size: {eps_sq:.4f}")
        print(f"    ⓘ  Both tests {'agree' if (p_anova < 0.05) == (p_kw_backup < 0.05) else 'DISAGREE'} on significance.")
    else:
        print("\n  ✅ Normality assumption met — ANOVA results are reliable.")

    # Group means for interpretation
    for lbl, grp in zip(anova_labels, anova_groups):
        print(f"    • {lbl:20s}  mean = ${grp.mean():.2f}/hr  (n={len(grp)})")

    # Tukey HSD post-hoc
    if p_anova < 0.05:
        print("\n  ➤ Tukey HSD Post-Hoc Comparisons:")
        tukey = pairwise_tukeyhsd(anova_df[rate_col_ov], anova_df[tier_col_ov], alpha=0.05)
        print(tukey)

    print("\n  📊 Business Interpretation:")
    if p_anova < 0.05:
        print("     Hourly rates differ SIGNIFICANTLY across pricing maturity tiers.")
        print("     Businesses with stronger pricing systems charge measurably more.")
    else:
        print("     No statistically significant difference in hourly rates across tiers.")
else:
    F_stat, p_anova, eta_sq = np.nan, np.nan, np.nan
    print("  ⚠ Insufficient groups or data for ANOVA.")

print()

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  TEST 2: Chi-Square — Pricing Confidence vs Perceived Profitability     ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
print("─" * 72)
print("  TEST 2: Chi-Square — Pricing Confidence vs Perceived Profitability")
print("─" * 72)

conf_col = _col(df_bench_clean, 'pricing_confidence_level')
prof_col = _col(df_bench_clean, 'perceived_profitability')

chi_df = df_bench_clean[[conf_col, prof_col]].dropna()
contingency = pd.crosstab(chi_df[conf_col], chi_df[prof_col])

if contingency.shape[0] >= 2 and contingency.shape[1] >= 2:
    chi2, p_chi, dof, expected = chi2_contingency(contingency)
    n_chi = contingency.values.sum()
    k_chi = min(contingency.shape)
    cramers_v = np.sqrt(chi2 / (n_chi * (k_chi - 1))) if (n_chi * (k_chi - 1)) > 0 else 0.0

    print(f"\n  Contingency table shape : {contingency.shape[0]} × {contingency.shape[1]}")
    print(f"  Chi-square statistic    : {chi2:.4f}")
    print(f"  p-value                 : {p_chi:.6f}  {'*** SIGNIFICANT' if p_chi < 0.05 else '(not significant)'}")
    print(f"  Degrees of freedom      : {dof}")
    print(f"  Cramér's V              : {cramers_v:.4f}  ({'strong' if cramers_v > 0.5 else 'moderate' if cramers_v > 0.3 else 'weak'})")
    print(f"\n  Contingency Table (observed counts):")
    print(contingency.to_string())
else:
    chi2, p_chi, cramers_v = np.nan, np.nan, np.nan
    print("  ⚠ Insufficient categories for chi-square test.")

# --- Visual 23: Chi-Square Heatmap ----------------------------------------
print("\n  ➤ Generating Visual 23: Chi-Square Contingency Heatmap …")
fig, ax = plt.subplots(figsize=(12, 8))
sns.heatmap(contingency, annot=True, fmt='d', cmap='Blues', linewidths=0.5,
            linecolor='white', cbar_kws={'label': 'Count'}, ax=ax)
ax.set_title('Pricing Confidence vs Perceived Profitability\n(Observed Counts)',
             fontsize=16, fontweight='bold', pad=15)
ax.set_xlabel('Perceived Profitability', fontsize=13)
ax.set_ylabel('Pricing Confidence Level', fontsize=13)
if not np.isnan(chi2):
    ax.text(0.98, 0.02, f"χ² = {chi2:.2f}  |  p = {p_chi:.4f}  |  V = {cramers_v:.3f}",
            transform=ax.transAxes, ha='right', va='bottom', fontsize=10,
            bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow', edgecolor='gray'))
plt.tight_layout()
plt.savefig(OUT_DIR / '23_chi_square_mosaic.png', dpi=200, bbox_inches='tight')
plt.close()
print("  ✅ Saved: 23_chi_square_mosaic.png")
print()

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  TEST 3: Mann-Whitney U — Diagnostic Scores: Overlap vs Non-Overlap    ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
print("─" * 72)
print("  TEST 3: Mann-Whitney U — Overlap vs Non-Overlap Diagnostic Scores")
print("─" * 72)

score_col_ov = _col(df_overlap, 'diagnostic_total_score')
score_col_do = _col(df_diag_only, 'diagnostic_total_score')

overlap_scores = df_overlap[score_col_ov].dropna()
non_overlap_scores = df_diag_only[score_col_do].dropna()

if len(overlap_scores) >= 2 and len(non_overlap_scores) >= 2:
    U_stat, p_mw = mannwhitneyu(overlap_scores, non_overlap_scores, alternative='greater')
    n1, n2 = len(overlap_scores), len(non_overlap_scores)
    rank_biserial = 1 - (2 * U_stat) / (n1 * n2)

    print(f"\n  Overlap group      : n={n1}, median={overlap_scores.median():.1f}, mean={overlap_scores.mean():.2f}")
    print(f"  Non-overlap group  : n={n2}, median={non_overlap_scores.median():.1f}, mean={non_overlap_scores.mean():.2f}")
    print(f"  U-statistic        : {U_stat:.1f}")
    print(f"  p-value (one-tail) : {p_mw:.6f}  {'*** SIGNIFICANT' if p_mw < 0.05 else '(not significant)'}")
    print(f"  Rank-biserial r    : {rank_biserial:.4f}  ({'large' if abs(rank_biserial) > 0.5 else 'medium' if abs(rank_biserial) > 0.3 else 'small'} effect)")

    print("\n  📊 Business Interpretation:")
    if p_mw < 0.05:
        print("     Businesses that completed BOTH surveys score significantly HIGHER")
        print("     on the diagnostic — suggesting more pricing-aware businesses also")
        print("     chose to benchmark their rates (self-selection effect).")
    else:
        print("     No significant difference — overlap participation is not linked")
        print("     to diagnostic score level.")
else:
    U_stat, p_mw, rank_biserial = np.nan, np.nan, np.nan
    print("  ⚠ Insufficient data for Mann-Whitney U test.")

print()

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  TEST 4: Spearman Correlation — Years in Business vs Diagnostic Score   ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
print("─" * 72)
print("  TEST 4: Spearman Correlation — Years in Business vs Diagnostic Score")
print("─" * 72)

years_col = _col(df_diag_clean, 'years_in_operation')
score_col_d = _col(df_diag_clean, 'diagnostic_total_score')

spear_df = df_diag_clean[[years_col, score_col_d]].dropna()

if len(spear_df) >= 10:
    rho_spear, p_spear = spearmanr(spear_df[years_col], spear_df[score_col_d])

    print(f"\n  n (pairs)     : {len(spear_df)}")
    print(f"  Spearman ρ    : {rho_spear:.4f}")
    print(f"  p-value       : {p_spear:.6f}  {'*** SIGNIFICANT' if p_spear < 0.05 else '(not significant)'}")

    print("\n  📊 Interpretation:")
    if abs(rho_spear) < 0.1:
        print("     Negligible monotonic relationship between experience and diagnostic score.")
    elif rho_spear > 0:
        print(f"     Positive correlation (ρ={rho_spear:.3f}): more experienced businesses")
        print("     tend to have somewhat higher diagnostic scores.")
    else:
        print(f"     Negative correlation (ρ={rho_spear:.3f}): experience alone does not")
        print("     guarantee better pricing practices.")
else:
    rho_spear, p_spear = np.nan, np.nan
    print("  ⚠ Insufficient data for Spearman correlation.")

# --- Visual 24: Years vs Score Scatter with LOWESS -------------------------
print("\n  ➤ Generating Visual 24: Years vs Diagnostic Score (LOWESS) …")

tier_col_d = _col(df_diag_clean, 'pricing_maturity_tier')
plot_df_24 = df_diag_clean[[years_col, score_col_d, tier_col_d]].dropna()

fig, ax = plt.subplots(figsize=(12, 8))

palette_24 = {
    'Pricing Blind Spots': '#e74c3c',
    'Partial Pricing System': '#f39c12',
    'Strong Pricing System': '#27ae60',
}
# Scatter by tier
for tier_name, color in palette_24.items():
    mask = plot_df_24[tier_col_d] == tier_name
    if mask.any():
        ax.scatter(plot_df_24.loc[mask, years_col] + np.random.normal(0, 0.2, mask.sum()),
                   plot_df_24.loc[mask, score_col_d] + np.random.normal(0, 0.3, mask.sum()),
                   c=color, label=tier_name, alpha=0.55, edgecolors='white', s=50, linewidth=0.5)

# LOWESS trend line
try:
    sns.regplot(x=years_col, y=score_col_d, data=plot_df_24, lowess=True,
                scatter=False, color='#2c3e50', line_kws={'linewidth': 2.5, 'linestyle': '--'},
                ax=ax)
except Exception:
    pass  # graceful fallback if lowess fails

ax.set_xlabel('Years in Operation', fontsize=13)
ax.set_ylabel('Diagnostic Total Score', fontsize=13)
ax.set_title('Experience vs Pricing Diagnostic Score\n(with LOWESS Trend)',
             fontsize=16, fontweight='bold', pad=15)
ax.legend(title='Maturity Tier', fontsize=10, title_fontsize=11)

if not np.isnan(rho_spear):
    ax.annotate(f"Spearman ρ = {rho_spear:.3f}\np = {p_spear:.4f}",
                xy=(0.02, 0.96), xycoords='axes fraction', fontsize=11,
                va='top', ha='left',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow',
                          edgecolor='gray', alpha=0.9))

ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_DIR / '24_years_vs_score.png', dpi=200, bbox_inches='tight')
plt.close()
print("  ✅ Saved: 24_years_vs_score.png")
print()

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  TEST 5: Kruskal-Wallis — Diagnostic Score by Region                    ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
print("─" * 72)
print("  TEST 5: Kruskal-Wallis — Diagnostic Score by Region")
print("─" * 72)

region_col_d = _col(df_diag_clean, 'region')
kw_df = df_diag_clean[[region_col_d, score_col_d]].dropna()
kw_groups = {name: grp[score_col_d].values for name, grp in kw_df.groupby(region_col_d) if len(grp) >= 3}

if len(kw_groups) >= 2:
    H_stat, p_kw = kruskal(*kw_groups.values())

    print(f"\n  Regions tested : {len(kw_groups)}")
    for rname, rvals in kw_groups.items():
        print(f"    • {rname:25s}  median={np.median(rvals):.1f}  n={len(rvals)}")
    print(f"\n  H-statistic    : {H_stat:.4f}")
    print(f"  p-value        : {p_kw:.6f}  {'*** SIGNIFICANT' if p_kw < 0.05 else '(not significant)'}")

    # Dunn's post-hoc
    if p_kw < 0.05 and HAS_POSTHOCS:
        print("\n  ➤ Dunn's Post-Hoc Test (Bonferroni-corrected):")
        try:
            dunn_result = sp.posthoc_dunn(kw_df, val_col=score_col_d, group_col=region_col_d, p_adjust='bonferroni')
            print(dunn_result.round(4).to_string())
        except Exception as e:
            print(f"  ⚠ Dunn's test failed: {e}")
    elif p_kw < 0.05:
        print("  ⚠ scikit-posthocs not available — Dunn's test skipped.")
else:
    H_stat, p_kw = np.nan, np.nan
    print("  ⚠ Insufficient regional groups for Kruskal-Wallis.")

# --- Visual 25: Score by Region Box + Strip --------------------------------
print("\n  ➤ Generating Visual 25: Score by Region (Box + Strip) …")

fig, ax = plt.subplots(figsize=(12, 7))
order_25 = kw_df.groupby(region_col_d)[score_col_d].median().sort_values(ascending=False).index.tolist()

sns.boxplot(x=region_col_d, y=score_col_d, data=kw_df, order=order_25,
            palette='Set2', width=0.6, linewidth=1.2, fliersize=0, ax=ax)
sns.stripplot(x=region_col_d, y=score_col_d, data=kw_df, order=order_25,
              color='#2c3e50', alpha=0.4, jitter=True, size=4, ax=ax)

# Mean markers
means_25 = kw_df.groupby(region_col_d)[score_col_d].mean()
for i, reg in enumerate(order_25):
    if reg in means_25.index:
        ax.scatter(i, means_25[reg], color='red', marker='D', s=80, zorder=5, edgecolors='white', linewidth=1)

ax.set_xlabel('Region', fontsize=13)
ax.set_ylabel('Diagnostic Total Score', fontsize=13)
ax.set_title('Diagnostic Score Distribution by Region\n(with mean ◆ markers)',
             fontsize=16, fontweight='bold', pad=15)
ax.tick_params(axis='x', rotation=30)
ax.grid(axis='y', alpha=0.3)

if not np.isnan(H_stat):
    ax.text(0.98, 0.98, f"Kruskal-Wallis H = {H_stat:.2f}\np = {p_kw:.4f}",
            transform=ax.transAxes, ha='right', va='top', fontsize=10,
            bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow', edgecolor='gray'))

plt.tight_layout()
plt.savefig(OUT_DIR / '25_score_by_region.png', dpi=200, bbox_inches='tight')
plt.close()
print("  ✅ Saved: 25_score_by_region.png")
print()

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  TEST 6: Ordinal Logistic Regression — Predictors of Pricing Maturity   ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
print("─" * 72)
print("  TEST 6: Ordinal Logistic Regression — Predictors of Maturity Tier")
print("─" * 72)

ordinal_success = False
odds_ratio_df = None

try:
    olr_df = df_diag_clean.copy()

    # Encode dependent variable as ordinal integer
    tier_map = {'Pricing Blind Spots': 0, 'Partial Pricing System': 1, 'Strong Pricing System': 2}
    tier_col_olr = _col(olr_df, 'pricing_maturity_tier')
    olr_df['tier_ordinal'] = olr_df[tier_col_olr].map(tier_map)

    # Independent variables
    years_col_olr = _col(olr_df, 'years_in_operation')
    mc_col_olr = _col(olr_df, 'machine_count')
    bs_col_olr = _col(olr_df, 'business_size')
    rev_col_olr = _col(olr_df, 'revenue_segment')
    cust_col_olr = _col(olr_df, 'customer_focus_segment')

    # Dummy-encode categoricals
    olr_df = pd.get_dummies(olr_df, columns=[bs_col_olr], prefix='bs', drop_first=False, dtype=float)
    olr_df = pd.get_dummies(olr_df, columns=[rev_col_olr], prefix='rev', drop_first=False, dtype=float)
    olr_df = pd.get_dummies(olr_df, columns=[cust_col_olr], prefix='cust', drop_first=False, dtype=float)

    # Build predictor list — drop reference categories
    bs_dummies = [c for c in olr_df.columns if c.startswith('bs_') and 'Solo' not in c]
    rev_dummies = [c for c in olr_df.columns if c.startswith('rev_') and 'Under' not in c and '100k' not in c]
    cust_dummies = [c for c in olr_df.columns if c.startswith('cust_') and 'Mixed' not in c]

    predictor_cols = [years_col_olr, mc_col_olr] + bs_dummies + rev_dummies + cust_dummies
    predictor_cols = [c for c in predictor_cols if c in olr_df.columns]

    # Drop rows with NaN in any predictor or dependent
    model_df = olr_df[['tier_ordinal'] + predictor_cols].dropna()

    if len(model_df) < 50:
        raise ValueError(f"Only {len(model_df)} complete cases — too few for ordinal regression.")

    y = model_df['tier_ordinal'].astype(int)
    X = model_df[predictor_cols].astype(float)

    print(f"\n  Complete cases : {len(model_df)}")
    print(f"  Predictors     : {len(predictor_cols)}")
    print(f"  Dependent var  : tier_ordinal (0=Blind Spots, 1=Partial, 2=Strong)")

    # Fit ordinal logistic model
    ord_model = OrderedModel(y, X, distr='logit')
    ord_result = ord_model.fit(method='bfgs', disp=False, maxiter=500)

    print("\n  ── Model Summary ──")
    print(ord_result.summary())

    # Extract results
    coef_names = predictor_cols
    coefs = ord_result.params[:len(coef_names)]
    pvals = ord_result.pvalues[:len(coef_names)]

    # Confidence intervals
    try:
        ci = ord_result.conf_int()[:len(coef_names)]
        ci_low = ci[:, 0] if isinstance(ci, np.ndarray) else ci.iloc[:, 0].values
        ci_high = ci[:, 1] if isinstance(ci, np.ndarray) else ci.iloc[:, 1].values
    except Exception:
        se = ord_result.bse[:len(coef_names)]
        ci_low = coefs - 1.96 * se
        ci_high = coefs + 1.96 * se

    odds_ratio_df = pd.DataFrame({
        'Predictor': coef_names,
        'Coefficient': coefs,
        'Odds_Ratio': np.exp(coefs),
        'OR_CI_Low': np.exp(ci_low),
        'OR_CI_High': np.exp(ci_high),
        'p_value': pvals,
        'Significant': ['✔' if p < 0.05 else '' for p in pvals]
    })

    print("\n  ── Odds Ratios ──")
    print(odds_ratio_df.to_string(index=False, float_format='{:.4f}'.format))
    ordinal_success = True

    if tier_col_ov in df_overlap.columns and rate_col_ov in df_overlap.columns:
        tier_rates = df_overlap.groupby(tier_col_ov)[rate_col_ov].mean()
        rate_blind = tier_rates.get('Pricing Blind Spots', 0)
        rate_strong = tier_rates.get('Strong Pricing System', 0)
        gap = rate_strong - rate_blind
    else:
        rate_blind, rate_strong, gap = 0, 0, 0

except Exception as e:
    print(f"\n  ⚠ Ordinal logistic regression failed: {e}")
    print("    (Common causes: perfect separation, sparse categories, convergence issues)")
    print("    Skipping forest plot (Visual 26).")

# --- Visual 26: Forest Plot (Odds Ratios) ----------------------------------
if ordinal_success and odds_ratio_df is not None and len(odds_ratio_df) > 0:
    print("\n  ➤ Generating Visual 26: Odds Ratio Forest Plot …")

    fig, ax = plt.subplots(figsize=(12, max(10, len(odds_ratio_df) * 0.7)))

    y_pos = np.arange(len(odds_ratio_df))
    ors = odds_ratio_df['Odds_Ratio'].values
    ci_lo = odds_ratio_df['OR_CI_Low'].values
    ci_hi = odds_ratio_df['OR_CI_High'].values
    p_vals_forest = odds_ratio_df['p_value'].values
    labels = odds_ratio_df['Predictor'].values

    # Colours: significant vs not
    colours = ['#e74c3c' if p < 0.05 else '#95a5a6' for p in p_vals_forest]

    # Horizontal bars (error bars)
    for i in range(len(odds_ratio_df)):
        ax.plot([ci_lo[i], ci_hi[i]], [y_pos[i], y_pos[i]], color=colours[i],
                linewidth=2.5, solid_capstyle='round')
        ax.scatter(ors[i], y_pos[i], color=colours[i], s=100, zorder=5,
                   edgecolors='white', linewidth=1)

    # Reference line at OR = 1
    ax.axvline(x=1.0, color='#2c3e50', linestyle='--', linewidth=1.5, alpha=0.7, label='OR = 1 (no effect)')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=11)
    ax.set_xlabel('Odds Ratio (95% CI)', fontsize=13)
    ax.set_title('Ordinal Logistic Regression — Predictors of Pricing Maturity\n(Forest Plot of Odds Ratios)',
                 fontsize=16, fontweight='bold', pad=15)
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='#e74c3c', label='Significant (p < 0.05)',
               markerfacecolor='#e74c3c', markersize=10, linestyle='None'),
        Line2D([0], [0], marker='o', color='#95a5a6', label='Not significant',
               markerfacecolor='#95a5a6', markersize=10, linestyle='None'),
        Line2D([0], [0], color='#2c3e50', linestyle='--', linewidth=1.5, label='OR = 1 (null)'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=10)

    plt.tight_layout()
    plt.savefig(OUT_DIR / '26_odds_ratio_forest.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("  ✅ Saved: 26_odds_ratio_forest.png")

print()
print("=" * 72)
print("  PHASE 7 COMPLETE — All 6 hypothesis tests executed.")
print("=" * 72)
print()


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 8 — EXECUTIVE SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════

print("\n")
print("=" * 72)
print("█" * 72)
print("██                                                              ██")
print("██       PHASE 8: EXECUTIVE SUMMARY                             ██")
print("██       Key findings, insights & recommendations               ██")
print("██                                                              ██")
print("█" * 72)
print("=" * 72)
print()

# ── Gather statistics for the summary ──────────────────────────────────────

raw_diag = 734
raw_bench = 224
clean_diag = len(df_diag_clean)
clean_bench = len(df_bench_clean)
n_overlap = len(df_overlap)
n_diag_only = len(df_diag_only)

total_raw = raw_diag + raw_bench
total_clean = clean_diag + clean_bench
excluded_total = total_raw - total_clean
retention_rate = (total_clean / total_raw * 100) if total_raw > 0 else 0.0

# Audit log counts
if df_audit is not None and len(df_audit) > 0:
    reason_col = [c for c in df_audit.columns if 'reason' in c.lower() or 'action' in c.lower()]
    if reason_col:
        audit_reasons = df_audit[reason_col[0]].str.lower()
        n_dupes = audit_reasons.str.contains('duplic', na=False).sum()
        n_partials = audit_reasons.str.contains('partial|incomplet', na=False).sum()
        n_test = audit_reasons.str.contains('test|spam|bot', na=False).sum()
    else:
        n_dupes, n_partials, n_test = 0, 0, 0
else:
    n_dupes, n_partials, n_test = 0, 0, 0

# Maturity tier breakdown
tier_col_sum = _col(df_diag_clean, 'pricing_maturity_tier')
tier_counts = df_diag_clean[tier_col_sum].value_counts(normalize=True) * 100
pct_blind = tier_counts.get('Pricing Blind Spots', 0)
pct_partial = tier_counts.get('Partial Pricing System', 0)
pct_strong = tier_counts.get('Strong Pricing System', 0)

# Business size breakdown
bs_col_sum = _col(df_diag_clean, 'business_size')
bs_top = df_diag_clean[bs_col_sum].value_counts().head(3)

# Revenue distribution
rev_col_sum = _col(df_diag_clean, 'revenue_segment')
rev_top = df_diag_clean[rev_col_sum].value_counts().head(1)

# NLP theme data
theme_col = _col(df_diag_clean, 'comment_theme')
if theme_col in df_diag_clean.columns:
    theme_counts = df_diag_clean[theme_col].value_counts().head(3)
else:
    theme_counts = pd.Series(dtype='object')

sent_col = _col(df_diag_clean, 'comment_sentiment')
if sent_col in df_diag_clean.columns:
    sent_dist = df_diag_clean[sent_col].value_counts()
else:
    sent_dist = pd.Series(dtype='object')

# Mean hourly rates by tier (from overlap)
if tier_col_ov in df_overlap.columns and rate_col_ov in df_overlap.columns:
    tier_rates = df_overlap.groupby(tier_col_ov)[rate_col_ov].mean()
else:
    tier_rates = pd.Series(dtype='float64')

# ── Print the executive summary ───────────────────────────────────────────

W = 68  # inner width

print("╔" + "═" * W + "╗")
print("║" + "EXECUTIVE SUMMARY — KEY FINDINGS & RECOMMENDATIONS".center(W) + "║")
print("╚" + "═" * W + "╝")
print()

# Section 1 — Data Quality
print("┌" + "─" * W + "┐")
print("│" + "  § 1  DATA QUALITY & CLEANING SUMMARY".ljust(W) + "│")
print("└" + "─" * W + "┘")
print(f"  • Raw rows processed    : Diagnostic = {raw_diag}, Benchmark = {raw_bench}")
print(f"  • Clean rows produced   : Diagnostic = {clean_diag}, Benchmark = {clean_bench}, Overlap = {n_overlap}")
print(f"  • Rows excluded         : {excluded_total} total (duplicates: {n_dupes}, partials: {n_partials}, test/spam: {n_test})")
print(f"  • Data quality score    : {retention_rate:.1f}% retention rate")
print()

# Section 2 — Segmentation
print("┌" + "─" * W + "┐")
print("│" + "  § 2  SEGMENTATION OVERVIEW".ljust(W) + "│")
print("└" + "─" * W + "┘")
print(f"  • Pricing Maturity Breakdown:")
print(f"      Pricing Blind Spots : {pct_blind:.1f}%")
print(f"      Partial Systems     : {pct_partial:.1f}%")
print(f"      Strong Systems      : {pct_strong:.1f}%")
print(f"  • Business Size (top 3):")
for sz, ct in bs_top.items():
    print(f"      {sz:25s} : {ct} ({ct/clean_diag*100:.1f}%)")
if len(rev_top) > 0:
    print(f"  • Most common revenue segment : {rev_top.index[0]} ({rev_top.iloc[0]} respondents)")
print()

# Section 3 — NLP Insights
print("┌" + "─" * W + "┐")
print("│" + "  § 3  NLP INSIGHTS".ljust(W) + "│")
print("└" + "─" * W + "┘")
if len(theme_counts) > 0:
    print("  • Top comment themes:")
    for i, (thm, cnt) in enumerate(theme_counts.items(), 1):
        print(f"      {i}. {thm:30s} — {cnt} mentions")
else:
    print("  • No theme data available.")

if len(sent_dist) > 0:
    print("  • Sentiment distribution:")
    for s_name, s_cnt in sent_dist.items():
        print(f"      {s_name:20s} : {s_cnt} ({s_cnt/sent_dist.sum()*100:.1f}%)")

    # Which maturity tier has most negative sentiment?
    if sent_col in df_diag_clean.columns and tier_col_sum in df_diag_clean.columns:
        neg_mask = df_diag_clean[sent_col].str.lower().isin(['negative', 'anxious', 'frustrated'])
        if neg_mask.any():
            neg_tier = df_diag_clean.loc[neg_mask, tier_col_sum].value_counts()
            if len(neg_tier) > 0:
                print(f"  • Key finding: '{neg_tier.index[0]}' tier has the MOST negative/anxious sentiment")
                print(f"    ({neg_tier.iloc[0]} negative comments, {neg_tier.iloc[0]/neg_mask.sum()*100:.0f}% of all negative)")
else:
    print("  • No sentiment data available.")
print()

# Section 4 — Statistical Findings
print("┌" + "─" * W + "┐")
print("│" + "  § 4  STATISTICAL FINDINGS SUMMARY".ljust(W) + "│")
print("└" + "─" * W + "┘")

stat_results = []

# Test 1
sig1 = "Significant" if (not np.isnan(p_anova) and p_anova < 0.05) else "Not Significant"
metric1 = f"F={F_stat:.2f}, η²={eta_sq:.3f}" if not np.isnan(F_stat) else "N/A"
stat_results.append(("ANOVA: Rate by Maturity", sig1, metric1, "Maturity tier predicts hourly rate"))

# Test 2
sig2 = "Significant" if (not np.isnan(p_chi) and p_chi < 0.05) else "Not Significant"
metric2 = f"χ²={chi2:.2f}, V={cramers_v:.3f}" if not np.isnan(chi2) else "N/A"
stat_results.append(("Chi²: Confidence × Profitability", sig2, metric2, "Confidence and profitability are linked"))

# Test 3
sig3 = "Significant" if (not np.isnan(p_mw) and p_mw < 0.05) else "Not Significant"
metric3 = f"U={U_stat:.0f}, r={rank_biserial:.3f}" if not np.isnan(U_stat) else "N/A"
stat_results.append(("Mann-Whitney: Overlap vs Rest", sig3, metric3, "Overlap respondents score higher"))

# Test 4
sig4 = "Significant" if (not np.isnan(p_spear) and p_spear < 0.05) else "Not Significant"
metric4 = f"ρ={rho_spear:.3f}" if not np.isnan(rho_spear) else "N/A"
stat_results.append(("Spearman: Years × Score", sig4, metric4, "Experience relates to pricing maturity"))

# Test 5
sig5 = "Significant" if (not np.isnan(p_kw) and p_kw < 0.05) else "Not Significant"
metric5 = f"H={H_stat:.2f}" if not np.isnan(H_stat) else "N/A"
stat_results.append(("Kruskal-Wallis: Score by Region", sig5, metric5, "Regional differences in pricing"))

# Test 6
if ordinal_success:
    n_sig_predictors = (odds_ratio_df['p_value'] < 0.05).sum() if odds_ratio_df is not None else 0
    stat_results.append(("Ordinal LR: Maturity Predictors", f"{n_sig_predictors} sig. predictors",
                         f"n={len(model_df)}", "Multi-factor model of maturity"))
else:
    stat_results.append(("Ordinal LR: Maturity Predictors", "Did not converge", "N/A", "Skipped due to convergence"))

print(f"  {'Test':<38s} {'Result':<20s} {'Metric':<25s} {'Interpretation'}")
print(f"  {'─'*38} {'─'*20} {'─'*25} {'─'*35}")
for tname, tres, tmet, tinterp in stat_results:
    print(f"  {tname:<38s} {tres:<20s} {tmet:<25s} {tinterp}")
print()

# Section 5 — Recommendations
print("┌" + "─" * W + "┐")
print("│" + "  § 5  TOP 5 ACTIONABLE RECOMMENDATIONS".ljust(W) + "│")
print("└" + "─" * W + "┘")

# Build data-driven recommendation text
rate_blind = tier_rates.get('Pricing Blind Spots', 0) if len(tier_rates) > 0 else 0
rate_strong = tier_rates.get('Strong Systems', 0) if len(tier_rates) > 0 else 0
rate_gap = rate_strong - rate_blind

rec_texts = [
    (f"FORMALIZE YOUR PRICING SYSTEM. Businesses with Pricing Blind Spots "
     f"charge ~${rate_blind:.0f}/hr vs ${rate_strong:.0f}/hr for Strong Systems "
     f"(a ${rate_gap:.0f} gap). Implementing structured pricing alone could "
     f"unlock significant revenue."),

    (f"ADDRESS UNDERCHARGING ANXIETY. {pct_blind:.0f}% of respondents fall "
     f"into the 'Pricing Blind Spots' tier. Provide these businesses with "
     f"competitive rate benchmarks, minimum-job-charge templates, and "
     f"value-based pricing worksheets."),

    (f"TARGET REGIONAL DISPARITIES. Diagnostic scores vary by region. "
     f"Develop region-specific pricing guides and localized training "
     f"programs to bring underperforming regions up to parity."),

    (f"LEVERAGE EXPERIENCE — BUT DON'T ASSUME IT. "
     f"Spearman ρ = {rho_spear:.3f} suggests experience has only a "
     f"{'modest' if abs(rho_spear) < 0.3 else 'moderate'} correlation with "
     f"pricing maturity. Even long-established businesses need structured "
     f"pricing education.") if not np.isnan(rho_spear) else
    ("LEVERAGE EXPERIENCE. Even long-established businesses benefit from "
     "formal pricing training programs."),

    (f"BUILD CONFIDENCE TO BUILD PROFITS. Chi-square analysis shows "
     f"pricing confidence is {'significantly' if not np.isnan(p_chi) and p_chi < 0.05 else 'potentially'} "
     f"linked to perceived profitability (V={cramers_v:.3f}). "
     f"Invest in pricing confidence workshops, mentorship programs, and "
     f"community benchmarking tools to break the undercharging cycle.")
]

for i, rec in enumerate(rec_texts, 1):
    print(f"\n  {i}. {rec}")

print()

# ── Final Banner ──────────────────────────────────────────────────────────

# ── NEW Visual 27: Statistical Summary Table ──────────────────────────────
print("\n" + "─" * 72)
print("  Generating Visual 27: Statistical Summary Table")
print("─" * 72)

try:
    summary_rows = []
    # Test 1: ANOVA
    summary_rows.append({
        'Test': 'One-Way ANOVA',
        'Hypothesis': 'Hourly Rate ~ Maturity Tier',
        'Statistic': f'F = {F_stat:.3f}' if not np.isnan(F_stat) else 'N/A',
        'p-value': f'{p_anova:.4f}' if not np.isnan(p_anova) else 'N/A',
        'Effect Size': f'η² = {eta_sq:.4f}' if not np.isnan(eta_sq) else 'N/A',
        'Significant': '✅ Yes' if not np.isnan(p_anova) and p_anova < 0.05 else '❌ No',
    })
    # Test 2: Chi-Square
    summary_rows.append({
        'Test': 'Chi-Square',
        'Hypothesis': 'Confidence ~ Profitability',
        'Statistic': f'χ² = {chi2:.3f}' if not np.isnan(chi2) else 'N/A',
        'p-value': f'{p_chi:.4f}' if not np.isnan(p_chi) else 'N/A',
        'Effect Size': f"V = {cramers_v:.4f}" if not np.isnan(cramers_v) else 'N/A',
        'Significant': '✅ Yes' if not np.isnan(p_chi) and p_chi < 0.05 else '❌ No',
    })
    # Test 3: Mann-Whitney U
    summary_rows.append({
        'Test': 'Mann-Whitney U',
        'Hypothesis': 'Score: Setup-fee vs No-fee',
        'Statistic': f'U = {U_stat:.1f}' if not np.isnan(U_stat) else 'N/A',
        'p-value': f'{p_mw:.4f}' if not np.isnan(p_mw) else 'N/A',
        'Effect Size': f'r = {rank_biserial:.4f}' if not np.isnan(rank_biserial) else 'N/A',
        'Significant': '✅ Yes' if not np.isnan(p_mw) and p_mw < 0.05 else '❌ No',
    })
    # Test 4: Spearman
    summary_rows.append({
        'Test': 'Spearman ρ',
        'Hypothesis': 'Experience ~ Score',
        'Statistic': f'ρ = {rho_spear:.3f}' if not np.isnan(rho_spear) else 'N/A',
        'p-value': f'{p_spear:.4f}' if not np.isnan(p_spear) else 'N/A',
        'Effect Size': f'ρ = {rho_spear:.4f}' if not np.isnan(rho_spear) else 'N/A',
        'Significant': '✅ Yes' if not np.isnan(p_spear) and p_spear < 0.05 else '❌ No',
    })
    # Test 5: Kruskal-Wallis
    summary_rows.append({
        'Test': 'Kruskal-Wallis',
        'Hypothesis': 'Score ~ Region',
        'Statistic': f'H = {H_stat:.3f}' if not np.isnan(H_stat) else 'N/A',
        'p-value': f'{p_kw:.4f}' if not np.isnan(p_kw) else 'N/A',
        'Effect Size': '—',
        'Significant': '✅ Yes' if not np.isnan(p_kw) and p_kw < 0.05 else '❌ No',
    })

    summary_df = pd.DataFrame(summary_rows)

    fig, ax = plt.subplots(figsize=(16, 4))
    ax.axis('off')
    ax.set_title("Statistical Test Results Summary", fontsize=18, fontweight='bold', pad=20, loc='left')

    table = ax.table(
        cellText=summary_df.values,
        colLabels=summary_df.columns,
        loc='center',
        cellLoc='center',
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.0, 1.8)

    # Style header
    for j, col in enumerate(summary_df.columns):
        cell = table[0, j]
        cell.set_facecolor('#2c3e50')
        cell.set_text_props(color='white', fontweight='bold')

    # Alternate row colors
    for i in range(len(summary_df)):
        for j in range(len(summary_df.columns)):
            cell = table[i + 1, j]
            cell.set_facecolor('#f8f9fa' if i % 2 == 0 else '#ffffff')
            cell.set_edgecolor('#dee2e6')

    plt.tight_layout()
    plt.savefig(OUT_DIR / '27_statistical_summary_table.png', bbox_inches='tight')
    plt.close()
    print("  ✅ Saved 27_statistical_summary_table.png")
except Exception as e:
    print(f"  ⚠ Could not generate summary table: {e}")

# ── NEW Visual 28: Word Cloud from Open-Text Responses ───────────────────
print("\n" + "─" * 72)
print("  Generating Visual 28: Open-Text Word Cloud")
print("─" * 72)

try:
    from wordcloud import WordCloud
    HAS_WORDCLOUD = True
except ImportError:
    HAS_WORDCLOUD = False
    print("  ⚠ wordcloud not installed. Install with: pip install wordcloud")

if HAS_WORDCLOUD:
    try:
        # Collect all open-text columns from both datasets
        text_cols_diag = [c for c in df_diag_clean.columns if 'comment' in c.lower() or 'text' in c.lower() or 'open' in c.lower() or 'describe' in c.lower() or 'challenge' in c.lower()]
        text_cols_bench = [c for c in df_bench_clean.columns if 'comment' in c.lower() or 'text' in c.lower() or 'open' in c.lower() or 'describe' in c.lower() or 'challenge' in c.lower()]

        all_text = []
        for col in text_cols_diag:
            all_text.extend(df_diag_clean[col].dropna().astype(str).tolist())
        for col in text_cols_bench:
            all_text.extend(df_bench_clean[col].dropna().astype(str).tolist())

        combined_text = ' '.join(all_text)

        if len(combined_text.strip()) > 50:
            # Custom stopwords for survey context
            custom_stops = {'nan', 'none', 'na', 'n/a', 'null', 'the', 'and', 'for',
                           'that', 'this', 'with', 'not', 'are', 'was', 'but', 'have',
                           'has', 'had', 'been', 'will', 'would', 'could', 'should',
                           'just', 'more', 'very', 'also', 'really', 'think', 'know',
                           'like', 'don', 'didn', 'doesn', 'isn', 'can', 'our', 'from'}

            wc = WordCloud(
                width=1600, height=800,
                background_color='white',
                colormap='viridis',
                max_words=150,
                stopwords=custom_stops,
                contour_width=2,
                contour_color='#2c3e50',
                min_font_size=10,
                max_font_size=120,
                relative_scaling=0.5,
            ).generate(combined_text)

            fig, ax = plt.subplots(figsize=(16, 8))
            ax.imshow(wc, interpolation='bilinear')
            ax.axis('off')
            ax.set_title("What Embroidery Businesses Say About Pricing\n(Open-Text Response Word Cloud)",
                        fontsize=18, fontweight='bold', pad=20)
            plt.tight_layout()
            plt.savefig(OUT_DIR / '28_word_cloud.png', bbox_inches='tight')
            plt.close()
            print("  ✅ Saved 28_word_cloud.png")
        else:
            print("  ⚠ Not enough open-text data for word cloud")
    except Exception as e:
        print(f"  ⚠ Word cloud generation failed: {e}")


print()
print("━" * 72)
print("  Analysis complete. All visualizations saved to outputs/")
print("  Cleaned datasets saved to cleaned/")
print("━" * 72)
print()
print("╔" + "═" * W + "╗")
print("║" + "Thank you for using the Embroidery & Decoration Industry".center(W) + "║")
print("║" + "Pricing Analytics Suite — v4.0".center(W) + "║")
print("╚" + "═" * W + "╝")
print()
