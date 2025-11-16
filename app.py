# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

st.set_page_config(layout="wide", page_title="Telco Churn Dashboard", initial_sidebar_state="expanded")
sns.set_style("whitegrid")

DATA_PATHS = [
     "clean_df.csv",
    "df_clean.csv",
    "telco_customer_churn_clean.csv"
]

@st.cache_data
def load_data():
    # Try multiple filenames commonly used
    for p in DATA_PATHS:
        if Path(p).exists():
            df = pd.read_csv(p)
            st.session_state['_data_source'] = p
            break
    else:
        st.error("Can't find cleaned CSV. Place `df_clean.csv` (or `telco_customer_churn_clean.csv`) in the app folder.")
        st.stop()
    # Basic cleaning & standardization
    df = df.copy()

    # Standardize churn to boolean
    if df['churn'].dtype == object:
        df['churn'] = df['churn'].astype(str).str.strip().str.lower().map({"yes": True, "no": False, "true": True, "false": False})
    df['churn'] = df['churn'].astype(bool)

    # Senior citizen: ensure numeric 0/1
    if 'senior_citizen' in df.columns:
        df['senior_citizen'] = pd.to_numeric(df['senior_citizen'], errors='coerce').fillna(0).astype(int)

    # Ensure numeric charges
    for col in ['monthly_charges', 'total_charges', 'tenure']:
        if col in df.columns:
            # unify names to lower-case versions used later
            pass

    # Normalize column names for consistent code below
    df.columns = [c.strip() for c in df.columns]
    # Create consistent lower-case alias columns
    # Map likely column names to canonical ones
    col_map = {}
    def find(col_options):
        for c in col_options:
            if c in df.columns:
                return c
        return None
    col_map['customer_id'] = find(['customerID','Customer Id','CustomerId','customer_id','customer id'])
    col_map['churn'] = find(['Churn','churn'])
    col_map['tenure'] = find(['tenure','Tenure'])
    col_map['senior'] = find(['SeniorCitizen','Senior Citizen','Senior_Citizen','SeniorCitizen'])
    col_map['monthly'] = find(['MonthlyCharges','monthly_charges','Monthly Charges','Monthly Charge'])
    col_map['total'] = find(['TotalCharges','total_charges','Total Charges'])
    col_map['contract'] = find(['Contract','contract'])
    col_map['gender'] = find(['gender','Gender'])
    col_map['payment'] = find(['PaymentMethod','Payment Method','payment_method','payment method'])
    col_map['internet'] = find(['InternetService','Internet Service','internet_service'])
    col_map['phone'] = find(['PhoneService','Phone Service','phone_service'])
    col_map['tech'] = find(['TechSupport','Tech Support','tech_support'])
    col_map['online_backup'] = find(['OnlineBackup','Online Backup','online_backup'])
    # Create canonical columns if present
    if col_map['churn'] and col_map['churn'] != 'churn':
        df.rename(columns={col_map['churn']:'churn'}, inplace=True)
    if col_map['tenure'] and col_map['tenure'] != 'tenure':
        df.rename(columns={col_map['tenure']:'tenure'}, inplace=True)
    if col_map['senior'] and col_map['senior'] != 'senior_citizen':
        df.rename(columns={col_map['senior']:'senior_citizen'}, inplace=True)
    if col_map['monthly'] and col_map['monthly'] != 'monthly_charges':
        df.rename(columns={col_map['monthly']:'monthly_charges'}, inplace=True)
    if col_map['total'] and col_map['total'] != 'total_charges':
        df.rename(columns={col_map['total']:'total_charges'}, inplace=True)
    if col_map['contract'] and col_map['contract'] != 'contract':
        df.rename(columns={col_map['contract']:'contract'}, inplace=True)
    if col_map['gender'] and col_map['gender'] != 'gender':
        df.rename(columns={col_map['gender']:'gender'}, inplace=True)
    if col_map['payment'] and col_map['payment'] != 'payment_method':
        df.rename(columns={col_map['payment']:'payment_method'}, inplace=True)
    if col_map['internet'] and col_map['internet'] != 'internet_service':
        df.rename(columns={col_map['internet']:'internet_service'}, inplace=True)
    if col_map['phone'] and col_map['phone'] != 'phone_service':
        df.rename(columns={col_map['phone']:'phone_service'}, inplace=True)
    if col_map['tech'] and col_map['tech'] != 'tech_support':
        df.rename(columns={col_map['tech']:'tech_support'}, inplace=True)

    # ensure numeric types
    if 'monthly_charges' in df.columns:
        df['monthly_charges'] = pd.to_numeric(df['monthly_charges'], errors='coerce')
    if 'total_charges' in df.columns:
        df['total_charges'] = pd.to_numeric(df['total_charges'], errors='coerce')
    if 'tenure' in df.columns:
        df['tenure'] = pd.to_numeric(df['tenure'], errors='coerce').fillna(0).astype(int)
    # churn boolean again if renamed
    if 'churn' in df.columns:
        df['churn'] = df['churn'].astype(bool)

    # Create additional helper columns
    if 'tenure_group' not in df.columns and 'tenure' in df.columns:
        df['tenure_group'] = pd.cut(df['tenure'], bins=[-1,12,36,999], labels=['0-12','13-36','37+'])

    # churn_flag numeric (0/1) for aggregations
    df['churn_flag'] = df['churn'].astype(int)

    return df

df = load_data()

# Sidebar filters
st.sidebar.header("Filters")
# Optional filters depending on columns present
filters = {}
if 'internet_service' in df.columns:
    filters['internet_service'] = st.sidebar.multiselect("Internet Service", options=sorted(df['internet_service'].dropna().unique()), default=sorted(df['internet_service'].dropna().unique()))
if 'contract' in df.columns:
    filters['contract'] = st.sidebar.multiselect("Contract", options=sorted(df['contract'].dropna().unique()), default=sorted(df['contract'].dropna().unique()))
if 'gender' in df.columns:
    filters['gender'] = st.sidebar.multiselect("Gender", options=sorted(df['gender'].dropna().unique()), default=sorted(df['gender'].dropna().unique()))
if 'payment_method' in df.columns:
    filters['payment_method'] = st.sidebar.multiselect("Payment Method", options=sorted(df['payment_method'].dropna().unique()), default=sorted(df['payment_method'].dropna().unique()))
if 'tenure_group' in df.columns:
    # allow selecting tenure groups
    filters['tenure_group'] = st.sidebar.multiselect("Tenure group", options=list(df['tenure_group'].cat.categories) if pd.api.types.is_categorical_dtype(df['tenure_group']) else sorted(df['tenure_group'].dropna().unique()), default=list(df['tenure_group'].cat.categories) if pd.api.types.is_categorical_dtype(df['tenure_group']) else sorted(df['tenure_group'].dropna().unique()))

# Apply filters
df_filtered = df.copy()
for k,v in filters.items():
    if v:
        df_filtered = df_filtered[df_filtered[k].isin(v)]

# Top row: KPIs
st.title("Customer Churn — Interactive Dashboard")
col1, col2, col3, col4 = st.columns([1.2,1,1,1])
with col1:
    st.metric("Total customers", f"{df_filtered['customerID'].nunique() if 'customerID' in df_filtered.columns else df_filtered.shape[0]:,}")
with col2:
    churn_count = int(df_filtered['churn_flag'].sum())
    st.metric("Churned customers", f"{churn_count:,}")
with col3:
    churn_rate = df_filtered['churn_flag'].mean()*100
    st.metric("Churn rate", f"{churn_rate:.2f}%")
with col4:
    avg_tenure = df_filtered['tenure'].mean() if 'tenure' in df_filtered.columns else np.nan
    st.metric("Avg tenure (months)", f"{avg_tenure:.1f}")

st.markdown("---")

# Section: Overview charts (counts and charges)
st.header("Overview")
col1, col2 = st.columns(2)

# Left: churn counts by churn boolean (bar)
with col1:
    st.subheader("Total churn count")
    fig, ax = plt.subplots(figsize=(6,4))
    order = [False, True]
    if 'churn' in df_filtered.columns:
        sns.countplot(x='churn', data=df_filtered, order=order, palette=['#4E79A7','#E15759'], ax=ax)
        totals = df_filtered['churn'].value_counts().reindex(order).fillna(0).astype(int)
        for p, t in zip(ax.patches, totals):
            ax.annotate(f"{int(t)}", (p.get_x()+p.get_width()/2., p.get_height()), ha='center', va='bottom')
        ax.set_xlabel("Churn")
        ax.set_ylabel("Count")
        ax.set_xticklabels(['Retained','Churned'])
    st.pyplot(fig)

# Right: monthly and total charges by churn
with col2:
    st.subheader("Monthly charges & Total charges by churn")
    fig, axes = plt.subplots(2,1, figsize=(6,6))
    if 'monthly_charges' in df_filtered.columns:
        sns.barplot(x='churn', y='monthly_charges', data=df_filtered, ax=axes[0], estimator=np.mean, palette=['#4E79A7','#E15759'])
        axes[0].set_title("Avg Monthly Charges")
        axes[0].set_xlabel("")
        axes[0].set_xticklabels(['Retained','Churned'])
    if 'total_charges' in df_filtered.columns:
        sns.barplot(x='churn', y='total_charges', data=df_filtered, ax=axes[1], estimator=np.mean, palette=['#4E79A7','#E15759'])
        axes[1].set_title("Avg Total Charges")
        axes[1].set_xlabel("")
        axes[1].set_xticklabels(['Retained','Churned'])
    plt.tight_layout()
    st.pyplot(fig)

st.markdown("---")

# Section: Demographics & Tenure
st.header("Demographics & Tenure")
col1, col2 = st.columns(2)

with col1:
    st.subheader("Churn rate by Gender")
    if 'gender' in df_filtered.columns:
        gender_churn = df_filtered.groupby('gender')['churn_flag'].mean().reset_index()
        fig, ax = plt.subplots(figsize=(5,3))
        sns.barplot(x='gender', y='churn_flag', data=gender_churn, ax=ax, palette='muted')
        ax.set_ylabel("Churn rate (proportion)")
        for p in ax.patches:
            ax.annotate(f"{p.get_height()*100:.1f}%", (p.get_x()+p.get_width()/2., p.get_height()), ha='center', va='bottom')
        st.pyplot(fig)
    else:
        st.write("Gender column not found.")

with col2:
    st.subheader("Churn rate by Senior Citizen")
    if 'senior_citizen' in df_filtered.columns:
        senior_churn = df_filtered.groupby('senior_citizen')['churn_flag'].mean().reset_index()
        fig, ax = plt.subplots(figsize=(5,3))
        sns.barplot(x='senior_citizen', y='churn_flag', data=senior_churn, ax=ax, palette=['#4E79A7','#E15759'])
        ax.set_xticklabels(['Not Senior','Senior'])
        ax.set_ylabel("Churn rate (proportion)")
        for p in ax.patches:
            ax.annotate(f"{p.get_height()*100:.1f}%", (p.get_x()+p.get_width()/2., p.get_height()), ha='center', va='bottom')
        st.pyplot(fig)
    else:
        st.write("Senior citizen column not found.")

st.markdown("### Tenure distributions")
col1, col2 = st.columns(2)
with col1:
    st.subheader("Average tenure by churn")
    if 'tenure' in df_filtered.columns:
        tenure_avg = df_filtered.groupby('churn')['tenure'].mean().reset_index()
        fig, ax = plt.subplots(figsize=(5,3))
        sns.barplot(x='churn', y='tenure', data=tenure_avg, ax=ax, palette=['#4E79A7','#E15759'])
        ax.set_xticklabels(['Retained','Churned'])
        ax.set_ylabel("Avg tenure (months)")
        for p in ax.patches:
            ax.annotate(f"{p.get_height():.1f}", (p.get_x()+p.get_width()/2., p.get_height()), ha='center', va='bottom')
        st.pyplot(fig)
    else:
        st.write("Tenure column not found.")
with col2:
    st.subheader("Tenure distribution by churn")
    if 'tenure' in df_filtered.columns:
        fig, ax = plt.subplots(figsize=(5,3))
        sns.boxplot(x='churn', y='tenure', data=df_filtered, palette=['#4E79A7','#E15759'], ax=ax)
        ax.set_xticklabels(['Retained','Churned'])
        ax.set_ylabel("Tenure (months)")
        st.pyplot(fig)
    else:
        st.write("Tenure column not found.")

st.markdown("---")

# Section: Contract & Services
st.header("Service & Contract Insights")
col1, col2 = st.columns(2)
with col1:
    st.subheader("Churn rate by Contract type")
    if 'contract' in df_filtered.columns:
        contract_df = df_filtered.groupby('contract')['churn_flag'].mean().reset_index()
        fig, ax = plt.subplots(figsize=(6,3))
        sns.barplot(x='contract', y='churn_flag', data=contract_df, order=contract_df.sort_values('churn_flag',ascending=False)['contract'], palette='mako', ax=ax)
        ax.set_ylabel("Churn rate (proportion)")
        for p in ax.patches:
            ax.annotate(f"{p.get_height()*100:.1f}%", (p.get_x()+p.get_width()/2., p.get_height()), ha='center', va='bottom')
        st.pyplot(fig)
    else:
        st.write("Contract column not found.")

with col2:
    st.subheader("Churn by Tech Support & Phone Service")
    # stacked/grouped view using crosstab
    if 'tech_support' in df_filtered.columns and 'phone_service' in df_filtered.columns:
        ctab = pd.crosstab([df_filtered['contract'], df_filtered['phone_service']], df_filtered['tech_support'], values=df_filtered['churn_flag'], aggfunc='mean').fillna(0)
        # display as heatmap for quick interpretation
        fig, ax = plt.subplots(figsize=(6,3))
        sns.heatmap(ctab, annot=True, fmt=".2f", cmap='coolwarm', ax=ax)
        st.pyplot(fig)
    else:
        st.write("Required columns (tech_support, phone_service) not found for this view.")

st.markdown("---")

# Section: Charges vs Tenure scatter
st.header("Charges vs Tenure")
if 'tenure' in df_filtered.columns and 'monthly_charges' in df_filtered.columns:
    fig, ax = plt.subplots(figsize=(8,4))
    sns.scatterplot(x='tenure', y='monthly_charges', hue='churn', data=df_filtered, palette=['#4E79A7','#E15759'], alpha=0.6, ax=ax)
    ax.set_xlabel("Tenure (months)")
    ax.set_ylabel("Monthly Charges")
    st.pyplot(fig)
else:
    st.write("Tenure or Monthly Charges missing.")

st.markdown("---")

# Section: Correlation & advanced stats
st.header("Advanced Analysis")
col1, col2 = st.columns([2,1])
with col1:
    st.subheader("Correlation matrix (numeric features)")
    num_cols = df_filtered.select_dtypes(include=np.number).columns.tolist()
    if len(num_cols) > 1:
        corr = df_filtered[num_cols].corr()
        fig, ax = plt.subplots(figsize=(8,6))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap='vlag', center=0, ax=ax)
        st.pyplot(fig)
    else:
        st.write("Not enough numeric columns for correlation.")

with col2:
    st.subheader("Feature importance (simple logistic)")
    # We'll train a simple logistic regression on a few features to get coefficients
    # choose a mix of numeric and categorical features if available
    features = []
    numeric_feats = ['tenure','monthly_charges','total_charges']
    cat_feats = []
    for f in numeric_feats:
        if f in df_filtered.columns:
            features.append(f)
    for c in ['contract','internet_service','payment_method','gender','tech_support']:
        if c in df_filtered.columns:
            cat_feats.append(c)
    # build pipeline
    if features or cat_feats:
        X = df_filtered[features + cat_feats].copy()
        y = df_filtered['churn_flag']
        # simple preprocessing
        num_transform = Pipeline([('impute', SimpleImputer(strategy='median'))])
        cat_transform = Pipeline([('impute', SimpleImputer(strategy='most_frequent')), ('ohe', OneHotEncoder(handle_unknown='ignore'))])
        preproc = ColumnTransformer([('num', num_transform, features), ('cat', cat_transform, cat_feats)], remainder='drop', sparse_threshold=0)
        pipe = Pipeline([('prep', preproc), ('clf', LogisticRegression(max_iter=1000))])
        try:
            pipe.fit(X, y)
            # get feature names
            ohe = pipe.named_steps['prep'].transformers_[1][1].named_steps['ohe']
            num_names = features
            cat_names = pipe.named_steps['prep'].transformers_[1][1].named_steps['ohe'].get_feature_names_out(cat_feats).tolist() if cat_feats else []
            feature_names = num_names + cat_names
            coefs = pipe.named_steps['clf'].coef_[0]
            fi = pd.DataFrame({'feature': feature_names, 'coef': coefs})
            fi['abs'] = fi['coef'].abs()
            fi = fi.sort_values('abs', ascending=False).head(10)
            fig, ax = plt.subplots(figsize=(4,3))
            sns.barplot(x='coef', y='feature', data=fi, palette='crest', ax=ax)
            ax.set_title("Top features (logistic coef)")
            st.pyplot(fig)
        except Exception as e:
            st.write("Could not fit logistic model:", e)
    else:
        st.write("Not enough features for model-based importance.")

st.markdown("---")
st.write("Generated with Streamlit — visuals from your EDA & advanced analysis notebooks.")
