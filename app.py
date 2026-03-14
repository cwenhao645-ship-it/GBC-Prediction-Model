import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

# ==========================================
# 1. 页面配置与标题
# ==========================================
st.set_page_config(page_title="GBC Post-op SDM Predictor", layout="wide", page_icon="🔬")

st.title("Gallbladder Cancer Post-operative Risk Stratification")
st.subheader("Identifying the Risk of Synchronous Distant Metastasis (SDM) via LNR-Enhanced XGBoost")

st.markdown("""
**Clinical Context:** In routine practice, some gallbladder cancer (GBC) patients who undergo curative-intent resection are subsequently found to harbor synchronous distant metastasis (SDM) that was unrecognized during preoperative assessment. Traditional N-staging may be compromised by inadequate lymph node harvest, leading to stage migration. 
This tool utilizes the Lymph Node Ratio (LNR) to provide a quantitative, individualized risk prediction for such SDM events, serving as an adjunctive decision-support tool for postoperative management.
""")
st.divider()

# ==========================================
# 2. 加载最新模型 (包含 6 个特征)
# ==========================================
@st.cache_resource
def load_model():
    # ⚠️ 请确保您的 GitHub 仓库里已经上传了最新的 GBC_LNR_XGB_Model_Ultimate.pkl
    return joblib.load("GBC_LNR_XGB_Model_Ultimate.pkl")

model = load_model()

# ==========================================
# 3. 侧边栏：收集临床输入特征
# ==========================================
st.sidebar.header("📝 Patient Post-operative Parameters")
st.sidebar.markdown("Please input the pathology results:")

age = st.sidebar.number_input("Age at Diagnosis (Years)", min_value=18, max_value=100, value=65)
sex = st.sidebar.selectbox("Sex", ["Female", "Male"])
t_stage = st.sidebar.selectbox("Pathological T Stage", ["T1", "T2", "T3", "T4"])
grade = st.sidebar.selectbox("Histological Grade", ["Grade I (Well differentiated)", "Grade II (Moderately)", "Grade III (Poorly)", "Grade IV (Undifferentiated)"])
tumor_size = st.sidebar.number_input("Tumor Size (mm)", min_value=1, max_value=200, value=30)

st.sidebar.markdown("---")
nodes_examined = st.sidebar.number_input("Total Lymph Nodes Examined", min_value=1, max_value=100, value=6)
nodes_positive = st.sidebar.number_input("Positive Lymph Nodes", min_value=0, max_value=100, value=0)

# LNR 计算与防错机制
if nodes_positive > nodes_examined:
    st.sidebar.error("Error: Positive nodes cannot exceed examined nodes.")
    lnr = 0.0
else:
    lnr = nodes_positive / nodes_examined
st.sidebar.info(f"**Calculated Lymph Node Ratio (LNR): {lnr:.3f}**")

# ==========================================
# 4. 数据预处理与预测
# ==========================================
# 变量映射 (严格对齐模型底层代码)
sex_code = 1 if sex == "Male" else 0
t_mapping = {"T1": 1, "T2": 2, "T3": 3, "T4": 4}
grade_mapping = {
    "Grade I (Well differentiated)": 1, 
    "Grade II (Moderately)": 2, 
    "Grade III (Poorly)": 3, 
    "Grade IV (Undifferentiated)": 4
}

# 按照模型的输入特征顺序构建 DataFrame
features_core = ['Age_Numeric', 'Sex_Code', 'T_Code', 'Grade_Code', 'Tumor_Size_Num', 'LNR']
input_data = [[age, sex_code, t_mapping[t_stage], grade_mapping[grade], tumor_size, lnr]]
input_df = pd.DataFrame(input_data, columns=features_core)

# 为 SHAP 绘图准备优美的变量名
feature_display_names = [
    'Age (Years)', 'Sex', 'Pathological T Stage', 
    'Histological Grade', 'Tumor Size (mm)', 'Lymph Node Ratio (LNR)'
]

# 执行预测 (严格对应正文中预设的高敏阈值 0.374)
prob = model.predict_proba(input_df)[0, 1]
OPTIMAL_THRESHOLD = 0.374

# ==========================================
# 5. 结果展示与克制的临床建议
# ==========================================
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### 🎯 Risk Assessment")
    
    if prob >= OPTIMAL_THRESHOLD:
        st.error(f"⚠️ HIGH RISK of Synchronous Distant Metastasis")
        st.markdown(f"<h1 style='text-align: center; color: #E64B35;'>{prob:.1%}</h1>", unsafe_allow_html=True)
        st.warning(f"""
        **Clinical Recommendation:**
        The predicted risk probability ({prob:.1%}) exceeds the high-sensitivity clinical trigger threshold ({OPTIMAL_THRESHOLD:.1%}). 
        👉 **Prompt systemic imaging evaluation (e.g., whole-body PET-CT) is recommended** to rule out unrecognized distant metastasis before initiating routine adjuvant therapy.
        """)
    else:
        st.success(f"✅ LOW RISK of Synchronous Distant Metastasis")
        st.markdown(f"<h1 style='text-align: center; color: #00A087;'>{prob:.1%}</h1>", unsafe_allow_html=True)
        st.info(f"""
        **Clinical Recommendation:**
        The predicted risk probability ({prob:.1%}) is below the high-sensitivity threshold ({OPTIMAL_THRESHOLD:.1%}). 
        👉 **Routine post-operative surveillance and standard adjuvant treatments** according to current guidelines are appropriate.
        """)

# ==========================================
# 6. SHAP 瀑布图 (去除乱码，英文全名映射)
# ==========================================
with col2:
    st.markdown("### 🧠 AI Decision Explanation")
    st.markdown("*(Individual SHAP Waterfall Plot)*")
    
    # SHAP 解释器
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(input_df)
    
    # 强制将变量名替换为优美的英文全称
    shap_values.feature_names = feature_display_names
    
    # 绘制瀑布图 (防止截断)
    fig, ax = plt.subplots(figsize=(7, 5))
    shap.plots.waterfall(shap_values[0], show=False)
    plt.tight_layout()
    st.pyplot(fig)

st.divider()
st.caption("Disclaimer: This tool is developed based on the SEER database (n=2,830) for academic research and adjunctive risk stratification only. It does not establish biological causality and should not replace professional medical judgment or guideline-recommended standard of care.")
