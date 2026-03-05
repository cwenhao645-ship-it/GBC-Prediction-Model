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

st.title("Gallbladder Cancer Post-operative Staging Refinement")
st.subheader("Identifying Missed Synchronous Distant Metastasis (SDM) via LNR")

st.markdown("""
**Clinical Context:** Traditional N-staging often suffers from 'stage migration' due to inadequate lymph node harvest. This tool utilizes the Lymph Node Ratio (LNR) as a 'mathematical buffer' to identify patients who appear localized (M0) during surgery but harbor high risk of **Synchronous Distant Metastasis (SDM)**. 
*Designed as a "Rule-out" tool to guide post-operative deep imaging (e.g., PET-CT).*
""")
st.divider()

# ==========================================
# 2. 加载最新模型 (包含 6 个特征)
# ==========================================
@st.cache_resource
def load_model():
    # ⚠️ 请确保您的 GitHub 仓库里有这个最新的 pkl 模型文件！
    return joblib.load("GBC_LNR_XGB_Model_Ultimate.pkl")

model = load_model()

# ==========================================
# 3. 侧边栏：收集 6 个临床输入特征
# ==========================================
st.sidebar.header("📝 Post-operative Parameters")
st.sidebar.markdown("Please input the pathology results:")

age = st.sidebar.number_input("Age at Diagnosis", min_value=18, max_value=100, value=65)
sex = st.sidebar.selectbox("Sex", ["Female", "Male"])
t_stage = st.sidebar.selectbox("Pathological T Stage", ["T1", "T2", "T3", "T4"])
grade = st.sidebar.selectbox("Tumor Grade", ["Grade I (Well differentiated)", "Grade II (Moderately)", "Grade III (Poorly)", "Grade IV (Undifferentiated)"])
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
st.sidebar.info(f"**Calculated LNR: {lnr:.3f}**")

# ==========================================
# 4. 数据预处理与预测
# ==========================================
# 变量映射 (严格对齐您的 Python 训练代码)
sex_code = 1 if sex == "Male" else 0
t_mapping = {"T1": 1, "T2": 2, "T3": 3, "T4": 4}
grade_mapping = {
    "Grade I (Well differentiated)": 1, 
    "Grade II (Moderately)": 2, 
    "Grade III (Poorly)": 3, 
    "Grade IV (Undifferentiated)": 4
}

input_df = pd.DataFrame({
    'Age_Numeric': [age],
    'Sex_Code': [sex_code],
    'T_Code': [t_mapping[t_stage]],
    'Grade_Code': [grade_mapping[grade]],
    'Tumor_Size_Num': [tumor_size],
    'LNR': [lnr]
})

# 执行预测 (最佳阈值为 0.546)
prob = model.predict_proba(input_df)[0, 1]
OPTIMAL_THRESHOLD = 0.546

# ==========================================
# 5. 结果展示与高级临床话术
# ==========================================
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### 🎯 Risk Assessment")
    
    if prob >= OPTIMAL_THRESHOLD:
        st.error(f"⚠️ HIGH RISK of Synchronous Distant Metastasis")
        st.markdown(f"<h1 style='text-align: center; color: #E64B35;'>{prob:.1%}</h1>", unsafe_allow_html=True)
        st.warning("""
        **Clinical Recommendation (Rule-out Philosophy):**
        This patient exhibits a highly aggressive tumor biological profile despite potential initial negative exploration. 
        👉 **Immediate systemic restaging via PET-CT or high-resolution imaging is strongly advised** before initiating routine adjuvant therapy.
        """)
    else:
        st.success(f"✅ LOW RISK of Synchronous Distant Metastasis")
        st.markdown(f"<h1 style='text-align: center; color: #00A087;'>{prob:.1%}</h1>", unsafe_allow_html=True)
        st.info("""
        **Clinical Recommendation:**
        The systemic dissemination risk is relatively low. 
        👉 Routine post-operative follow-up and standard adjuvant treatments according to current guidelines are appropriate. Avoid unnecessary excessive imaging.
        """)

# ==========================================
# 6. SHAP 瀑布图 (透明化决策)
# ==========================================
with col2:
    st.markdown("### 🧠 AI Decision Explanation (SHAP)")
    st.markdown("Displays how each variable pushes the risk from the baseline.")
    
    # SHAP 解释器
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(input_df)
    
    # 绘制瀑布图
    fig, ax = plt.subplots(figsize=(6, 4))
    shap.plots.waterfall(shap_values[0], show=False)
    plt.tight_layout()
    st.pyplot(fig)

st.divider()
st.caption("Disclaimer: This tool is for academic research and adjunctive clinical decision-making only. It does not replace professional medical judgment. (Trained on SEER cohort n=2,830)")
