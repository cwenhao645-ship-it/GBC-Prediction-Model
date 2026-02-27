import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

# ==========================================
# 1. 网页全局配置 (匹配新叙事逻辑)
# ==========================================
st.set_page_config(
    page_title="GBC Occult Metastasis Assessor",
    page_icon="🩺",
    layout="wide"
)

# 隐藏 Streamlit 默认的菜单和页脚
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

# ==========================================
# 2. 加载模型 (使用缓存机制提升加载速度)
# ==========================================
@st.cache_resource
def load_model():
    # 请确保 GBC_LNR_XGB_Model.pkl 与 app.py 在同一目录下
    try:
        model = joblib.load("GBC_LNR_XGB_Model.pkl")
        return model
    except FileNotFoundError:
        st.error("Error: Model file 'GBC_LNR_XGB_Model.pkl' not found. Please ensure it is in the same directory as app.py.")
        st.stop()

model = load_model()

# ==========================================
# 3. 网页主标题与说明
# ==========================================
st.title("🩺 Post-operative Occult Metastasis Risk Assessment for Gallbladder Cancer")
st.markdown("""
**A Machine Learning Tool Powered by Lymph Node Ratio (LNR)**

This tool is designed to identify the risk of *occult systemic dissemination* using post-operative pathological parameters. 
By rectifying potential stage migration caused by inadequate lymph node examination, it assists clinicians in determining whether intensive adjuvant interventions (e.g., PET-CT screening, systemic chemotherapy) are required after primary surgery.
""")
st.divider()

# ==========================================
# 4. 侧边栏：收集患者临床基线数据
# ==========================================
st.sidebar.header("📋 Patient Post-operative Parameters")

# 4.1 年龄
age = st.sidebar.number_input("Age (Years)", min_value=18, max_value=100, value=65, step=1)

# 4.2 性别
sex = st.sidebar.selectbox("Sex", options=["Female", "Male"])

# 4.3 病理 T 分期
t_stage = st.sidebar.selectbox("Pathological T Stage (pT)", options=["T1", "T2", "T3", "T4"])

# 4.4 淋巴结比率 (核心变量)
st.sidebar.markdown("---")
st.sidebar.markdown("**Pathological Lymph Node Status**")
lnr = st.sidebar.slider(
    "Lymph Node Ratio (LNR)", 
    min_value=0.0, 
    max_value=1.0, 
    value=0.30, 
    step=0.01,
    help="LNR = Positive Nodes / Total Examined Nodes from the pathology report."
)

st.sidebar.markdown("---")
predict_btn = st.sidebar.button("Assess Occult Metastasis Risk 🚀", use_container_width=True)

# ==========================================
# 5. 数据转换逻辑
# ==========================================
# 将输入转化为模型需要的数值格式
sex_code = 1 if sex == "Male" else 0
t_code_map = {"T1": 1, "T2": 2, "T3": 3, "T4": 4}
t_code = t_code_map[t_stage]

# 必须与模型训练时的特征名称和顺序严格一致
feature_names = ['Age_Numeric', 'Sex_Code', 'T_Code', 'LNR']
input_data = pd.DataFrame([[age, sex_code, t_code, lnr]], columns=feature_names)

# ==========================================
# 6. 核心预测与结果展示
# ==========================================
# 定义高危阈值 (我们在 Table 4 中确定的最佳阈值)
OPTIMAL_THRESHOLD = 0.546

if predict_btn:
    # 模型预测
    prob = model.predict_proba(input_data)[0][1]
    
    col1, col2 = st.columns([1, 1.2])
    
    with col1:
        st.subheader("📊 Risk Stratification Result")
        
        # 结果分级判定
        if prob >= OPTIMAL_THRESHOLD:
            st.error(f"### Probability of Occult Distant Metastasis: {prob * 100:.1f}%")
            st.error("🚨 **Risk Level: HIGH RISK (Deep Screening Alert)**")
            st.warning("""
            **Clinical Recommendation:** Despite potentially negative intraoperative exploration, the high pathological lymph node burden (LNR) indicates a significant risk of occult systemic dissemination. 
            **Immediate PET-CT screening and early initiation of intensive adjuvant systemic therapy (e.g., chemotherapy) are strongly recommended.**
            """)
        else:
            st.success(f"### Probability of Occult Distant Metastasis: {prob * 100:.1f}%")
            st.success("✅ **Risk Level: LOW RISK**")
            st.info("""
            **Clinical Recommendation:** The patient has a low probability of occult metastasis (High Negative Predictive Value). Standard post-operative follow-up and regular surveillance are recommended.
            """)
            
    # ==========================================
    # 7. SHAP 个体化解释瀑布图
    # ==========================================
    with col2:
        st.subheader("🧠 Model Interpretation (SHAP)")
        st.markdown("The waterfall plot explains how each feature pushes the patient's risk higher (red) or lower (blue) from the baseline risk.")
        
        with st.spinner("Generating explanation..."):
            try:
                # 初始化 SHAP 解释器
                explainer = shap.TreeExplainer(model)
                shap_values = explainer(input_data)
                
                # 绘制瀑布图
                fig = plt.figure(figsize=(8, 5))
                # 临时调整 matplotlib 设置以适应网页显示
                plt.rcParams.update({'font.size': 10})
                shap.plots.waterfall(shap_values[0], max_display=10, show=False)
                
                # 在 Streamlit 中显示图表
                st.pyplot(fig)
                plt.clf() # 清除画布防止内存泄漏
            except Exception as e:
                st.error(f"Error generating SHAP plot: {e}")

else:
    # 初始欢迎界面
    st.info("👈 Please input the post-operative parameters in the sidebar and click **'Assess Occult Metastasis Risk'** to see the individualized evaluation and SHAP explanation.")

# 页脚信息
st.markdown("---")
st.caption("Disclaimer: This tool is for academic and research purposes only and should not replace professional clinical judgment. The model prioritizes sensitivity to avoid missing high-risk patients.")
