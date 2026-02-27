import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

# ==========================================
# 1. 页面基本设置与标题
# ==========================================
st.set_page_config(page_title="GBC 远处转移风险预测", layout="wide", page_icon="🏥")

st.markdown("## 🏥 胆囊癌远处转移风险预测系统")
st.markdown("### Prediction System for Gallbladder Cancer Distant Metastasis")
st.caption("Based on XGBoost Machine Learning Model")

st.info("💡 请在左侧侧边栏输入患者的临床参数，点击按钮获取预测结果。")

# ==========================================
# 2. 加载模型
# ==========================================
@st.cache_resource
def load_model():
    return joblib.load('xgboost_model.pkl')

try:
    model = load_model()
except Exception as e:
    st.error(f"❌ 模型加载失败: {e}")
    st.warning("请先上传模型文件 xgboost_model.pkl 到当前目录！")
    st.stop()

# ==========================================
# 3. 侧边栏：患者信息输入
# ==========================================
st.sidebar.markdown("### 📋 Patient Information (患者信息)")

age = st.sidebar.slider("Age (年龄)", min_value=18, max_value=100, value=65, step=1)
sex_display = st.sidebar.selectbox("Sex (性别)", options=["Female (女性)", "Male (男性)"])
t_stage_display = st.sidebar.selectbox("T Stage (T分期)", options=["T1", "T2", "T3", "T4"])
lnr = st.sidebar.slider("Lymph Node Ratio (LNR, 淋巴结比率)", min_value=0.0, max_value=1.0, value=0.10, step=0.01)

st.sidebar.caption("LNR = 阳性淋巴结数 / 清扫淋巴结总数")

# 特征格式转换 (与模型训练时保持完全一致)
sex_code = 1 if "Male" in sex_display else 0
t_code = int(t_stage_display.replace("T", ""))

# 组合输入数据
input_features = ['Age_Numeric', 'Sex_Code', 'T_Code', 'LNR']
input_data = pd.DataFrame([[age, sex_code, t_code, lnr]], columns=input_features)

# ==========================================
# 4. 主界面：显示当前参数
# ==========================================
st.markdown("### 1. User Input Parameters (当前参数)")
st.dataframe(input_data, use_container_width=True)

# 设置约登指数计算出的最佳阈值
OPTIMAL_THRESHOLD = 0.546

# ==========================================
# 5. 预测按钮与结果展示
# ==========================================
if st.button("🚀 Start Prediction (开始预测)", type="primary"):
    
    # 获取高危类别的预测概率
    prob = model.predict_proba(input_data)[0][1]
    
    st.markdown("### 2. Prediction Result (预测结果)")
    
    # 绘制进度条
    st.progress(float(prob))
    
    # 使用列布局美化输出
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="Predicted Probability (转移概率)", value=f"{prob * 100:.2f}%")
        
    with col2:
        if prob >= OPTIMAL_THRESHOLD:
            st.error("⚠️ **High Risk (高危)**")
            st.markdown("**建议：** 密切随访，考虑进一步影像学检查及系统性治疗。")
        else:
            st.success("✅ **Low Risk (低危)**")
            st.markdown("**建议：** 常规随访，以局部手术治疗为主。")
            
    # ==========================================
    # 6. SHAP 瀑布图 (个体化预测逻辑)
    # ==========================================
    st.markdown("---")
    st.markdown("### 3. Individualized Prediction Logic (个体化预测逻辑)")
    st.markdown("下图展示了各临床特征对该患者预测结果的驱动情况：**红色**代表增加远处转移风险，**蓝色**代表降低风险。")
    
    try:
        # 设置英文字体，防止云端 Linux 服务器缺少中文字体导致方块乱码
        plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 初始化 SHAP 解释器
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(input_data)
        
        # 创建一个指定大小的画布
        fig, ax = plt.subplots(figsize=(8, 5))
        # 绘制瀑布图 (传入单个患者的 SHAP 值)
        shap.plots.waterfall(shap_values[0], max_display=10, show=False)
        
        # 调整排版并展示到网页上
        plt.tight_layout()
        st.pyplot(fig)
        
        # 清理内存
        plt.clf() 
        
    except Exception as e:
        st.warning(f"⚠️ SHAP 图表生成失败，错误信息: {e}")

# ==========================================
# 7. 页脚免责声明
# ==========================================
st.markdown("---")
st.caption("© 2026 GBC Prediction Model. For Research Use Only. 本系统结果仅供学术参考，不可替代临床医生的专业诊断。")
