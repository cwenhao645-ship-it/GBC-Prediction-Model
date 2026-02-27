import streamlit as st
import joblib
import pandas as pd
import numpy as np

# ==========================================
# 1. 网页基础设置
# ==========================================
st.set_page_config(
    page_title="GBC Survival Predictor",
    layout="centered",
    initial_sidebar_state="expanded"
)

# 标题和简介
st.title("🏥 胆囊癌远处转移风险预测系统")
st.markdown("### Prediction System for Gallbladder Cancer Distant Metastasis")
st.markdown("Based on XGBoost Machine Learning Model")
st.info("💡 请在左侧侧边栏输入患者的临床参数，点击按钮获取预测结果。")
st.markdown("---")


# ==========================================
# 2. 加载模型 (读取您刚才保存的 .pkl)
# ==========================================
@st.cache_resource
def load_model():
    try:
        # 这里的名字必须和您上传的文件名完全一致！
        model = joblib.load('xgboost_model.pkl')
        return model
    except Exception as e:
        st.error(f"❌ 模型加载失败: {e}")
        return None


model = load_model()

# ==========================================
# 3. 侧边栏：输入患者参数
# ==========================================
st.sidebar.header("📋 Patient Information (患者信息)")


def user_input_features():
    # 1. 年龄 (Age)
    age = st.sidebar.slider("Age (年龄)", 18, 100, 65)

    # 2. 性别 (Sex) -> 需要转换为 0/1
    sex_display = st.sidebar.selectbox("Sex (性别)", ("Female (女性)", "Male (男性)"))
    # 逻辑：Male=1, Female=0 (根据您之前的代码逻辑)
    sex_code = 1 if "Male" in sex_display else 0

    # 3. T分期 (T Stage) -> 需要转换为 1/2/3/4
    t_display = st.sidebar.selectbox("T Stage (T分期)", ("T1", "T2", "T3", "T4"))
    t_map = {"T1": 1, "T2": 2, "T3": 3, "T4": 4}
    t_code = t_map[t_display.split()[0]]

    # 4. 淋巴结比率 (LNR)
    lnr = st.sidebar.slider("Lymph Node Ratio (LNR, 淋巴结比率)", 0.0, 1.0, 0.1, 0.01)
    st.sidebar.caption("LNR = 阳性淋巴结数 / 清扫淋巴结总数")

    # 封装成 DataFrame (列名必须和训练时完全一致！)
    data = {
        'Age_Numeric': age,
        'Sex_Code': sex_code,
        'T_Code': t_code,
        'LNR': lnr
    }
    features = pd.DataFrame(data, index=[0])
    return features


# 获取用户输入
if model is not None:
    input_df = user_input_features()

    # ==========================================
    # 4. 主界面：显示预测结果
    # ==========================================
    # 显示用户输入的参数概览
    st.subheader("1. User Input Parameters (当前参数)")
    # 美化显示
    display_df = input_df.copy()
    display_df['Sex_Code'] = "Male" if display_df['Sex_Code'][0] == 1 else "Female"
    display_df['T_Code'] = f"T{display_df['T_Code'][0]}"
    st.table(display_df)

    # 预测按钮
    if st.button("🚀 Start Prediction (开始预测)", type="primary"):
        with st.spinner('Calculating...'):
            # 预测概率
            prediction_proba = model.predict_proba(input_df)
            risk_score = prediction_proba[0][1]  # 取出属于类别1(转移)的概率

        st.subheader("2. Prediction Result (预测结果)")

        # 进度条可视化
        st.progress(float(risk_score))

        col1, col2 = st.columns(2)
        with col1:
            st.metric(label="Predicted Probability (转移概率)",
                      value=f"{risk_score * 100:.2f}%")

        with col2:
            # 阈值判断 (使用您文章中的最佳阈值 0.546)
            threshold = 0.546
            if risk_score > threshold:
                st.error("⚠️ High Risk (高危)")
                st.write("**建议：** 密切随访，考虑进一步检查。")
            else:
                st.success("✅ Low Risk (低危)")
                st.write("**建议：** 常规随访。")

else:
    st.warning("请先上传模型文件 xgboost_model.pkl")

# 页脚声明
st.markdown("---")
st.markdown("© 2024 GBC Prediction Model. For Research Use Only.")