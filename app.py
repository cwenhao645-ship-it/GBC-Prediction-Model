import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

# ==========================================
# 1. Page Configuration and Titles
# ==========================================
st.set_page_config(page_title="GBC Distant Metastasis Prediction", layout="wide", page_icon="🏥")

st.markdown("## 🏥 Prediction System for Gallbladder Cancer Distant Metastasis")
st.caption("Based on XGBoost Machine Learning Model")

st.info("💡 Please enter the patient's clinical parameters in the left sidebar and click the button to get the prediction.")

# ==========================================
# 2. Load the Pre-trained Model
# ==========================================
@st.cache_resource
def load_model():
    return joblib.load('xgboost_model.pkl')

try:
    model = load_model()
except Exception as e:
    st.error(f"❌ Model loading failed: {e}")
    st.warning("Please upload the model file 'xgboost_model.pkl' to the current directory.")
    st.stop()

# ==========================================
# 3. Sidebar: Patient Information Input
# ==========================================
st.sidebar.markdown("### 📋 Patient Information")

age = st.sidebar.slider("Age", min_value=18, max_value=100, value=65, step=1)
sex_display = st.sidebar.selectbox("Sex", options=["Female", "Male"])
t_stage_display = st.sidebar.selectbox("T Stage", options=["T1", "T2", "T3", "T4"])
lnr = st.sidebar.slider("Lymph Node Ratio (LNR)", min_value=0.0, max_value=1.0, value=0.10, step=0.01)

st.sidebar.caption("LNR = Positive Lymph Nodes / Total Examined Lymph Nodes")

# Format conversion for the model
sex_code = 1 if sex_display == "Male" else 0
t_code = int(t_stage_display.replace("T", ""))

# Combine input data
input_features = ['Age_Numeric', 'Sex_Code', 'T_Code', 'LNR']
input_data = pd.DataFrame([[age, sex_code, t_code, lnr]], columns=input_features)

# ==========================================
# 4. Main Interface: Display Parameters
# ==========================================
st.markdown("### 1. User Input Parameters")
st.dataframe(input_data, use_container_width=True)

# Optimal threshold determined by Youden Index
OPTIMAL_THRESHOLD = 0.546

# ==========================================
# 5. Prediction Button and Results
# ==========================================
if st.button("🚀 Start Prediction", type="primary"):
    
    # Get the predicted probability for the positive class (Metastasis)
    prob = model.predict_proba(input_data)[0][1]
    
    st.markdown("### 2. Prediction Result")
    
    # Display progress bar
    st.progress(float(prob))
    
    # Use columns for better layout
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="Predicted Probability", value=f"{prob * 100:.2f}%")
        
    with col2:
        if prob >= OPTIMAL_THRESHOLD:
            st.error("⚠️ **High Risk**")
            st.markdown("**Recommendation:** Close follow-up is advised. Consider further comprehensive imaging and systemic therapy.")
        else:
            st.success("✅ **Low Risk**")
            st.markdown("**Recommendation:** Routine follow-up. Primary surgical treatment is recommended.")
            
    # ==========================================
    # 6. SHAP Waterfall Plot
    # ==========================================
    st.markdown("---")
    st.markdown("### 3. Individualized Prediction Logic")
    st.markdown("The plot below illustrates how each clinical feature drives the prediction for this specific patient: **red** arrows indicate an increased risk of distant metastasis, while **blue** arrows indicate a decreased risk.")
    
    try:
        # Set English font for matplotlib
        plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        # Initialize SHAP explainer
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(input_data)
        
        # Create figure and plot
        fig, ax = plt.subplots(figsize=(8, 5))
        shap.plots.waterfall(shap_values[0], max_display=10, show=False)
        
        # Adjust layout and display
        plt.tight_layout()
        st.pyplot(fig)
        
        # Clear memory
        plt.clf() 
        
    except Exception as e:
        st.warning(f"⚠️ Failed to generate SHAP plot. Error: {e}")

# ==========================================
# 7. Footer Disclaimer
# ==========================================
st.markdown("---")
st.caption("© 2026 GBC Prediction Model. For Research Use Only. This system provides academic reference and cannot replace professional clinical diagnosis.")
