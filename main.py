import streamlit as st
import joblib
import numpy as np

# Thiết kế giao diện
st.set_page_config(page_title="Hệ thống Tư vấn Dinh dưỡng", page_icon="🥗")
st.title("🥗 Diet Strategy Classifier")
st.subheader("Tư vấn chế độ ăn dựa trên AI")

# Nạp mô hình và các bộ mã hóa
@st.cache_resource
def load_assets():
    model = joblib.load('diet_model.pkl')
    le_disease = joblib.load('le_disease.pkl')
    le_diet = joblib.load('le_diet.pkl')
    return model, le_disease, le_diet

try:
    model, le_disease, le_diet = load_assets()

    # Nhập liệu từ người dùng
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Nhập Tuổi:", min_value=1, max_value=120, value=25)
        bmi = st.number_input("Chỉ số BMI:", min_value=10.0, max_value=50.0, value=22.0, step=0.1)
    
    with col2:
        # Lấy danh sách bệnh từ bộ mã hóa đã train
        disease_options = le_disease.classes_
        disease_selected = st.selectbox("Tình trạng sức khỏe:", disease_options)

    # Nút dự đoán
    if st.button("Phân tích ngay"):
        # Chuyển đổi input thành định dạng số
        disease_encoded = le_disease.transform([disease_selected])[0]
        input_data = np.array([[bmi, disease_encoded, age]])
        
        # Dự đoán
        prediction_idx = model.predict(input_data)
        recommendation = le_diet.inverse_transform(prediction_idx)[0]
        
        # Hiển thị kết quả
        st.success(f"### Kết quả: {recommendation}")
        
        # Giải thích logic Decision Tree (tùy chọn)
        with st.expander("Xem giải thích logic"):
            st.write(f"Dựa trên tình trạng {disease_selected} và chỉ số BMI {bmi}, "
                     f"thuật toán Cây quyết định xếp bạn vào nhóm {recommendation}.")

except Exception as e:
    st.warning(" Bạn cần chạy file 'train.py' trước để tạo mô hình.")