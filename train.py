import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

def train_system():
    # 1. Tải dữ liệu
    try:
        df = pd.read_csv('diet_recommendations_dataset.csv')
    except FileNotFoundError:
        print("Lỗi: Không tìm thấy file diet_recommendations_dataset.csv")
        return

    # 2. Tiền xử lý dữ liệu (Label Encoding)
    # Vì Decision Tree cần dữ liệu số, ta chuyển đổi Disease_Type và Output
    le_disease = LabelEncoder()
    le_diet = LabelEncoder()

    df['Disease_Type'] = le_disease.fit_transform(df['Disease_Type'])
    df['Diet_Recommendation'] = le_diet.fit_transform(df['Diet_Recommendation'])

    # 3. Chuẩn bị Features và Label
    X = df[['BMI', 'Disease_Type', 'Age']]
    y = df['Diet_Recommendation']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 4. Huấn luyện mô hình Cây quyết định
    model = DecisionTreeClassifier(criterion='entropy', max_depth=None)
    model.fit(X_train, y_train)

    # 5. Lưu mô hình và bộ giải mã
    joblib.dump(model, 'diet_model.pkl')
    joblib.dump(le_disease, 'le_disease.pkl')
    joblib.dump(le_diet, 'le_diet.pkl')
    
    print("✅ Đã huấn luyện thành công! File 'diet_model.pkl' đã sẵn sàng.")

if __name__ == "__main__":
    train_system()