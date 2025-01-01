import pandas as pd
import numpy as np
import pickle
import ast  # Dùng để chuyển chuỗi thành list

# 1. Đọc dữ liệu từ file CSV
def load_csv(file_path):
    try:
        df = pd.read_csv(file_path)
        print("Dữ liệu đã được tải thành công.")
        return df
    except Exception as e:
        print(f"Lỗi khi tải file CSV: {e}")
        return None

# 2. Chuyển đổi chuỗi biểu diễn danh sách thành thực tế danh sách
def convert_data_to_list(df):
    # Sử dụng ast.literal_eval để chuyển chuỗi thành list
    df['data'] = df['data'].apply(ast.literal_eval)
    print("Dữ liệu đã được chuyển đổi thành danh sách.")
    return df

# 3. Chuẩn hóa hoặc xử lý dữ liệu (nếu cần)
def preprocess_data(df):
    # Kiểm tra và xử lý giá trị thiếu (nếu có)
    print("Kiểm tra giá trị thiếu:")
    print(df.isnull().sum())
    df.fillna(df.mean(), inplace=True)
    
    # Chuẩn hóa dữ liệu số nếu cần
    # df['data'] = df['data'].apply(lambda x: np.array(x))
    
    print("Dữ liệu đã được xử lý.")
    return df

# 4. Lưu dữ liệu thành file pickle
def save_to_pickle(data, file_path):
    try:
        with open(file_path, 'wb') as file:
            pickle.dump(data, file)
        print(f"Dữ liệu đã được lưu thành công vào file: {file_path}")
    except Exception as e:
        print(f"Lỗi khi lưu file pickle: {e}")

# 5. Quy trình tiền xử lý và lưu dữ liệu
def preprocess_and_save_csv(input_csv_file, output_pickle_file):
    # Đọc dữ liệu từ file CSV
    df = load_csv(input_csv_file)
    
    if df is not None:
        # Chuyển dữ liệu chuỗi thành danh sách
        df = convert_data_to_list(df)
        
        # Tiền xử lý dữ liệu
        df = preprocess_data(df)
        
        # Lưu dữ liệu đã tiền xử lý vào file pickle
        save_to_pickle(df, output_pickle_file)

# 6. Thực thi
if __name__ == "__main__":
    # Đường dẫn file CSV và pickle đầu ra
    input_csv_file = "data_processed2.csv"  # Thay bằng đường dẫn thực tế
    output_pickle_file = "processed_data222.pkl"  # Thay tên file pickle mong muốn
    
    # Tiền xử lý và lưu dữ liệu
    preprocess_and_save_csv(input_csv_file, output_pickle_file)
