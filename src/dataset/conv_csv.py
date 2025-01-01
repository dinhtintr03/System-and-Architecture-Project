import pickle
import pandas as pd

# Bước 1: Đọc dữ liệu từ file .pickle
with open('data_processed2.pickle', 'rb') as file:
    data = pickle.load(file)

# Bước 2: Chuyển dữ liệu thành DataFrame
# Nếu dữ liệu là dictionary với các cặp key-value
df = pd.DataFrame(data)

# Bước 3: Lưu DataFrame vào file CSV
df.to_csv('data_processed2.csv', index=False)
