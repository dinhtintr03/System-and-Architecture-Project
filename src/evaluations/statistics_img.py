# import os

# # Định nghĩa nhãn tương ứng với từng folder
# labels_mapping = {
#     0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E',
#     5: 'F', 6: 'I', 7: 'K', 8: 'L', 9: 'O',
#     10: 'U', 11: 'V', 12: 'W', 13: 'Y'
# }

# # Hàm đếm số lượng hình ảnh trong một thư mục
# def count_images_in_folders(base_path):
#     stats = {}
#     for i in range(14):  # Duyệt qua các folder từ 0 đến 13
#         folder_name = str(i)
#         label = labels_mapping[i]
#         folder_path = os.path.join(base_path, folder_name)

#         if not os.path.isdir(folder_path):
#             print(f"Thư mục {folder_name} không tồn tại, bỏ qua.")
#             continue
        
#         # Đếm số lượng hình ảnh trong thư mục
#         image_count = sum(
#             1 for file in os.listdir(folder_path)
#             if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff'))
#         )
#         stats[label] = image_count
#     return stats

# if __name__ == "__main__":
#     # Thay đường dẫn này bằng đường dẫn đến thư mục gốc của bạn
#     base_directory = "data\\processed_data"
    
#     # Thống kê số lượng hình ảnh
#     result = count_images_in_folders(base_directory)
    
#     # In kết quả
#     print("Thống kê số lượng hình ảnh trong thư mục:")
#     for label, count in result.items():
#         print(f"Label {label}: {count} hình ảnh")

import os
import matplotlib.pyplot as plt

# Định nghĩa nhãn tương ứng với từng folder
labels_mapping = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E',
    5: 'F', 6: 'I', 7: 'K', 8: 'L', 9: 'O',
    10: 'U', 11: 'V', 12: 'W', 13: 'Y'
}

# Hàm đếm số lượng hình ảnh trong một thư mục
def count_images_in_folders(base_path):
    stats = {}
    for i in range(14):  # Duyệt qua các folder từ 0 đến 13
        folder_name = str(i)
        label = labels_mapping[i]
        folder_path = os.path.join(base_path, folder_name)

        if not os.path.isdir(folder_path):
            print(f"Thư mục {folder_name} không tồn tại, bỏ qua.")
            continue
        
        # Đếm số lượng hình ảnh trong thư mục
        image_count = sum(
            1 for file in os.listdir(folder_path)
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff'))
        )
        stats[label] = image_count
    return stats

# Hàm vẽ biểu đồ
def plot_image_statistics(stats):
    labels = list(stats.keys())
    counts = list(stats.values())

    plt.figure(figsize=(12, 7))
    plt.bar(labels, counts, color='skyblue', edgecolor='black')
    plt.xlabel("Labels", fontsize=14)
    plt.ylabel("Number of Images", fontsize=14)
    plt.title("Image Counts by Labels", fontsize=16)
    plt.xticks(fontsize=12)
    plt.yscale('log')  # Sử dụng scale logarit trên trục Y để dễ nhìn hơn
    plt.yticks(fontsize=12)

    # Hiển thị số lượng trên mỗi cột (tùy chỉnh vị trí để tránh trùng lặp)
    for i, count in enumerate(counts):
        plt.text(i, count + 0.5, str(count), ha='center', fontsize=10)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Thay đường dẫn này bằng đường dẫn đến thư mục gốc của bạn
    base_directory = "data\\processed_data"
    
    # Thống kê số lượng hình ảnh
    result = count_images_in_folders(base_directory)
    
    # In kết quả
    print("Thống kê số lượng hình ảnh trong thư mục:")
    for label, count in result.items():
        print(f"Label {label}: {count} hình ảnh")
    
    # Vẽ biểu đồ
    plot_image_statistics(result)
