import os
import cv2
import matplotlib.pyplot as plt
import mediapipe as mp

# Định nghĩa nhãn tương ứng với từng folder
labels_mapping = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E',
    5: 'F', 6: 'I', 7: 'K', 8: 'L', 9: 'O',
    10: 'U', 11: 'V', 12: 'W', 13: 'Y'
}

# Sử dụng MediaPipe Hands để phát hiện bàn tay
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5)

# Hàm kiểm tra hình ảnh có trích xuất được đặc trưng bàn tay không
def check_hand_landmarks(image_path):
    try:
        image = cv2.imread(image_path)
        if image is None:
            return False
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = hands.process(image_rgb)
        return result.multi_hand_landmarks is not None
    except Exception as e:
        print(f"Lỗi khi xử lý {image_path}: {e}")
        return False

# Hàm kiểm tra từng folder và đếm số hình ảnh không nhận diện được
# def count_failed_images(base_path):
#     stats = {}
#     for i in range(14):  # Duyệt qua các folder từ 0 đến 13
#         folder_name = str(i)
#         label = labels_mapping[i]
#         folder_path = os.path.join(base_path, folder_name)

#         if not os.path.isdir(folder_path):
#             print(f"Thư mục {folder_name} không tồn tại, bỏ qua.")
#             continue
        
#         failed_count = 0
#         for file in os.listdir(folder_path):
#             if file.lower().endswith(('.png', '.jpg', '.jpeg')):
#                 image_path = os.path.join(folder_path, file)
#                 if not check_hand_landmarks(image_path):
#                     failed_count += 1
        
#         stats[label] = failed_count
#     return stats

def count_failed_images(base_path):
    stats = {}
    failed_images = {}
    for i in range(14):  # Duyệt qua các folder từ 0 đến 13
        folder_name = str(i)
        label = labels_mapping[i]
        folder_path = os.path.join(base_path, folder_name)

        if not os.path.isdir(folder_path):
            print(f"Thư mục {folder_name} không tồn tại, bỏ qua.")
            continue
        
        failed_count = 0
        failed_image_list = []
        for file in os.listdir(folder_path):
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(folder_path, file)
                if not check_hand_landmarks(image_path):
                    failed_count += 1
                    failed_image_list.append(file)
        
        stats[label] = failed_count
        failed_images[label] = failed_image_list
    return stats, failed_images

def save_failed_images_to_csv(failed_images, csv_path):
    with open(csv_path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["Label", "Failed Image"])
        for label, images in failed_images.items():
            for image in images:
                writer.writerow([label, image])

# Hàm vẽ biểu đồ
def plot_failed_image_statistics(stats):
    labels = list(stats.keys())
    counts = list(stats.values())

    plt.figure(figsize=(12, 7))
    plt.bar(labels, counts, color='orange', edgecolor='black')
    plt.xlabel("Labels", fontsize=14)
    plt.ylabel("Failed Images", fontsize=14)
    plt.title("Failed Hand Landmark Detection by Labels", fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    # Hiển thị số lượng trên mỗi cột
    for i, count in enumerate(counts):
        plt.text(i, count + 0.5, str(count), ha='center', fontsize=10)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Thay đường dẫn này bằng đường dẫn đến thư mục gốc của bạn
    base_directory = "data\\processed_data"
    
    # Đếm số lượng hình ảnh không nhận diện được
    result = count_failed_images(base_directory)
    
    # In kết quả
    print("Thống kê số lượng hình ảnh không nhận diện được landmark:")
    for label, count in result.items():
        print(f"Label {label}: {count} hình ảnh không nhận diện được")

    # Lưu tên các hình ảnh không nhận diện được vào file CSV
    csv_path = "failed_images.csv"
    save_failed_images_to_csv(failed_images, csv_path)
    print(f"Danh sách các hình ảnh không nhận diện được đã được lưu vào {csv_path}")
    
    # Vẽ biểu đồ
    plot_failed_image_statistics(result)
