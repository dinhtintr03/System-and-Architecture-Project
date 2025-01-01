# from pydrive.auth import GoogleAuth
# from pydrive.drive import GoogleDrive
# import os

# # Xác thực với Google Drive
# def authenticate_drive():
#     gauth = GoogleAuth()
#     gauth.LocalWebserverAuth()  # Sẽ mở trình duyệt để xác thực
#     return GoogleDrive(gauth)

# # Tải folder từ Google Drive
# def download_folder(drive, folder_id, destination):
#     os.makedirs(destination, exist_ok=True)  # Tạo thư mục đích nếu chưa tồn tại
#     file_list = drive.ListFile({'q': f"'{folder_id}' in parents and trashed=false"}).GetList()
    
#     for file in file_list:
#         file_name = file['title']
#         file_id = file['id']
#         file_path = os.path.join(destination, file_name)

#         if file['mimeType'] == 'application/vnd.google-apps.folder':
#             # Nếu là thư mục, tải đệ quy
#             print(f"Tải thư mục: {file_name}")
#             download_folder(drive, file_id, file_path)
#         else:
#             # Nếu là tệp, tải tệp
#             print(f"Tải tệp: {file_name}")
#             file.GetContentFile(file_path)

# if __name__ == "__main__":
#     drive = authenticate_drive()
    
#     # Thay thế bằng folder ID và đường dẫn lưu trữ
#     FOLDER_ID = '1YkDGv0hc6A64ri4mfy5LPpNn2kXgBx-X?fbclid=IwY2xjawGpaY5leHRuA2FlbQIxMAABHTs8ZhYnTZ6OKd63rxkxQxFPyR2KJx-8yfO7O-OhLWLoiv00VUkrRBcHhw_aem_-_1lIDxPHb9nL99-Dmi1ug'
#     DESTINATION_PATH = 'data'
    
#     download_folder(drive, FOLDER_ID, DESTINATION_PATH)
#     print("Tải xuống hoàn tất!")
