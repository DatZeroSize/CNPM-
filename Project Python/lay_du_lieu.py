import cv2
import os
import face_recognition

# Đường dẫn lưu ảnh
dataset_path = "dataset"

# Tạo thư mục "dataset" nếu chưa có
if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)

# Mở webcam (sử dụng camera mặc định)
cap = cv2.VideoCapture(0)

# Lấy tên người dùng và tạo thư mục cho người đó trong dataset
name = input("Nhập tên người mẫu: ")
person_folder = os.path.join(dataset_path, name)

if not os.path.exists(person_folder):
    os.makedirs(person_folder)

# Đếm số ảnh đã chụp
image_count = 0

# Giới hạn số ảnh tối thiểu cần thu thập
target_images = 5  # Bạn có thể thay đổi số lượng ảnh ở đây

print(f"Bắt đầu thu thập ảnh cho khuôn mặt của {name}. Nhấn 'q' để dừng hoặc đạt đủ {target_images} ảnh.")

while image_count < target_images:
    # Đọc từng khung hình từ webcam
    ret, frame = cap.read()

    # Nếu không có khung hình, tiếp tục
    if not ret:
        print("Không thể lấy khung hình từ webcam.")
        break

    # Phát hiện vị trí khuôn mặt trong khung hình
    face_locations = face_recognition.face_locations(frame)

    # Nếu phát hiện khuôn mặt
    if face_locations:
        for face_location in face_locations:
            # Lưu ảnh khuôn mặt vào thư mục
            image_path = os.path.join(person_folder, f"{name}_{image_count}.jpg")
            cv2.imwrite(image_path, frame)

            # Tăng số lượng ảnh đã chụp
            image_count += 1
            print(f"Đã lưu ảnh {image_count}/{target_images}")

            # Dừng khi đạt đủ số lượng ảnh
            if image_count >= target_images:
                print(f"Đã thu thập đủ {target_images} ảnh cho {name}.")
                break

    # Hiển thị khung hình
    cv2.imshow("Capture Face", frame)

    # Dừng khi nhấn phím 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng webcam và đóng tất cả cửa sổ
cap.release()
cv2.destroyAllWindows()
