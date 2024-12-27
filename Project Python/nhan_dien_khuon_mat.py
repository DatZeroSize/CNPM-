import json

import cv2
import face_recognition
import os
import numpy as np

# Đường dẫn đến thư mục chứa dataset
path = "C:\\Users\\HP\\PycharmProjects\\Open_CV_Conda\\dataset"
images = []
className = []
detected_faces = []  # Danh sách lưu các khuôn mặt đã nhận diện
myList = os.listdir(path)

for cl in myList:
    myList_Image = os.listdir(f"{path}/{cl}")
    for ci in myList_Image:
        curImg = cv2.imread(f"{path}/{cl}/{ci}")
        images.append(curImg)
        className.append(cl)

# Hàm mã hóa ảnh (Encoding Image)
def Encod_Image(images):
    EncodeList = []  # Danh sách chứa các mã hóa khuôn mặt
    for i in images:
        image = cv2.cvtColor(i, cv2.COLOR_BGR2RGB)  # Chuyển đổi ảnh từ BGR sang RGB
        encodings = face_recognition.face_encodings(image)  # Mã hóa khuôn mặt
        if encodings:  # Kiểm tra nếu có khuôn mặt trong ảnh
            EncodeList.append(encodings[0])  # Lưu mã hóa khuôn mặt đầu tiên
    return EncodeList

# Mã hóa tất cả ảnh trong dataset
encode_List = Encod_Image(images)

# Khởi tạo webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():

    exit()

# Vòng lặp lấy khung hình từ webcam và nhận diện khuôn mặt
while True:
    ret, frame = cap.read()  # Đọc khung hình từ webcam
    if not ret:

        break

    FramS = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    encodeInput_List = face_recognition.face_encodings(FramS)
    locateInput_List = face_recognition.face_locations(FramS)

    for encode, locate in zip(encodeInput_List, locateInput_List):
        faceDis = face_recognition.face_distance(encode_List, encode)
        matchIndex = np.argmin(faceDis)  # Tìm chỉ số của khuôn mặt giống nhất
        name = ''

        if faceDis[matchIndex] < 0.4:
            name = className[matchIndex]
            if name not in detected_faces:  # Chỉ thêm nếu chưa có trong danh sách
                detected_faces.append(name)
                # print(f"Kh: {name}")
        else:
            name = "Unknown"

        y1, x2, y2, x1 = locate
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, name, (x2, y2), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow("Nhan dien khuon mat", frame)

    # Bấm 'q' để thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Kết thúc, hiển thị các khuôn mặt đã nhận diện
#print("Danh sách khuôn mặt nhận diện:")
print(json.dumps(detected_faces))

cap.release()
cv2.destroyAllWindows()
