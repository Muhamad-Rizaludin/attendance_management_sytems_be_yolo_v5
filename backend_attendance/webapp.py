# Import library yang diperlukan
import argparse
import json
# from flask import Flask, render_template, Response, jsonify
from flask import Flask, render_template, Response, jsonify
from flask_cors import CORS
import os
from subprocess import Popen
import numpy as np
# opsi 3
import cv2
import torch
from datetime import datetime
import csv
import os
import time

# Define the video_feed_runing variable at the global scope
video_feed_runing = False

# Inisialisasi aplikasi Flask
app = Flask(__name__)
CORS(app, methods=['POST', 'GET','PUT'])
# Muat model YOLOv5 yang telah dilatih
model = torch.hub.load('.', 'custom', path='best.pt', source='local')
model.eval()

# Tentukan ambang kepercayaan untuk deteksi objek
confidence_threshold = 0.75

# Muat nama-nama kelas
class_names = ['1219002', '1219010', '1219016', '1219019', '3219004', '3219017']


# Inisialisasi kamus kehadiran, kamus akurasi, dan penghitung deteksi tidak diketahui
global attendance

attendance = {class_name: set() for class_name in class_names}
true_positives = {class_name: 0 for class_name in class_names}
false_positives = {class_name: 0 for class_name in class_names}
false_negatives = {class_name: 0 for class_name in class_names}
accuracy = {class_name: 0 for class_name in class_names}

# Folder untuk menyimpan gambar yang terdeteksi
image_folder = 'result_images_pengujian'

# Buat folder gambar jika belum ada
if not os.path.exists(image_folder):
    os.makedirs(image_folder)

# Fungsi untuk meningkatkan gambar
def enhance_image(image):
    # Terapkan histogram equalization untuk meningkatkan kontras
    enhanced_image = cv2.equalizeHist(image)
    
    # Tambahkan nilai intensitas konstan ke semua piksel (mensimulasikan pencahayaan tambahan)
    brightness_factor = 20  # Sesuaikan nilai ini jika diperlukan
    enhanced_image = cv2.add(enhanced_image, brightness_factor)
    
    return enhanced_image  

# Route untuk streaming video dari webcam
@app.route('/video_feed', methods=['GET'])
def video_feed():
        global video_feed_runing
        
        video_feed_runing = True
        
        def detect_objects(image):
            # Buat salinan gambar asli
            image_copy = image.copy()

            # Ubah gambar BGR menjadi RGB
            image_rgb = cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB)

            # Lakukan deteksi objek
            results = model([image_rgb])

            # Dapatkan objek yang terdeteksi dan nilai kepercayaannya
            detections = results.pandas().xyxy[0]
            
            # Urutkan deteksi berdasarkan nilai kepercayaan secara menurun
            detections = detections.sort_values(by='confidence', ascending=False)

            # Inisialisasi daftar untuk menyimpan nama kelas yang terdeteksi dan kotak pembatasnya
            detected_classes = []

            # Iterasi melalui deteksi
            for _, detection in detections.iterrows():
                # Ekstrak nama kelas, kepercayaan, dan koordinat kotak pembatas
                class_id = int(detection['class'])
                confidence = detection['confidence']
                x_min = int(detection['xmin'])
                y_min = int(detection['ymin'])
                x_max = int(detection['xmax'])
                y_max = int(detection['ymax'])

                # Periksa apakah nilai kepercayaan deteksi di atas ambang dan kelas cocok dengan class_names
                if confidence > confidence_threshold and class_id < len(class_names) and class_names[class_id]:
                    # Ekstrak nama kelas
                    class_name = class_names[class_id]

                    # Tambahkan nama kelas yang terdeteksi dan kotak pembatasnya ke daftar
                    detected_classes.append((class_name, [x_min, y_min, x_max, y_max]))

            # Perbarui kamus kehadiran dan akurasi berdasarkan kelas yang terdeteksi
            current_datetime = datetime.now()
            for class_name, bounding_box in detected_classes:
                # Perbarui kamus kehadiran
                attendance[class_name].add(current_datetime)

                # Tambahkan true positives dan false positives sesuai dengan kelas yang terdeteksi
                if class_name in true_positives:
                    true_positives[class_name] += 1
                else:
                    false_positives[class_name] += 1

                # Gambar kotak pembatas dan label pada gambar_copy
                cv2.rectangle(image_copy, (bounding_box[0], bounding_box[1]), (bounding_box[2], bounding_box[3]), (0, 255, 0), 2)
                cv2.putText(image_copy, f"{class_name}: {confidence:.2f}", (bounding_box[0], bounding_box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Simpan gambar_copy dengan kotak pembatas dan label
                image_path = os.path.join(image_folder, f"{class_name}_{current_datetime.strftime('%Y%m%d%H%M%S')}.jpg")
                cv2.imwrite(image_path, image_copy)

                # Tangkap kelas dan simpan kehadiran
                capture_class(class_name, current_datetime, bounding_box, image_path)

            # Gambar kotak pembatas untuk kelas yang tidak diketahui
            for _, detection in detections.iterrows():
                class_id = int(detection['class'])
                confidence = detection['confidence']
                x_min = int(detection['xmin'])
                y_min = int(detection['ymin'])
                x_max = int(detection['xmax'])
                y_max = int(detection['ymax'])

                if confidence > confidence_threshold and  (class_id >= len(class_names) or not class_names[class_id]):
                    cv2.rectangle(image_copy, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
                    cv2.putText(image_copy, "Unknown", (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    
                    # Simpan gambar_copy dengan kotak pembatas dan label untuk kelas yang tidak diketahui
                    current_datetime = datetime.now()
                    image_path = os.path.join(image_folder, f"Unknown_{current_datetime.strftime('%Y%m%d%H%M%S')}.jpg")
                    cv2.imwrite(image_path, image_copy)
                    
                    # Tangkap kelas dan simpan kehadiran untuk kelas yang tidak diketahui
                    capture_class('Unknown', current_datetime, [x_min, y_min, x_max, y_max], image_path)
                    
            return image_copy
        
         # Function to get the attendance bounding box for a given timestamp
        def capture_class(class_name, timestamp, bounding_box, image_path):
            global true_positives, false_positives, false_negatives, accuracy

            if class_name in class_names or class_name == 'Unknown':
                # Check if the attendance CSV file exists
                if not os.path.isfile('attendance_pengujian.csv'):
                    # Create a new CSV file and write the header
                    with open('attendance_pengujian.csv', 'w', newline='') as file:
                        csv_writer = csv.writer(file)
                        csv_writer.writerow(['class_name', 'timestamp', 'x_min', 'y_min', 'x_max', 'y_max', 'image_path', 'is_unknown'])

                # Append the attendance data to the CSV file
                with open('attendance_pengujian.csv', 'a', newline='') as file:
                    csv_writer = csv.writer(file)
                    csv_writer.writerow([class_name, str(timestamp), bounding_box[0], bounding_box[1], bounding_box[2], bounding_box[3], image_path, class_name == 'Unknown'])

                print(f"Presensi Berhasil! Nim: {class_name}, Time: {timestamp}")
                # Return the JSON response to send it to the network
                # Increment true positives and update accuracy for the detected class
                if class_name in true_positives:
                    true_positives[class_name] += 1
                    accuracy[class_name] = true_positives[class_name] / (true_positives[class_name] + false_positives[class_name])
                else:
                    false_positives[class_name] += 1

                # Return a JSON response
                response = {
                    "status": "success",
                    "class_name": class_name,
                    "timestamp": str(timestamp),
                    "bounding_box": bounding_box,
                    "image_path": image_path,
                    "is_unknown": class_name == 'Unknown'
                }
            else:
                # If unknown class, skip capturing and display a message
                print(f"Unknown detected! Time: {timestamp}")

                # Increment false negatives for the unknown class
                false_negatives['Unknown'] += 1

                # Return a JSON response
                response = {
                    "status": "error",
                    "message": "Unknown detected"
                }

            # Convert the response to JSON format and return
            return json.dumps(response)

        # Fungsi untuk mendapatkan frame dari webcam
        def get_frame():
            
            global attendance
            start_time = time.time()
            frame_count = 0
            
            cap = cv2.VideoCapture(0)
            cap.set(3, 640)  # Atur lebar frame
            cap.set(4, 480)  # Atur tinggi frame

            while True:
                success, frame = cap.read()
                frame = cv2.resize(frame, (640, 480))  # Mengubah resolusi
                
                if not success:
                    break
                
                frame_count += 1
                elapsed_time = time.time() - start_time
                fps = frame_count / elapsed_time

                # Cetak FPS ke konsol
                print(f"FPS: {fps:.2f}")
                
                ret, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()

                # Decode buffer yang dienkoding dalam format JPEG menjadi gambar BGR
                image = cv2.imdecode(np.frombuffer(frame, dtype=np.uint8), cv2.IMREAD_COLOR)
                output_frame = detect_objects(image)

                # Konversi output_frame ke format JPEG
                ret, output_buffer = cv2.imencode('.jpg', output_frame)
                output_frame_encoded = output_buffer.tobytes()

                # Kirimkan output_frame_encoded
                yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + output_frame_encoded + b'\r\n')

                # Periksa tombol 'q' untuk keluar
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                # Bersihkan data kehadiran setelah memproses setiap frame
                attendance = {class_name: set() for class_name in class_names}
                
            # Lepaskan webcam dan tutup jendela OpenCV
            cap.release()
            cv2.destroyAllWindows()
    
        return Response(get_frame(), mimetype='multipart/x-mixed-replace; boundary=frame')
    
# Route untuk hasil deteksi dalam format JSON
@app.route('/detection_results_json', methods=['GET'])
def detection_results_json():
    
    global video_feed_runing
    
    if video_feed_runing :
        response_data = []
        for class_name, timestamps in attendance.items():
            for timestamp in timestamps:
                response_data.append({
                    "class_name": class_name,
                    "timestamp": str(timestamp),
                })
        return jsonify(response_data)
    else:
      return jsonify({"message": "Video feed is not running"})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aplikasi Flask untuk memamerkan model yolov5")
    parser.add_argument("--port", default=5000, type=int, help="nomor port")
    args = parser.parse_args()
    model.eval()
    app.run(host="0.0.0.0", port=args.port, debug=True)  # debug=True menyebabkan Restarting with stat
    