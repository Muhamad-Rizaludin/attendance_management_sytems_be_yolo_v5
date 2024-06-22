# Backend Attendance System using yolo-v5

## Setup Instructions

### 1. Install Python
Ensure you have Python installed on your system. You can download it from the official [Python website](https://www.python.org/).
### 2. Clone Repository
### 3. Move to folder backend_attendance
        ```bash
        cd backend_attendance
        ```
### 3. Install dependency
        ```bash
        pip install -r requirements.txt
        ```
### 4. fyi: in requirements.txt not full dependency information for use this app
### 5. if you have error dependency, you must be install manually using pip command 
### 6. finish install

## How to run yolo-v5 only?

### 1. change model best.pt or yolov5s.pt with your uptodate model
### 3. success run
### 2. if you are run only yolo-v5
        ```bash
        python detect.py
        ``` 

## How to run backend attendance systems  ?

### 1. change model best.pt or yolov5s.pt with your uptodate model in webapp.py
### 2. change list class for youre uptodate class in youre model
### 3. run using command
        ```bash
        flask --app webapp run
        ```
### 4. succes run