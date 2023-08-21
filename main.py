import cv2
import face_recognition
import datetime
import json
import os
import threading
import winsound
import smtplib
from PIL import Image, ImageDraw, ImageFont
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
import torch
from pathlib import Path
from models.experimental import attempt_load

# Load the YOLOv8 model
yolov8_weights = 'path_to_yolov8_weights.pt'  # Replace with the actual path to your YOLOv8 weights file
yolov8_config = 'path_to_yolov8_config.yaml'  # Replace with the actual path to your YOLOv8 config file
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

yolo_model = attempt_load(yolov8_weights, map_location=device)
yolo_model = yolo_model.eval()

# Load YOLOv8 class names (if available)
yolo_class_names = []
with open('coco.names', 'r') as file:  # Replace with the path to your YOLO class names file
    yolo_class_names = [line.strip() for line in file.readlines()]

# Initialize the OpenCV object tracker
tracker = cv2.TrackerCSRT_create()

# Load sample images of known individuals and their names
known_images = [
    ("known_person1.jpg", "Dhanush"),
    ("known_person2.jpg", "Sanjay"),
    ("known_person3.jpg", "Varma"),
    ("known_person4.jpg", "Shyam"),
    ("known_person5.jpg", "Thamil")
    # Add more known individuals as needed
]

known_face_encodings = []
known_face_names = []

for image_path, name in known_images:
    image = face_recognition.load_image_file(image_path)
    face_encoding = face_recognition.face_encodings(image)[0]
    known_face_encodings.append(face_encoding)
    known_face_names.append(name)

# Directory to store known face images and detected frames
KNOWN_FACES_DIR = "known_faces"

# Dictionary to store detected persons' information
detected_persons = {}

def save_detected_persons():
    with open('detection_records.json', 'w') as file:
        json.dump(detected_persons, file)

def load_detected_persons():
    try:
        with open('detection_records.json', 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        return {}

def search_person(person_name):
    # Initialize a list to store matching names
    matching_names = []

    # Loop through detected_persons and check if the entered person_name matches any name (case-insensitive)
    for name in detected_persons.keys():
        if person_name.lower() in name.lower():
            matching_names.append(name)

    # Check if any matching names were found
    if matching_names:
        print(f"Last detection record for {', '.join(matching_names)}:")
        for name in matching_names:
            detection_times = detected_persons[name]
            if detection_times:  # Check if there are detection records for this person
                last_detection_time = detection_times[-1]  # Get the last detection timestamp
                image_path = f"{KNOWN_FACES_DIR}/{name}_{last_detection_time.replace(':', '-')}.jpg"  # Full image path
                image = cv2.imread(image_path)
                if image is not None:
                    cv2.imshow(f"{name} - Last Detection", image)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                    print(f"{name} was last detected at {last_detection_time}")
                else:
                    print(f"Image not found for {name} at {last_detection_time}")
            else:
                print(f"No detection records found for {name}")
    else:
        print(f"No detection records found for {person_name}")

def save_detected_frame(person_name, frame):
    current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    detected_persons.setdefault(person_name, []).append(current_time)

    # Create a copy of the frame for overlay
    frame_copy = frame.copy()

    # Overlay timestamp and name on the image
    timestamp_text = f"{person_name} - {current_time}"
    cv2.putText(frame_copy, timestamp_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Save entire detected frame as an image
    frame_path = f"{KNOWN_FACES_DIR}/{person_name}_{current_time.replace(':', '-')}.jpg"
    cv2.imwrite(frame_path, frame_copy)

def send_email_notification(image_path):
    msg = MIMEMultipart()
    msg['From'] = EMAIL_ADDRESS
    msg['To'] = RECIPIENT_EMAIL
    msg['Subject'] = "Unknown Person Detected"

    text = MIMEText("An unknown person was detected by the security system.")
    msg.attach(text)

    with open(image_path, 'rb') as f:
        img = MIMEImage(f.read())
    img.add_header('Content-Disposition', 'attachment', filename=os.path.basename(image_path))
    msg.attach(img)

    try:
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
        server.sendmail(EMAIL_ADDRESS, RECIPIENT_EMAIL, msg.as_string())
        server.quit()
        print("Email notification sent successfully!")
    except Exception as e:
        print("Error sending email:", e)

def load_known_faces():
    if not os.path.exists(KNOWN_FACES_DIR):
        os.makedirs(KNOWN_FACES_DIR)

    for image_path, name in known_images:
        known_face_path = f"{KNOWN_FACES_DIR}/{name}.jpg"
        if not os.path.exists(known_face_path):
            image = face_recognition.load_image_file(image_path)
            cv2.imwrite(known_face_path, image)

# Secure mode flag
secure_mode = False

# Email configuration
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
EMAIL_ADDRESS = "your_email@gmail.com"  # Replace with your email address
EMAIL_PASSWORD = "your_email_password"  # Replace with your email password
RECIPIENT_EMAIL = "recipient_email@example.com"  # Replace with the recipient's email address

# Open the camera feed
cap = cv2.VideoCapture(0)
frame_counter = 0
skip_frames = 3

# Initialize variables for object tracking
tracking_started = False
tracker = None

while True:
    # Read a frame from the camera
    ret, frame = cap.read()
    frame_counter += 1

    if not ret:
        break

    # Process the frame for face detection and recognition
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    for face_encoding, (top, right, bottom, left) in zip(face_encodings, face_locations):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown Person"

        for i, match in enumerate(matches):
            if match:
                name = known_face_names[i]
                break

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

        # Secure mode: If an unknown person is detected, play a beep sound and send an email
        if secure_mode and name == "Unknown Person":
            beep_thread = threading.Thread(target=winsound.Beep, args=(1000, 500))
            beep_thread.start()

            # Overlay timestamp on the image
            timestamp_text = f"Unknown Person - {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil_frame)
            font = ImageFont.truetype("arial.ttf", 36)
            draw.text((10, 10), timestamp_text, font=font, fill=(255, 0, 0))

            # Save the image with a timestamp overlay
            unknown_image_path = "unknown_person.jpg"
            pil_frame.save(unknown_image_path)

            # Send an email notification with the image
            email_thread = threading.Thread(target=send_email_notification, args=(unknown_image_path,))
            email_thread.start()

        # Save detected person's information and frame
        save_detected_frame(name, frame)  # Save the entire detected frame

        # Add the person's name to the frame
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

        # Start tracking the detected face
        if not tracking_started:
            tracker = cv2.TrackerCSRT_create()
            tracking_started = tracker.init(frame, (left, top, right - left, bottom - top))
        else:
            tracking_ok, bbox = tracker.update(frame)
            if tracking_ok:
                p1 = (int(bbox[0]), int(bbox[1]))
                p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                cv2.rectangle(frame, p1, p2, (255, 0, 0), 2)

    # Use YOLOv8 for whole-body detection
    results = yolo_model(frame)

    # Process YOLOv8 results
    for result in results.pred[0]:
        label = int(result[-1])
        conf = float(result[-2])
        if conf > 0.5:
            class_name = yolo_class_names[label] if yolo_class_names else str(label)
            x1, y1, x2, y2 = map(int, result[:4])

            # Label the whole body with the name
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

    cv2.imshow('Face Recognition', frame)  # Display the frame

    # Load detected persons' data from the JSON file
    loaded_data = load_detected_persons()
    detected_persons.update(loaded_data)

    # Search functionality
    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        person_name = input("Enter person's name: ")
        search_person(person_name)

    if key == ord('t'):  # Turn on secure mode
        secure_mode = True
        print("Secure mode ON")

    if key == ord('f'):  # Turn off secure mode
        secure_mode = False
        print("Secure mode OFF")

    if key == ord('q'):  # Exit loop when 'q' is pressed
        save_detected_persons()  # Save detected information before quitting
        break

cap.release()  # Release the camera feed
cv2.destroyAllWindows()
