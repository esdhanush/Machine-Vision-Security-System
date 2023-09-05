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
import tkinter as tk
from tkinter import filedialog
from PIL import Image
import glob

# Load sample images of known individuals and their names
known_images = [
    ("data/known_person1.jpg", "Dhanush"),
    ("data/known_person8.jpg", "Varma")
]

FACE_DISTANCE_THRESHOLD = 0.6  # You can adjust this value as needed
# Dictionary to store the number of displayed photos for each person
displayed_photos_count = {}

known_face_encodings = []
known_face_names = []

UNKNOWN_PERSONS_DIR = "unknown_persons"
if not os.path.exists(UNKNOWN_PERSONS_DIR):
    os.makedirs(UNKNOWN_PERSONS_DIR)

for image_path, name in known_images:
    image = face_recognition.load_image_file(image_path)
    face_encoding = face_recognition.face_encodings(image)[0]
    known_face_encodings.append(face_encoding)
    known_face_names.append(name)

# Directory to store known face images and detected frames
KNOWN_FACES_DIR = "known_faces"
if not os.path.exists(KNOWN_FACES_DIR):
    os.makedirs(KNOWN_FACES_DIR)

# Dictionary to store detected persons' information
detected_persons = {}

def display_last_detected_photos(person_name):
    if person_name not in displayed_photos_count:
        displayed_photos_count[person_name] = 0

    photo_files = glob.glob(f"known_faces/{person_name}_*.jpg")

    if not photo_files:
        print(f"No photos found for {person_name}.")
        return

    # Sort the photo files by creation date in descending order
    photo_files.sort(key=os.path.getctime, reverse=True)

    # Display the last five detected photos, if available
    num_photos_to_display = min(5, len(photo_files))
    for i in range(displayed_photos_count[person_name], displayed_photos_count[person_name] + num_photos_to_display):
        if i >= len(photo_files):
            break

        photo_path = photo_files[i]
        photo = cv2.imread(photo_path)
        cv2.imshow(f"{person_name} - Photo {i + 1}", photo)

    displayed_photos_count[person_name] += num_photos_to_display

def display_matches(known_matches, unknown_matches):
    if len(known_matches) == 0 and len(unknown_matches) == 0:
        print("No matches found. Searching...")
        return

    print("Known Persons Matches:")
    for name, distance in known_matches:
        print(f"Name: {name}, Face Distance: {distance}")

    print("\nUnknown Persons Matches:")
    if len(unknown_matches) == 0:
        print("No matches found in the 'unknown_persons' folder.")
    else:
        for filename in unknown_matches:
            print(f"File Name: {filename}")

            # Display the matching image
            image_path = os.path.join(UNKNOWN_PERSONS_DIR, filename)
            image = cv2.imread(image_path)
            cv2.imshow(f"Matching Image: {filename}", image)


def save_detected_persons():
    with open('detection_records.json', 'w') as file:
        json.dump(detected_persons, file)


def save_detected_frame(person_name, frame):
    current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    detected_persons[person_name].append(current_time)

    # Overlay timestamp on the image
    timestamp_text = f"{person_name} - {current_time}"
    cv2.putText(frame, timestamp_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Save entire detected frame as image
    frame_path = f"{KNOWN_FACES_DIR}/{person_name}_{current_time.replace(':', '-')}.jpg"
    cv2.imwrite(frame_path, frame)


def send_email_notification(image_path):
    # Email configuration
    SMTP_SERVER = "smtp.gmail.com"
    SMTP_PORT = 587
    EMAIL_ADDRESS = "shansenthilsesd@gmail.com"
    EMAIL_PASSWORD = "uahcnrwlbnywbthx"
    RECIPIENT_EMAIL = "dhanush.cs21@bitsathy.ac.in"

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


# Secure mode flag
secure_mode = False

# Open the camera feed
cap = cv2.VideoCapture(0)

# Create a tkinter window for browsing unknown person's image
root = tk.Tk()
root.withdraw()  # Hide the tkinter window


def display_last_detected_photo(person_name):
        photo_files = glob.glob(f"known_faces/{person_name}_*.jpg")

        if not photo_files:
            print(f"No photos found for {person_name}.")
            return

        # Sort the photo files by creation date in descending order
        photo_files.sort(key=os.path.getctime, reverse=True)

        # Display the last five detected photos, if available
        num_photos_to_display = min(5, len(photo_files))
        for i in range(num_photos_to_display):
            photo_path = photo_files[i]
            photo = cv2.imread(photo_path)
            cv2.imshow(f"{person_name} - Photo {i + 1}", photo)


while True:
    ret, frame = cap.read()  # Read a frame from the camera

    if not ret:
        break

    # Process the frame for face detection and recognition
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    for face_location, face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown Person"

        for i, match in enumerate(matches):
            if match:
                name = known_face_names[i]
                break

        top, right, bottom, left = face_location
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, bottom + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

        # Save the entire frame when an unknown person is detected
        if name == "Unknown Person":
            timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            filename = f"unknown_person_{timestamp}.jpg"
            filepath = os.path.join(UNKNOWN_PERSONS_DIR, filename)
            cv2.imwrite(filepath, frame)

        # Secure mode: If unknown person is detected, play beep sound and send email
        if secure_mode and name == "Unknown Person":
            beep_thread = threading.Thread(target=winsound.Beep, args=(1000, 500))
            beep_thread.start()

            # Overlay timestamp on the image
            timestamp_text = f"Unknown Person - {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil_frame)
            font = ImageFont.truetype("arial.ttf", 36)
            draw.text((10, 10), timestamp_text, font=font, fill=(255, 0, 0))

            # Save the image with timestamp overlay
            unknown_image_path = "unknown_person.jpg"
            pil_frame.save(unknown_image_path)

            # Send email notification with image
            email_thread = threading.Thread(target=send_email_notification, args=(unknown_image_path,))
            email_thread.start()

        # Save detected person's information and frame
        if name != "Unknown Person":
            if name not in detected_persons:
                detected_persons[name] = []
            save_detected_frame(name, frame)  # Save the entire detected frame

    cv2.imshow('Face Recognition', frame)  # Display the frame

    key = cv2.waitKey(1) & 0xFF
    if key == ord('t'):  # Turn on secure mode
        secure_mode = True
        print("Secure mode ON")

    if key == ord('f'):  # Turn off secure mode
        secure_mode = False
        print("Secure mode OFF")

    if key == ord('m'):  # Open the option to browse for an unknown person's image
        file_path = filedialog.askopenfilename()
        if file_path:
            print("Searching...")
            # Load the selected image and analyze it
            unknown_image = face_recognition.load_image_file(file_path)
            unknown_face_encodings = face_recognition.face_encodings(unknown_image)

            if len(unknown_face_encodings) == 0:
                print("No faces found in the selected image.")
                continue

            unknown_face_encoding = unknown_face_encodings[0]  # Assuming only one face is in the image

            # Search for matches in known and unknown persons
            matches = []
            for known_face_encoding, known_face_name in zip(known_face_encodings, known_face_names):
                # Compare the unknown face with known faces
                face_distance = face_recognition.face_distance([known_face_encoding], unknown_face_encoding)
                if face_distance < FACE_DISTANCE_THRESHOLD:
                    matches.append((known_face_name, face_distance))

            # Search for matches in the "unknown_persons" folder
            unknown_persons_matches = []
            for filename in os.listdir(UNKNOWN_PERSONS_DIR):
                unknown_person_image = face_recognition.load_image_file(os.path.join(UNKNOWN_PERSONS_DIR, filename))
                unknown_person_face_encodings = face_recognition.face_encodings(unknown_person_image)

                for unknown_person_face_encoding in unknown_person_face_encodings:
                    face_distance = face_recognition.face_distance([unknown_person_face_encoding],
                                                                   unknown_face_encoding)
                    if face_distance < FACE_DISTANCE_THRESHOLD:
                        unknown_persons_matches.append(filename)

            # Display matching images and details
            display_matches(matches, unknown_persons_matches)

            # If a known person is detected, display their last 5 photos
            if matches:
                for name, _ in matches:
                    display_last_detected_photos(name)
    if key == ord('s'):  # Press 's' to search for the last detected photo of a person
        person_name_input = input("Enter the name: ")
        display_last_detected_photo(person_name_input)

    if key == ord('q'):  # Exit loop when 'q' is pressed
        save_detected_persons()  # Save detected information before quitting
        break

cap.release()  # Release the camera feed
cv2.destroyAllWindows()  # Close the display window
