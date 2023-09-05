import streamlit as st
import cv2
import os
import glob
from PIL import Image

# Load sample images of known individuals and their names
known_images = [
    ("data/known_person1.jpg", "Dhanush"),
    ("data/known_person2.jpg", "Sanjay")

]

# Directory to store known face images and detected frames
KNOWN_FACES_DIR = "known_faces"
if not os.path.exists(KNOWN_FACES_DIR):
    os.makedirs(KNOWN_FACES_DIR)

# Initialize the Streamlit app
st.title("Face Search App")

# Create a sidebar with options
option = st.sidebar.selectbox("Select an option:", ["Search by Name", "Search by Image"])

# Function to display the last detected photos of a person
def display_last_detected_photos(person_name):
    photo_files = glob.glob(f"known_faces/{person_name}_*.jpg")

    if not photo_files:
        st.write(f"No photos found for {person_name}.")
        return

    # Sort the photo files by creation date in descending order
    photo_files.sort(key=os.path.getctime, reverse=True)

    # Display the last five detected photos, if available
    num_photos_to_display = min(5, len(photo_files))
    for i in range(num_photos_to_display):
        photo_path = photo_files[i]
        st.image(photo_path, caption=f"{person_name} - Photo {i + 1}", use_column_width=True)

# Function to search by image
def search_by_image():
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

    if uploaded_image:
        st.write("Searching...")
        # Load and display the uploaded image
        st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

        # Implement your image comparison logic here
        # For example, you can use face recognition to find a match
        # If a match is found, set the match_found variable to True

        match_found = False  # Placeholder; replace with your actual logic

        # If a match is found, display the person's name and their last detected photos
        if match_found:
            person_name = "Matched Person"  # Replace with the actual name
            st.write(f"Match found! Person: {person_name}")
            display_last_detected_photos(person_name)

# Main Streamlit app logic
if option == "Search by Name":
    person_name_input = st.text_input("Enter the name:")
    if st.button("Search"):
        display_last_detected_photos(person_name_input)
elif option == "Search by Image":
    search_by_image()
