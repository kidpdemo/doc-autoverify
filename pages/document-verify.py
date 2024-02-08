import streamlit as st
import os
import pathlib
from os import listdir
from os.path import isfile, join

# -*- coding: utf-8 -*-

# !sudo apt-get install tesseract-ocr
# !sudo apt-get install libtesseract-dev
# !pip install pytesseract fitz PyMuPDF

import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from keras.applications.vgg16 import VGG16, preprocess_input
import fitz
import base64

import pytesseract
from PIL import Image
import re
import json

# Function to extract text from a region of interest (ROI) in an image
def extract_text_from_roi(image_path, roi_box):
    # Open the image
    image = Image.open(image_path)

    # Crop the image using the ROI coordinates
    cropped_image = image.crop(roi_box)

    # Use pytesseract to perform OCR on the cropped image
    text = pytesseract.image_to_string(cropped_image)
    return text


# Load the pre-trained VGG16 model
model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Function to classify image color
def classify_image_color(img, filename):
    if isinstance(img, str):
        # If img is a file path, read the image
        img = cv2.imread(img)

    if img is None or img.size == 0:
        print(f"Failed to load or empty image: {filename}")
        return "unknown"

    if len(img.shape) != 3 or img.shape[2] != 3:
        print(f"Invalid image format: {filename}")
        return "unknown"

    # Calculate differences for each channel
    r_g = np.count_nonzero(abs(img[:, :, 0] - img[:, :, 1]))
    r_b = np.count_nonzero(abs(img[:, :, 0] - img[:, :, 2]))
    g_b = np.count_nonzero(abs(img[:, :, 1] - img[:, :, 2]))

    # Sum of differences
    diff_sum = float(r_g + r_b + g_b)
    ratio = diff_sum / img.size

    print(ratio)
    st.write('confidence factor ' + str(ratio))
    if ratio > 0.005:
        return "Original"
    else:
        return "Duplicate"

# Function to convert PDF to images
def convert_pdf_to_images(pdf_path, output_folder):
    images = []
    pdf_doc = fitz.open(pdf_path)
    for page_num in range(pdf_doc.page_count):
        try:
            page = pdf_doc[page_num]
            image_matrix = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72))
            img_filename = f"{output_folder}/page_{page_num}.png"
            cv2.imwrite(img_filename, np.frombuffer(image_matrix.samples, dtype=np.uint8).reshape(image_matrix.h, image_matrix.w, 3))
            images.append(img_filename)
        except Exception as e:
            print(f"Error converting PDF page {page_num}: {e}")
    return images

# Function to extract color features from an image
def extract_color_features(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    std_dev = np.std(hsv, axis=(0, 1))
    return std_dev

# Function to load images from a folder
def load_images(folder):
    images = []
    labels = []
    for filename in os.listdir(folder):
        if filename.lower().endswith(('.jpg', '.png')):
            image_path = os.path.join(folder, filename)
            image = cv2.imread(image_path)
            images.append(image)
            labels.append(1 if extract_color_features(image)[1] > 0 else 0) # 1 for color, 0 for black and white
    return images, labels

# Load images and labels
folder_path = "./data"

# Function to classify image as black and white or color
def classify_image(img):
    if len(img.shape) != 3 or img.shape[2] != 3:
        print(f"Invalid image format")
        return "unknown"
    grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if cv2.countNonZero(grayscale) == 0:
        return "Duplicate"
    else:
        return "Original"

def process_files (filename):
    predictions = []
    total_images = 0
    duplicate_count = 0
    original_count = 0

    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
        img = cv2.imread(filename)
        result = classify_image_color(img, filename)
        print(f"{filename}: {result}")
        predictions.append((filename, result))
        total_images += 1
        if result == "Duplicate":
            duplicate_count += 1
        elif result == "Original":
            original_count += 1
    elif filename.lower().endswith('.pdf'):
        pdf_images = convert_pdf_to_images(filename, folder_path)
        for page_num, img_path in enumerate(pdf_images):
            img_filename = f"page_{page_num, filename}"
            img = cv2.imread(img_path)
            result = classify_image_color(img, img_filename)
            print(f"{img_filename}: {result}")
            predictions.append((img_filename, result))
            total_images += 1
            if result == "Duplicate":
                duplicate_count += 1
            elif result == "Original":
                original_count += 1

    if duplicate_count > 0:
        st.write("Document it not original")
    else:
        st.write("Document is original")

# Process files in the folder
def process_files_in_folder(folder_path):
    predictions = []
    total_images = 0
    duplicate_count = 0
    original_count = 0

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            img = cv2.imread(file_path)
            result = classify_image_color(img, filename)
            print(f"{filename}: {result}")
            predictions.append((filename, result))
            total_images += 1
            if result == "Duplicate":
                duplicate_count += 1
            elif result == "Original":
                original_count += 1
        elif filename.lower().endswith('.pdf'):
            pdf_images = convert_pdf_to_images(file_path, folder_path)
            for page_num, img_path in enumerate(pdf_images):
                img_filename = f"page_{page_num, filename}"
                img = cv2.imread(img_path)
                result = classify_image_color(img, img_filename)
                print(f"{img_filename}: {result}")
                predictions.append((img_filename, result))
                total_images += 1
                if result == "Duplicate":
                    duplicate_count += 1
                elif result == "Original":
                    original_count += 1

    st.write(f"Total images processed: {total_images}")
    st.write(f"Total Duplicate Images:{duplicate_count}")
    st.write(f"Total Original Images:{original_count}")
    st.write(f"Percentage of Duplicate Images: {(duplicate_count / total_images) * 100}%")
    st.write(f"Percentage of Original Images: {(original_count / total_images) * 100}%")

def displayPDF(file):
    # Opening file from file path
    with open(file, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')

    # Embedding PDF in HTML
    pdf_display = F'<embed src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf">'

    # Displaying File
    st.markdown(pdf_display, unsafe_allow_html=True)

st.write("""
# UMS Document verification
# Used by Document Verification Team
""")
parent_path = pathlib.Path(__file__).parent.parent.resolve()
data_path = os.path.join(parent_path, "data")
onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path, f))]
option = st.sidebar.selectbox('Pick a dataset', onlyfiles)
file_location=os.path.join(data_path, option)
# use `file_location` as a parameter to the main script

print(file_location)
st.write('Processing the document... :sunglasses:')
st.write(file_location)

if 'ALL' in file_location:
    process_files_in_folder(folder_path)
else:
    process_files(file_location)
    if 'pdf' in file_location:
        displayPDF(file_location)
    else:
        st.image(file_location, caption='Document', use_column_width=True)

#######

if 'pdf' not in file_location:
    # Define the ROI coordinates for the secondary region (for header)
    roi_sslc = (100, 100, 2000, 500)

    # Example usage
    image_path = file_location

    # Extract text from the secondary ROI
    text_sslc = extract_text_from_roi(image_path, roi_sslc)

    # Define the pattern for secondary education
    pattern_secondary_education = r"(S\.S\.L\.C|CLASS - X|SECONDARY EDUCATION|X STANDARD)"

    # Find matches for secondary education
    secondary_education_match = re.search(pattern_secondary_education, text_sslc, re.IGNORECASE)
    secondary_education = secondary_education_match.group(0).strip() if secondary_education_match else None
    print("Secondary Education:", secondary_education)

    # Check if any of the expected keywords is present in the extracted text
    if secondary_education:
        st.write("The uploaded document is Secondary Certificate. Proceeding with extraction.")
        # Add your extraction logic here based on identified patterns

        # Define the ROI coordinates for different fields
        roi_name = roi_fathers_name = roi_dob = (100, 100, 2000, 1150)
        roi_total_marks = (100, 1200, 2000, 2200)

        # Extract text from ROIs for different fields
        text_name = extract_text_from_roi(image_path, roi_name)
        text_fathers_name = extract_text_from_roi(image_path, roi_fathers_name)
        text_dob = extract_text_from_roi(image_path, roi_dob)
        text_total_marks = extract_text_from_roi(image_path, roi_total_marks)

        # Define regular expressions for extracting specific fields
        pattern_name = r"\bName\s*:\s*(.*)"
        pattern_fathers_name = r"\bFather’s Name\s*:\s*(.*)"
        pattern_dob = r"\b\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\b"
        pattern_total_marks = r'(\d{1,3}(?:\.\d{1,2})?)%'

        # Extract fields using regular expressions
        name = re.findall(pattern_name, text_name)[0].strip() if re.findall(pattern_name, text_name) else None
        fathers_name = re.findall(pattern_fathers_name, text_fathers_name)[0].strip() if re.findall(pattern_fathers_name, text_fathers_name) else None
        dob = re.findall(pattern_dob, text_dob)[0].strip() if re.findall(pattern_dob, text_dob) else None
        total_marks_match = re.search(pattern_total_marks, text_total_marks)
        total_marks = total_marks_match.group(1) if total_marks_match else None

        # Determine pass/fail status based on total marks percentage
        pass_threshold = 35
        pass_status = "Pass" if total_marks and float(total_marks) >= pass_threshold else "Fail"

        # Store extracted information in a dictionary
        extracted_data = {
            "Name": name,
            "Father's Name": fathers_name,
            "Date of Birth": dob,
            "Total Marks": total_marks,
            "Pass/Fail Status": pass_status
        }
        st.write(extracted_data)

        # Write the extracted data to a JSON file
        # output_json_file = "/content/extracted_data.json"
        # with open(output_json_file, 'w') as json_file:
        #     json.dump(extracted_data, json_file, indent=4)

        # print("Extracted data saved to:", output_json_file)
    else:
        st.write("The uploaded document is not recognized as secondary education.")
