# input_handlers/image_input.py

import os
import tempfile
import pytesseract
from PIL import Image
from google.colab import files

def extract_text_from_image(image_path):
    """
    Core OCR processing function that extracts text from a given image file.
    """
    if not os.path.exists(image_path):
        print(f"❌ File not found at path: {image_path}")
        return None

    try:
        image = Image.open(image_path)
        print(f"📁 Loaded: {os.path.basename(image_path)}")
        print("🔍 Extracting text from image...")

        # Perform OCR using pytesseract
        extracted_text = pytesseract.image_to_string(image)

        if extracted_text and extracted_text.strip():
            print("✅ Text successfully extracted!")
            return extracted_text.strip()
        else:
            print("⚠️ No readable text was found in the image.")
            return None
            
    except Exception as e:
        print(f"❌ Failed to process the image: {str(e)}")
        return None


def handle_image_input():
    """
    Provides an image upload button in Colab, saves the uploaded file temporarily,
    and extracts text from it. This is the main function to be called from the app.
    """
    print("📤 Please upload an image file (.png, .jpg, etc.).")
    uploaded = files.upload()

    if not uploaded:
        print("⚠️ No file uploaded.")
        return None

    # Get the name of the first uploaded file
    file_name = next(iter(uploaded))
    
    # Write the uploaded bytes to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file_name)[1]) as temp_f:
        temp_f.write(uploaded[file_name])
        temp_file_path = temp_f.name
    
    # Extract text from the temporary image file
    extracted_text = extract_text_from_image(temp_file_path)

    # Clean up the temporary file
    os.remove(temp_file_path)
    
    if extracted_text:
        print("\n--- Extracted Text ---")
        print(extracted_text)
        print("--------------------")

    return extracted_text


# --- Example of how to use this in a Colab notebook ---
if __name__ == '__main__':
    print("--- Testing ImageHandler in Colab ---")

    # You must run these installation commands in a separate Colab cell first:
    # !apt-get install -y tesseract-ocr
    # !pip install pytesseract

    # This will trigger the upload dialog in your notebook.
    handle_image_input()
