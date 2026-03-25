import requests
import json
import os
from datetime import datetime

# Configuration
URL = "http://localhost:8001/partition/pdf"
PDF_PATH = "/Users/manqin/FinanceAutoReport/data/新海开户许可证20000520.pdf"  # Change this to your PDF filename
OUTPUT_FOLDER = "extraction_results"

def save_results_to_folder():
    # 1. Create the folder if it doesn't exist
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
        print(f"Created folder: {OUTPUT_FOLDER}")

    try:
        print(f"--- Processing {PDF_PATH} ---")
        
        with open(PDF_PATH, "rb") as f:
            files = {"file": (PDF_PATH, f, "application/pdf")}
            response = requests.post(URL, files=files)

        if response.status_code == 200:
            data = response.json()
            
            # 2. Generate a unique filename using timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_name = os.path.splitext(os.path.basename(PDF_PATH))[0]
            output_filename = f"{base_name}_{timestamp}.json"
            output_path = os.path.join(OUTPUT_FOLDER, output_filename)

            # 3. Save the JSON data
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
            
            print(f"Success! Results saved to: {output_path}")
            print(f"Total elements extracted: {len(data)}")
        else:
            print(f"Error {response.status_code}: {response.text}")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    save_results_to_folder()