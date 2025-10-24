import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import quote
import csv

# ---------------- CONFIG ----------------
gender_list = ["Male", "Female"]
face_shapes = ["Round", "Square", "Heart", "Oval"]
images_per_style = 5  # Number of images per style
base_dir = "hairstyles_dataset"
csv_file = "hairstyles_dataset.csv"
# ---------------------------------------

def create_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)

def search_bing_images(query, num_images=5):
    """Scrapes Bing Image Search for image URLs."""
    query = quote(query)
    url = f"https://www.bing.com/images/search?q={query}&form=HDRSC2"
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, "html.parser")
    imgs = []
    for m in soup.find_all("a", {"class": "iusc"}):
        try:
            m_json = m.get("m")
            start = m_json.find('"murl":"') + len('"murl":"')
            end = m_json.find('"', start)
            img_url = m_json[start:end]
            imgs.append(img_url)
            if len(imgs) >= num_images:
                break
        except:
            continue
    return imgs

def download_images(url_list, save_path, gender, shape, csv_writer):
    for i, url in enumerate(url_list):
        try:
            response = requests.get(url, timeout=10)
            ext = url.split(".")[-1].split("?")[0]
            ext = ext if ext.lower() in ["jpg","jpeg","png"] else "jpg"
            file_name = f"{gender}_{shape}_{i+1}.{ext}"
            file_path = os.path.join(save_path, file_name)
            with open(file_path, "wb") as f:
                f.write(response.content)
            print(f"Downloaded: {file_path}")
            # Write to CSV
            csv_writer.writerow([file_path, gender, shape])
        except Exception as e:
            print(f"Failed to download {url}: {e}")

# ---------------- MAIN ----------------
create_folder(base_dir)

with open(csv_file, mode='w', newline='', encoding='utf-8') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(["file_path", "gender", "face_shape"])  # header

    for gender in gender_list:
        for shape in face_shapes:
            folder_path = os.path.join(base_dir, gender, shape)
            create_folder(folder_path)
            search_term = f"{gender} {shape} haircut"
            print(f"Searching images for: {search_term}")
            urls = search_bing_images(search_term, images_per_style)
            download_images(urls, folder_path, gender, shape, csv_writer)

print("Dataset generation complete! CSV mapping saved as:", csv_file)
