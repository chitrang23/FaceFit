import os
import requests
import time

# --- Configuration ---
# STEP 1: PASTE YOUR NEW, SECRET PEXELS API KEY IN THE LINE BELOW
API_KEY = "f7s1Kze8Z291vfXuLjtNvcEi7aUcYRvbgZjcPF7UOzDZya9rZQysj2Ko"

# You can adjust the number of images you want per category
IMAGES_PER_SHAPE = 40 

SEARCH_QUERIES = {
    "Round": "round face portrait",
    "Oval": "oval face portrait",
    "Square": "square jawline portrait",
    "Heart": "heart shaped face woman",
    "Long": "long face portrait"
}
HEADERS = {
    "Authorization": API_KEY
}

# --- Main Script ---
# STEP 2: DO NOT CHANGE THIS 'IF' CONDITION
if API_KEY == "YOUR_NEW_API_KEY_HERE" or not API_KEY:
    # This block runs ONLY if the API_KEY at the top is still the placeholder
    print("="*50)
    print("!!! ERROR: PLEASE PASTE YOUR PEXELS API KEY on line 6 !!!")
    print("Get your free key from: https://www.pexels.com/api/")
    print("="*50)
else:
    # This block runs ONLY if you have correctly replaced the API key
    if not os.path.exists('dataset'):
        os.makedirs('dataset')

    print("--- Starting Image Dataset Download (API Method) ---")

    for shape, query in SEARCH_QUERIES.items():
        print(f"\n--- Searching for: {shape} faces ---")
        
        shape_folder = os.path.join('dataset', shape)
        if not os.path.exists(shape_folder):
            os.makedirs(shape_folder)
            
        try:
            search_url = f"https://api.pexels.com/v1/search"
            params = {
                'query': query,
                'per_page': IMAGES_PER_SHAPE,
                'orientation': 'portrait'
            }
            
            response = requests.get(search_url, headers=HEADERS, params=params)
            response.raise_for_status()

            data = response.json()
            photos = data.get('photos', [])
            
            if not photos:
                print(f"  > No results found for '{query}'. Try a different search term.")
                continue

            for i, photo in enumerate(photos):
                img_url = photo['src']['large']
                
                try:
                    img_data = requests.get(img_url).content
                    file_path = os.path.join(shape_folder, f"{shape}_{i + 1}.jpg")
                    with open(file_path, 'wb') as handler:
                        handler.write(img_data)
                    print(f"  > Downloaded {os.path.basename(file_path)}")
                    time.sleep(0.2)
                except Exception as e:
                    print(f"  > Could not download image {img_url}. Error: {e}")

        except requests.exceptions.RequestException as e:
            print(f"\nFatal error connecting to Pexels API. This is likely due to an invalid API key.")
            print(f"Please regenerate your key on the Pexels website and try again.")
            print(f"Error details: {e}")
            break # Stop the script if one key fails

    print("\n--- Download complete! ---")