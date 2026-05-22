import cv2
import os

# Use relative paths
input_folder  = "dataset"
output_folder = os.path.join("signs", "words")

os.makedirs(output_folder, exist_ok=True)

# Only word folders, not alphabet letters
words = ["Bad", "Good", "Hello", "Help", "No", "Please", "Sorry", "Stop", "Thank You"]

def sharpness_score(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

for word in words:
    word_folder = os.path.join(input_folder, word)
    if not os.path.exists(word_folder):
        print(f"Skipped: {word} — folder not found")
        continue

    images = [f for f in os.listdir(word_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if not images:
        print(f"Skipped: {word} — no images")
        continue

    best_img   = None
    best_score = -1

    for img_name in images:
        img_path = os.path.join(word_folder, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue
        score = sharpness_score(img)
        if score > best_score:
            best_score = score
            best_img   = img

    if best_img is not None:
        # Crop bottom half only — removes face at top
        h, w = best_img.shape[:2]
        cropped = best_img[h//2:h, :]
        out_path = os.path.join(output_folder, f"{word}.jpg")
        cv2.imwrite(out_path, cropped)
        print(f"✅ Saved: {word}.jpg (sharpness: {best_score:.1f})")

print("\n✅ Done! Best word photos saved to signs/words/")