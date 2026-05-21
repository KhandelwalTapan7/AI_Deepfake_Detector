"""
download_model.py — Aurora Deepfake Detector
Auto-downloads the model from Google Drive if not present locally.
"""

import os
import sys
import requests

# ── CONFIG ────────────────────────────────────────────────────
FILE_ID    = "1c-LZ6oggTAaN-Fn1poH0MBkE0HeXCt3_"
MODEL_NAME = "deepfake_detector_export.pth"
# ─────────────────────────────────────────────────────────────

MODEL_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
<<<<<<< HEAD
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_NAME)

DRIVE_URLS = [
    f"https://drive.usercontent.google.com/download?id={FILE_ID}&export=download&confirm=t",
    f"https://drive.google.com/uc?export=download&id={FILE_ID}&confirm=t",
]

=======
MODEL_PATH = os.path.join(MODEL_DIR, 'deepfake_detector_export.pth')  # must match app.py
FILE_ID    = "1c-LZ6oggTAaN-Fn1poH0MBkE0HeXCt3_"

URLS = [
    f"https://drive.usercontent.google.com/download?id={FILE_ID}&export=download&confirm=t",
    f"https://drive.google.com/uc?export=download&id={FILE_ID}&confirm=t",
]
>>>>>>> cd3605298 (ADD IMAGE)

def download_model():
    # Already present and valid?
    if os.path.exists(MODEL_PATH):
        mb = os.path.getsize(MODEL_PATH) / 1024 / 1024
        if mb > 5:
            print(f"✅ Model already present ({mb:.1f} MB) — skipping download")
            return True
<<<<<<< HEAD
        print(f"⚠️  Model file too small ({mb:.1f} MB) — re-downloading")
=======
        print(f"⚠️  Model too small ({mb:.1f} MB) — re-downloading")
>>>>>>> cd3605298 (ADD IMAGE)
        os.remove(MODEL_PATH)

    os.makedirs(MODEL_DIR, exist_ok=True)

<<<<<<< HEAD
    for attempt, url in enumerate(DRIVE_URLS, 1):
        print(f"📥 Download attempt {attempt}/{len(DRIVE_URLS)} ...")
=======
    for i, url in enumerate(URLS, 1):
        print(f"📥 Download attempt {i}/{len(URLS)} ...")
>>>>>>> cd3605298 (ADD IMAGE)
        try:
            session = requests.Session()
            session.headers['User-Agent'] = 'Mozilla/5.0'
            r = session.get(url, stream=True, timeout=300)
            r.raise_for_status()

<<<<<<< HEAD
            # If Drive returns HTML (virus-scan warning) instead of binary — try next URL
            content_type = r.headers.get('content-type', '')
            if 'text/html' in content_type:
                print("   ⚠️  Got HTML response (Drive scan page) — trying next URL")
=======
            # Drive returns HTML for large files (virus-scan page) — skip
            if 'text/html' in r.headers.get('content-type', ''):
                print("   ❌ Got HTML (Drive scan page) — trying next URL")
>>>>>>> cd3605298 (ADD IMAGE)
                continue

            downloaded = 0
            with open(MODEL_PATH, 'wb') as f:
                for chunk in r.iter_content(chunk_size=65536):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        print(f"\r   {downloaded / 1024 / 1024:.1f} MB...",
                              end='', flush=True)

            mb = downloaded / 1024 / 1024
            print(f"\n   Downloaded: {mb:.1f} MB")

            if mb < 5:
<<<<<<< HEAD
                print("   ❌ File too small — likely an error page, not the model")
=======
                print("   ❌ Too small — likely an error page, not the model")
>>>>>>> cd3605298 (ADD IMAGE)
                os.remove(MODEL_PATH)
                continue

            print(f"✅ Model saved → {MODEL_PATH}")
            return True

        except Exception as e:
<<<<<<< HEAD
            print(f"   ❌ Attempt {attempt} failed: {e}")
            if os.path.exists(MODEL_PATH):
                os.remove(MODEL_PATH)
            continue

    print("❌ All download attempts failed — app will run in forensic fallback mode")
=======
            print(f"   ❌ Attempt {i} failed: {e}")
            if os.path.exists(MODEL_PATH): os.remove(MODEL_PATH)

    print("❌ All attempts failed — app will run in forensic fallback mode")
>>>>>>> cd3605298 (ADD IMAGE)
    return False


if __name__ == '__main__':
    success = download_model()
    sys.exit(0 if success else 1)
