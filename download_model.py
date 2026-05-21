import os, sys, requests

MODEL_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
MODEL_PATH = os.path.join(MODEL_DIR, 'best_model.pth')
FILE_ID    = "YOUR_GOOGLE_DRIVE_FILE_ID"   # ← paste yours here

def download_model():
    if os.path.exists(MODEL_PATH):
        mb = os.path.getsize(MODEL_PATH) / 1024 / 1024
        if mb > 5:
            print(f"✅ Model already present ({mb:.1f} MB) — skipping download")
            return True
        print(f"⚠️  Found model but only {mb:.1f} MB — likely corrupt, re-downloading")
        os.remove(MODEL_PATH)

    os.makedirs(MODEL_DIR, exist_ok=True)

    urls = [
        f"https://drive.usercontent.google.com/download?id={FILE_ID}&export=download&confirm=t",
        f"https://drive.google.com/uc?export=download&id={FILE_ID}&confirm=t",
    ]

    for i, url in enumerate(urls, 1):
        print(f"📥 Download attempt {i}/2 ...")
        try:
            session = requests.Session()
            session.headers['User-Agent'] = 'Mozilla/5.0'
            r = session.get(url, stream=True, timeout=300)
            r.raise_for_status()

            # If Drive returns HTML (virus-scan page) instead of binary — skip
            content_type = r.headers.get('content-type', '')
            if 'text/html' in content_type:
                print(f"   ❌ Got HTML response (Drive scan page) — trying next URL")
                continue

            downloaded = 0
            with open(MODEL_PATH, 'wb') as f:
                for chunk in r.iter_content(65536):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        print(f"\r   {downloaded/1024/1024:.1f} MB...", end='', flush=True)

            mb = downloaded / 1024 / 1024
            print(f"\n   Downloaded {mb:.1f} MB")

            if mb < 5:
                print("   ❌ File too small — Drive probably returned error page")
                os.remove(MODEL_PATH)
                continue

            print(f"✅ Model saved to {MODEL_PATH}")
            return True

        except Exception as e:
            print(f"   ❌ Attempt {i} failed: {e}")
            if os.path.exists(MODEL_PATH):
                os.remove(MODEL_PATH)

    print("❌ All download attempts failed — will run in forensic mode")
    return False

if __name__ == '__main__':
    sys.exit(0 if download_model() else 1)