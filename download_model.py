import os, requests, sys

MODEL_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
MODEL_PATH = os.path.join(MODEL_DIR, 'deepfake_detector_export.pth')
FILE_ID    = "1pIat4WKLLZZFY9JflrdacFYsityCx-nb"
MODEL_URL  = f"https://drive.google.com/uc?export=download&id={FILE_ID}"

def download_model():
    if os.path.exists(MODEL_PATH):
        size_mb = os.path.getsize(MODEL_PATH) / 1024 / 1024
        print(f"✅ Model already present ({size_mb:.1f} MB) — skipping download")
        return True

    os.makedirs(MODEL_DIR, exist_ok=True)
    print("📥 Downloading model from Google Drive...")

    try:
        session  = requests.Session()
        response = session.get(MODEL_URL, stream=True, timeout=180)

        # Bypass Google Drive virus-scan warning page for large files
        confirm_token = None
        for key, value in response.cookies.items():
            if key.startswith("download_warning"):
                confirm_token = value
                break

        if confirm_token:
            print("   (large file — bypassing Drive scan page)")
            response = session.get(
                MODEL_URL + f"&confirm={confirm_token}",
                stream=True, timeout=180
            )

        response.raise_for_status()

        total_bytes = int(response.headers.get("content-length", 0))
        downloaded  = 0

        with open(MODEL_PATH, "wb") as f:
            for chunk in response.iter_content(chunk_size=32768):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_bytes:
                        pct = downloaded / total_bytes * 100
                        print(f"\r   {pct:.1f}%  ({downloaded/1024/1024:.1f} MB)", end="", flush=True)

        print(f"\n✅ Model saved → {MODEL_PATH}  ({downloaded/1024/1024:.1f} MB)")
        return True

    except Exception as e:
        print(f"\n❌ Download failed: {e}")
        if os.path.exists(MODEL_PATH):
            os.remove(MODEL_PATH)
        return False

if __name__ == "__main__":
    sys.exit(0 if download_model() else 1)