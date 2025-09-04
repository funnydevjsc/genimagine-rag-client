import os
import zipfile

def zip_files(files: list[str], destination: str) -> bool:
    try:
        with zipfile.ZipFile(destination, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file in files:
                zipf.write(file, os.path.basename(file))
        return True
    except Exception as e:
        print(f"Zip error: {e}")
        return False