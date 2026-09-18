"""Install the pinned, local manga inpainting model (207 MB)."""
import hashlib
from pathlib import Path
import requests
from prepare_engine import ROOT

REVISION = 'b55497aadbfcb9740e1ed16f008268d71b4f3f79'
SHA256 = '4512adab295ee5a5e02ccd1bdf8d45dccbac88309d9cff1532ffd5de876f02a4'
TARGET = ROOT/'inpainting/lama-manga.onnx'

def install():
    TARGET.parent.mkdir(parents=True,exist_ok=True)
    if TARGET.exists():
        with TARGET.open('rb') as existing:
            if hashlib.file_digest(existing,'sha256').hexdigest()==SHA256: return
    url=f'https://huggingface.co/mayocream/lama-manga-onnx/resolve/{REVISION}/lama-manga.onnx'
    temporary=TARGET.with_suffix('.download')
    digest=hashlib.sha256()
    with requests.get(url,stream=True,timeout=(15,60)) as response:
        response.raise_for_status()
        with temporary.open('wb') as output:
            for chunk in response.iter_content(1024*1024):
                output.write(chunk);digest.update(chunk)
    if digest.hexdigest()!=SHA256:
        raise RuntimeError('Inpainting model checksum mismatch')
    temporary.replace(TARGET)
    print('Verified model:',TARGET,flush=True)

if __name__=='__main__': install()
