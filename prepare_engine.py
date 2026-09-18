"""Download the official portable Ollama engine and local translation model."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import urllib.request
import zipfile

ROOT = Path(os.environ.get('MOONTRANSLATE_DATA', str(Path(os.environ['LOCALAPPDATA']) / 'MoonTranslate')))
PORT = 11439
API = f'http://127.0.0.1:{PORT}'
MODEL = 'qwen3:4b-instruct'

def request(path, payload=None, timeout=15):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(API + path, data=data, headers={'Content-Type': 'application/json'})
    return urllib.request.urlopen(req, timeout=timeout)

def ensure_server():
    try:
        with request('/api/version') as r:
            return json.load(r)
    except Exception:
        pass
    executable = ROOT / 'engine' / 'ollama.exe'
    if not executable.exists():
        raise RuntimeError('Ollama is not installed. Run Setup.cmd first.')
    env = os.environ.copy()
    env.update(OLLAMA_HOST=f'127.0.0.1:{PORT}', OLLAMA_MODELS=str(ROOT / 'models'),
               OLLAMA_NUM_PARALLEL='1', OLLAMA_MAX_LOADED_MODELS='1', OLLAMA_CONTEXT_LENGTH='2048',
               OLLAMA_NO_CLOUD='1')
    ROOT.mkdir(parents=True, exist_ok=True)
    with (ROOT / 'engine.log').open('ab') as log:
        subprocess.Popen([str(executable), 'serve'], env=env, stdout=log, stderr=log,
                         creationflags=subprocess.CREATE_NO_WINDOW, cwd=str(ROOT))
    for _ in range(60):
        time.sleep(.5)
        try:
            with request('/api/version') as r:
                return json.load(r)
        except Exception:
            continue
    raise RuntimeError('Cannot start local Ollama. See engine.log.')

def install():
    ROOT.mkdir(parents=True, exist_ok=True)
    executable = ROOT / 'engine' / 'ollama.exe'
    if not executable.exists():
        release_url = 'https://api.github.com/repos/ollama/ollama/releases/tags/v0.34.1'
        with urllib.request.urlopen(release_url, timeout=30) as response:
            release = json.load(response)
        asset = next(x for x in release['assets'] if x['name'] == 'ollama-windows-amd64.zip')
        archive = ROOT / 'ollama.zip'
        partial = ROOT / 'ollama.zip.part'
        print('Downloading official Ollama portable engine (about 1.5 GB)...', flush=True)
        total = partial.stat().st_size if partial.exists() else 0
        while total < asset['size']:
            end = min(total + 32 * 1024 * 1024 - 1, asset['size'] - 1)
            req = urllib.request.Request(asset['browser_download_url'], headers={'Range': f'bytes={total}-{end}'})
            with urllib.request.urlopen(req, timeout=120) as response:
                if total and response.status != 206:
                    raise RuntimeError('Server did not honor resumable download')
                if response.status == 206 and not response.headers.get('Content-Range', '').startswith(f'bytes {total}-'):
                    raise RuntimeError('Wrong download range')
                before = total
                with partial.open('ab') as out:
                    while chunk := response.read(4 * 1024 * 1024):
                        out.write(chunk)
                        total += len(chunk)
                if before == total:
                    raise RuntimeError('Empty download response')
                print(f'Engine: {total / asset["size"]:.0%}', flush=True)
        if total != asset['size']:
            raise RuntimeError('Incomplete engine download')
        digest = asset.get('digest')
        if not digest or not digest.startswith('sha256:'):
            raise RuntimeError('Official release has no SHA256 checksum')
        with partial.open('rb') as f:
            actual = hashlib.file_digest(f, 'sha256').hexdigest()
        if actual != digest.split(':', 1)[1]:
            raise RuntimeError('Engine checksum mismatch')
        partial.replace(archive)
        print('Verified checksum. Extracting engine...', flush=True)
        with zipfile.ZipFile(archive) as z:
            destination = (ROOT / 'engine').resolve()
            for member in z.infolist():
                if not (destination / member.filename).resolve().is_relative_to(destination):
                    raise RuntimeError('Unsafe path in archive')
            z.extractall(destination)
    print('Starting local engine...', ensure_server(), flush=True)
    print(f'Downloading translation model {MODEL} (about 2.5 GB)...', flush=True)
    last_percent = -10
    with request('/api/pull', {'model': MODEL, 'stream': True}, timeout=600) as response:
        for line in response:
            event = json.loads(line)
            if event.get('error'):
                raise RuntimeError(event['error'])
            if event.get('total'):
                percent = int(event.get('completed', 0) * 100 / event['total'])
                if percent >= last_percent + 10:
                    print(f'Model: {percent}%', flush=True)
                    last_percent = percent
            else:
                print(event.get('status'), flush=True)
    print('Translation engine ready.', flush=True)

if __name__ == '__main__':
    install()
