"""Reuse the user's existing GGUF with an owned, loopback-only llama server."""
import atexit
import socket
import subprocess
import time
from pathlib import Path

import requests
from prepare_engine import ROOT, API

GEMMA = 'gemma-4-E4B-it (local GGUF)'
BASE = Path('C:/local-game-subs')
MODEL_PATH = BASE / 'models/gemma-4-E4B-it-Q4_K_M.gguf'
SERVER_PATH = BASE / 'llama.cpp/vulkan/llama-server.exe'


def local_models():
    return [GEMMA] if MODEL_PATH.is_file() and SERVER_PATH.is_file() else []


class GemmaBackend:
    def __init__(self):
        self.process = None
        self.log = None
        self.url = None
        self.session = requests.Session()
        self.session.trust_env = False
        atexit.register(self.close)

    def close(self):
        if self.process is not None:
            if self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=5)
            self.process = None
        if self.log:
            self.log.close()
            self.log = None

    def start(self, cancelled):
        if self.process is not None and self.process.poll() is None:
            return
        self.close()
        if not local_models():
            raise RuntimeError('ไม่พบไฟล์ Gemma หรือ llama-server ใน C:/local-game-subs')
        # Release only models in MoonTranslate's dedicated Ollama instance.
        try:
            response = self.session.get(API+'/api/ps', timeout=5)
            response.raise_for_status()
            for model in response.json().get('models', []):
                response = self.session.post(API+'/api/generate',
                    json={'model':model['name'], 'keep_alive':0}, timeout=30)
                response.raise_for_status()
        except requests.ConnectionError:
            pass  # Gemma can also run without Ollama.
        with socket.socket() as listener:
            listener.bind(('127.0.0.1', 0))
            port = listener.getsockname()[1]
        self.url = f'http://127.0.0.1:{port}'
        ROOT.mkdir(parents=True, exist_ok=True)
        self.log = (ROOT/'gemma-engine.log').open('ab')
        try:
            self.process = subprocess.Popen([
                str(SERVER_PATH), '-m', str(MODEL_PATH), '--host', '127.0.0.1',
                '--port', str(port), '--alias', 'moontranslate-gemma',
                '-c', '4096', '-np', '1', '-ngl', '99', '-fa', 'on', '--reasoning', 'off'],
                cwd=str(SERVER_PATH.parent), stdout=self.log, stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            deadline = time.monotonic()+120
            while time.monotonic() < deadline:
                if cancelled():
                    raise RuntimeError('ยกเลิกการโหลดโมเดล')
                if self.process.poll() is not None:
                    raise RuntimeError('Gemma เริ่มไม่สำเร็จ ดู gemma-engine.log')
                try:
                    response = self.session.get(self.url+'/health', timeout=1)
                    if response.status_code == 200:
                        return
                except requests.RequestException:
                    pass
                time.sleep(.3)
            raise RuntimeError('โหลด Gemma เกิน 120 วินาที ดู gemma-engine.log')
        except Exception:
            self.close()
            raise

    def chat(self, messages, schema, cancelled):
        self.start(cancelled)
        response = self.session.post(self.url+'/v1/chat/completions', json={
            'model':'moontranslate-gemma', 'messages':messages, 'stream':False,
            'temperature':.1, 'max_tokens':1200,
            'chat_template_kwargs':{'enable_thinking':False},
            'response_format':{'type':'json_schema', 'json_schema':{
                'name':'translations', 'strict':True, 'schema':schema}}}, timeout=(5,120))
        response.raise_for_status()
        choice = response.json()['choices'][0]
        if choice.get('finish_reason') == 'length':
            raise RuntimeError('คำแปลยาวเกินขีดจำกัด ลองเลือกพื้นที่เล็กลง')
        return choice['message']['content']
