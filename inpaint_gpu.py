"""Isolated DirectML worker; RapidOCR keeps its own CPU ONNX Runtime."""
import atexit
import os
from pathlib import Path
import queue
import struct
import subprocess
import sys
import threading

from prepare_engine import ROOT

MODEL = ROOT/'inpainting/lama-manga-dml-v1.onnx'
RUNTIME = ROOT/'acceleration/directml'
LIMIT = 8*1024*1024


def available():
    return MODEL.is_file() and (RUNTIME/'onnxruntime/__init__.py').is_file()


def read_packet(stream):
    def exact(size):
        chunks=[]
        while size:
            part=stream.read(size)
            if not part: raise EOFError('DirectML worker disconnected')
            chunks.append(part);size-=len(part)
        return b''.join(chunks)
    size=struct.unpack('<I',exact(4))[0]
    if size>LIMIT: raise ValueError('Invalid DirectML packet size')
    return exact(size)


def write_packet(stream,data):
    if len(data)>LIMIT: raise ValueError('DirectML packet too large')
    stream.write(struct.pack('<I',len(data)));stream.write(data);stream.flush()


class DmlSession:
    def __init__(self):
        self.process=None
        self.responses=queue.Queue(maxsize=2)
        python=Path(sys.executable).with_name('python.exe')
        # stderr can contain native UTF-16 messages; keep it out of binary IPC.
        log=(ROOT/'inpainting/directml.log').open('ab')
        try:
            self.process=subprocess.Popen([str(python),'-u',str(Path(__file__).resolve()),'--serve'],
                stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=log,
                creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        finally:log.close()
        threading.Thread(target=self._receive,daemon=True,name='directml-replies').start()
        try:
            if self._reply(60)!=b'READY': raise RuntimeError('Invalid DirectML greeting')
        except Exception:
            self.close();raise
        atexit.register(self.close)

    def _receive(self):
        try:
            while True:self.responses.put(read_packet(self.process.stdout))
        except Exception as exc:self.responses.put(exc)

    def _reply(self,timeout):
        try:reply=self.responses.get(timeout=timeout)
        except queue.Empty:raise TimeoutError('DirectML worker timed out') from None
        if isinstance(reply,Exception):raise reply
        if reply.startswith(b'ERROR:'):raise RuntimeError(reply.decode('utf-8',errors='replace'))
        return reply

    def run(self,output_names,inputs):
        import numpy as np
        try:
            data=inputs['image'].tobytes()+inputs['mask'].tobytes()
            write_packet(self.process.stdin,data)
            reply=self._reply(30)
            if len(reply)!=3*512*512*4:raise ValueError('Invalid DirectML output size')
            output=np.frombuffer(reply,dtype=np.float32).reshape(1,3,512,512)
            if not np.isfinite(output).all():raise ValueError('Invalid DirectML output pixels')
            return [output]
        except Exception:
            self.close();raise

    def close(self):
        process=self.process
        if process is None:return
        if process.poll() is None:
            process.terminate()
            try:process.wait(timeout=2)
            except subprocess.TimeoutExpired:process.kill();process.wait(timeout=2)
        for stream in (process.stdin,process.stdout):
            if stream:
                try:stream.close()
                except OSError:pass


def serve():
    sys.path.insert(0,str(RUNTIME))
    import numpy as np
    import onnxruntime as ort
    if 'DmlExecutionProvider' not in ort.get_available_providers():
        raise RuntimeError('DirectML provider is unavailable')
    options=ort.SessionOptions()
    options.enable_mem_pattern=False
    options.execution_mode=ort.ExecutionMode.ORT_SEQUENTIAL
    options.intra_op_num_threads=4;options.inter_op_num_threads=1
    options.add_free_dimension_override_by_name('batch',1)
    session=ort.InferenceSession(str(MODEL),options,
        providers=['DmlExecutionProvider','CPUExecutionProvider'])
    write_packet(sys.stdout.buffer,b'READY')
    while True:
        try:data=read_packet(sys.stdin.buffer)
        except EOFError:return
        if len(data)!=4*512*512*4:raise ValueError('Invalid DirectML input size')
        values=np.frombuffer(data,dtype=np.float32)
        result=session.run(None,{'image':values[:3*512*512].reshape(1,3,512,512),
                                 'mask':values[3*512*512:].reshape(1,1,512,512)})[0]
        write_packet(sys.stdout.buffer,result.tobytes())


if __name__=='__main__':
    try:serve()
    except Exception as exc:
        write_packet(sys.stdout.buffer,('ERROR: '+str(exc)).encode('utf-8'))
        raise
