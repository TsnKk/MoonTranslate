"""Local LaMa manga inference. Alter only the requested pixels."""
import threading
import os
import cv2
import numpy as np
from prepare_engine import ROOT

_session=None
_lock=threading.Lock()
_backend='not loaded'


def backend():
    return _backend


def cpu_session():
    global _backend
    import onnxruntime as ort
    options=ort.SessionOptions()
    options.intra_op_num_threads=4
    options.inter_op_num_threads=1
    session=ort.InferenceSession(str(ROOT/'inpainting/lama-manga.onnx'),
        sess_options=options,providers=['CPUExecutionProvider'])
    _backend='CPU'
    return session


def create_session():
    global _backend
    from inpaint_gpu import available as gpu_available,DmlSession
    if os.environ.get('MOONTRANSLATE_INPAINT','auto')!='cpu' and gpu_available():
        try:
            session=DmlSession();_backend='DirectML GPU'
            return session
        except Exception as exc:print('INPAINT GPU unavailable; using CPU:',str(exc),flush=True)
    return cpu_session()

def available():
    return (ROOT/'inpainting/lama-manga.onnx').is_file()

def warmup():
    if not available():return
    if _session is not None:return
    mask=np.zeros((64,64),np.uint8);mask[24:40,24:40]=255
    inpaint(np.full((64,64,3),200,np.uint8),mask)

def inpaint(image,mask):
    global _session
    if not np.any(mask): return image.copy()
    if not available(): return cv2.inpaint(image,mask,3,cv2.INPAINT_TELEA)
    source=image
    ys,xs=np.where(mask>0)
    margin=max(48,min(128,round(max(np.ptp(xs)+1,np.ptp(ys)+1)*.6)))
    left,top=max(0,int(xs.min())-margin),max(0,int(ys.min())-margin)
    right,bottom=min(image.shape[1],int(xs.max())+margin+1),min(image.shape[0],int(ys.max())+margin+1)
    image=image[top:bottom,left:right]
    mask=mask[top:bottom,left:right]
    with _lock:
        if _session is None:
            _session=create_session()
        h,w=image.shape[:2]
        scale=min(1,512/max(h,w))
        sw,sh=max(1,round(w*scale)),max(1,round(h*scale))
        rgb=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        rgb=cv2.resize(rgb,(sw,sh),interpolation=cv2.INTER_AREA)
        small_mask=cv2.resize(mask,(sw,sh),interpolation=cv2.INTER_NEAREST)
        rgb=cv2.copyMakeBorder(rgb,0,512-sh,0,512-sw,cv2.BORDER_REFLECT_101)
        small_mask=cv2.copyMakeBorder(small_mask,0,512-sh,0,512-sw,cv2.BORDER_CONSTANT,value=0)
        inputs={
            'image':np.ascontiguousarray(rgb.transpose(2,0,1)[None],dtype=np.float32)/255,
            'mask':(small_mask[None,None]>0).astype(np.float32)}
        try:prediction=_session.run(None,inputs)
        except Exception as exc:
            from inpaint_gpu import DmlSession
            if not isinstance(_session,DmlSession):raise
            _session.close()
            print('INPAINT GPU failed; using CPU:',str(exc),flush=True)
            _session=cpu_session();prediction=_session.run(None,inputs)
        output=prediction[0][0].transpose(1,2,0)
        if not np.isfinite(output).all(): raise RuntimeError('LaMa produced invalid pixels')
        output=np.clip(output*255,0,255).astype(np.uint8)
        restored=cv2.resize(output[:sh,:sw],(w,h),interpolation=cv2.INTER_CUBIC)
        restored=cv2.cvtColor(restored,cv2.COLOR_RGB2BGR)
        result=source.copy()
        result[top:bottom,left:right][mask>0]=restored[mask>0]
        return result
