"""Bounded crop-only OCR verification; never spell-correct names by dictionary."""
from difflib import SequenceMatcher
import re
import time
import cv2
import numpy as np


def canonical(text):
    return re.sub(r'\s+','',text).casefold()


def english_spacing(text):
    # Only common closed-class pronoun + verb forms, in an actual sentence.
    # Do not segment arbitrary words/names, and never alter any source letter.
    if len(re.findall(r'[A-Za-z]+',text))<3:return text
    pronouns='I|YOU|WE|THEY|HE|SHE|IT'
    verbs='AM|IS|ARE|WAS|WERE|HAVE|HAS|HAD|DO|DOES|DID|CAN|COULD|WILL|WOULD|SHOULD|MUST|KNOW|THINK|WANT|NEED|FEEL|LIKE|LOVE|HATE|MEAN|SAID|TOLD|SAW|SEE|HEARD|HOPE|GUESS|BELIEVE'
    return re.sub(r'\b('+pronouns+r')('+verbs+r')\b',r'\1 \2',text)


def non_latin_consensus(score,candidates):
    if score>=.75:return False
    readings=[canonical(t) for t,c in candidates if c>=.8 and not re.search('[A-Za-z]',t) and
              re.search(r'[0-9\u3040-\u30ff\u3400-\u9fff]',t)]
    return len(readings)>=2 and len(set(readings))==1


def choose_reading(original,score,candidates):
    """Require similar content and either clear confidence gain or agreement."""
    base=canonical(original)
    valid=[(text,float(conf)) for text,conf in candidates if text.strip() and
           SequenceMatcher(None,base,canonical(text)).ratio()>=.65]
    for text,conf in sorted(valid,key=lambda x:x[1],reverse=True):
        support=sum(canonical(other)==canonical(text) for other,_ in valid)
        if conf>=.88 and conf>=score+.06 and (support>=2 or conf>=.97):
            # Retain better word boundaries from a matching English result.
            matching=[t for t,c in valid if canonical(t)==canonical(text) and c>=.85]
            return max(matching,key=lambda t:t.count(' ')),conf
    return original,score


def refine_detections(image,detections,engine,language,limit=6,budget=.6,alternate_engine=None):
    output=list(detections);start=time.monotonic();attempts=0;changes=0
    suspicious=[]
    for i,(polygon,text,score) in enumerate(output):
        if score<.35 or not re.search(r'[A-Za-z\u3040-\u30ff\u3400-\u9fff]',text):continue
        if language=='English' and not re.search('[A-Za-z]',text):continue
        if score<.96 or re.search(r'[A-Z]{2}[a-z]|[a-z][A-Z]{2}',text):suspicious.append(i)
    for i in sorted(suspicious,key=lambda i:output[i][2])[:limit]:
        if time.monotonic()-start>=budget:break
        polygon,text,score=output[i]
        low,high=np.min(polygon,axis=0),np.max(polygon,axis=0)
        if high[1]-low[1]>2*(high[0]-low[0]):continue
        pad=max(3,round((high[1]-low[1])*.12))
        a,b=np.maximum(0,np.floor(low).astype(int)-pad)
        c,d=np.minimum(image.shape[1::-1],np.ceil(high).astype(int)+pad)
        crop=image[b:d,a:c]
        if not crop.size:continue
        h,w=crop.shape[:2]
        gray=cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY)
        border=np.r_[gray[0],gray[-1]];bg=int(np.median(border))
        if language=='English':
            # A second shape hypothesis helps slanted comic lettering.
            shear=-.2
            alternate=cv2.warpAffine(crop,np.array([[1,shear,-shear*h/2+pad],[0,1,0]],np.float32),
                (w+2*pad,h),borderValue=(bg,bg,bg))
        else:
            _,binary=cv2.threshold(gray,0,255,cv2.THRESH_BINARY|cv2.THRESH_OTSU)
            alternate=cv2.cvtColor(binary,cv2.COLOR_GRAY2BGR)
        candidates=[]
        for variant in (crop,alternate):
            result=engine(variant,use_det=False,use_cls=False,use_rec=True)
            if result.txts and result.scores:
                value=result.txts[0]
                if language=='English' and re.search(r'[\u3040-\u30ff\u3400-\u9fff]',value):continue
                candidates.append((value,result.scores[0]))
        attempts+=1
        if language=='English' and score<.75:
            if non_latin_consensus(score,candidates):
                output[i]=(polygon,'',score);changes+=1;continue
            if alternate_engine is not None and time.monotonic()-start<budget:
                foreign=[]
                for variant in (crop,alternate):
                    result=alternate_engine(variant,use_det=False,use_cls=False,use_rec=True)
                    if result.txts and result.scores:foreign.append((result.txts[0],result.scores[0]))
                if non_latin_consensus(score,foreign):
                    output[i]=(polygon,'',score);changes+=1;continue
        revised,confidence=choose_reading(text,float(score),candidates)
        changes+=revised!=text
        output[i]=(polygon,revised,confidence)
    return output,{'checked':attempts,'changed':changes,'seconds':round(time.monotonic()-start,3)}
