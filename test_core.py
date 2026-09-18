import os
import sys
import json
import time
import tempfile
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from core import Block, changed, group_lines, fingerprint, Reader, Translator
import numpy as np
from PIL import Image,ImageDraw,ImageFont

class LogicTests(unittest.TestCase):
    def test_sound_effect_is_not_joined_to_dialogue(self):
        lines=[Block((10,10,190,150),'STAB'),Block((30,160,180,190),'AND'),Block((20,195,190,225),'NOW WHAT')]
        blocks=group_lines(lines,'manga')
        self.assertEqual(len(blocks),2)
        self.assertIn('AND\nNOW WHAT',[b.text for b in blocks])

    def test_large_page_tile_overlap_deduplicates(self):
        from types import SimpleNamespace
        reader=Reader.__new__(Reader)
        offsets=iter([(0,0),(600,0),(0,600),(600,600)])
        def fake_engine(image,**kwargs):
            x,y=next(offsets)
            box=np.array([[[800-x,800-y],[1000-x,800-y],[1000-x,900-y],[800-x,900-y]]],dtype=float)
            return SimpleNamespace(boxes=box,txts=['same sentence'],scores=[.99])
        reader.engine=fake_engine
        results=reader.detect(np.zeros((2000,2000,3),np.uint8))
        self.assertEqual(len(results),1)
        self.assertEqual(results[0][0].min(axis=0).tolist(),[800,800])

    def test_change_and_blank(self):
        a=np.full((200,700,3),255,np.uint8)
        b=a.copy(); b[30:60,50:180]=0
        self.assertTrue(changed(None,fingerprint(a)))
        self.assertFalse(changed(fingerprint(a),fingerprint(a)))
        self.assertTrue(changed(fingerprint(a),fingerprint(b)))
        self.assertEqual(group_lines([],'game'),[])

    def test_bubble_separation(self):
        lines=[Block((10,10,150,35),'one'),Block((10,40,150,65),'two'),Block((300,10,450,35),'three')]
        blocks=group_lines(lines,'manga')
        self.assertEqual(len(blocks),2)
        self.assertIn('one\ntwo',[b.text for b in blocks])

    def test_vertical_reading_order(self):
        blocks=group_lines([Block((50,10,80,200),'left'),Block((90,10,120,200),'right')],'manga')
        self.assertEqual(len(blocks),1)
        self.assertEqual(blocks[0].text,'right\nleft')

    def test_same_row_baseline_jitter(self):
        blocks=group_lines([Block((230,46,790,99),'。ここで待って'),Block((22,52,225,99),'あなたを守る')],'game')
        self.assertEqual(blocks[0].text,'あなたを守る。ここで待って')

    def test_cancelled_never_translate(self):
        t=Translator()
        class NoNetwork:
            def post(self,*a,**k): raise AssertionError('cancelled job sent')
        t.session=NoNetwork()
        self.assertEqual(t.translate([Block((0,0,10,10),'hello')],'English',{},lambda:True),[])

    def test_cache_includes_glossary(self):
        t=Translator()
        class Response:
            def raise_for_status(self): pass
            def json(self): return {'message':{'content':'{"translations":["สวัสดี"]}'}}
        class Fake:
            calls=0
            def post(self,*a,**k): self.calls+=1; return Response()
        t.session=Fake()
        b=[Block((0,0,50,20),'hello')]
        t.translate(b,'English',{})
        t.translate(b,'English',{})
        self.assertEqual(t.session.calls,1)
        t.translate(b,'English',{'hello':'หวัดดี'})
        self.assertEqual(t.session.calls,2)


def integration():
    output=Path(tempfile.gettempdir())/'MoonTranslate-build'
    output.mkdir(parents=True,exist_ok=True)
    reader=Reader()
    translator=Translator()
    english=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',32)
    japanese=ImageFont.truetype('C:/Windows/Fonts/msgothic.ttc',36)
    cases=[]
    for language,text in [('English','I will protect you. Please stay here.'),('Japanese','あなたを守ります。ここで待っていてください。')]:
        im=Image.new('RGB',(1050,180),'white')
        ImageDraw.Draw(im).text((25,55),text,font=english if language=='English' else japanese,fill='black')
        cases.append((language,language,im,'game'))
    im=Image.new('RGB',(320,460),'white')
    draw=ImageDraw.Draw(im)
    for x,text in [(220,'あなたを守る'),(145,'ここで待って')]:
        for i,char in enumerate(text): draw.text((x,20+i*48),char,font=japanese,fill='black')
    cases.append(('Japanese-vertical','Japanese',im,'manga'))
    results=[]
    for name,language,im,mode in cases:
        im.save(output/(name+'.png'))
        started=time.monotonic()
        blocks=reader.read(np.asarray(im)[:,:,::-1].copy(),mode,language)
        ocr_time=time.monotonic()-started
        assert blocks, name+' OCR empty'
        print(name,'OCR:',[b.text for b in blocks],flush=True)
        translated=translator.translate(blocks,language,{})
        assert all(any('\u0e00'<=c<='\u0e7f' for c in b.translated) for b in translated)
        result={'case':name,'ocr_seconds':round(ocr_time,2),'total_seconds':round(time.monotonic()-started,2),
                'blocks':[vars(b) for b in translated]}
        print(json.dumps(result,ensure_ascii=False),flush=True)
        results.append(result)
    (output/'integration-results.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')

if __name__=='__main__':
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(LogicTests)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful(): sys.exit(1)
    if '--integration' in sys.argv: integration()

