import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
import unittest
import numpy as np
from PySide6.QtWidgets import QApplication
from app import fit_text,word_wrapped_text
from core import Reader

class ThaiWrapTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app=QApplication.instance() or QApplication([])

    def test_word_stays_on_one_line_by_shrinking(self):
        layout,height=fit_text('ความรู้สึก','Leelawadee UI',48,70,100,1)
        self.assertEqual(layout.lineCount(),1)
        self.assertLessEqual(layout.lineAt(0).naturalTextWidth(),70.1)
        self.assertLess(layout.font().pixelSize(),48)

    def test_newlines_are_reflowed_and_content_retained(self):
        text=word_wrapped_text('ฉันรู้สึก\nว่ารู้เหตุผลแล้ว')
        self.assertEqual(text.replace('\u200b','').replace('\u2060',''),'ฉันรู้สึก ว่ารู้เหตุผลแล้ว')

    def test_balanced_lines_keep_contents_inside_the_box(self):
        from app import layout_text,load_fonts
        from PySide6.QtGui import QFont
        load_fonts()
        text='ฉันจะรวบรวมทุกอย่างแล้วเอามาให้ทีหลัง'
        font=QFont('Tahoma');font.setPixelSize(22)
        baseline,_=layout_text(text,font,180)
        balanced,height=fit_text(text,'Tahoma',22,180,180)
        def spread(layout):
            widths=[layout.lineAt(i).naturalTextWidth() for i in range(layout.lineCount())]
            return max(widths)-min(widths)
        self.assertLessEqual(spread(balanced),spread(baseline)+.1)
        self.assertEqual(balanced.text(),baseline.text())
        self.assertLessEqual(height,180)
        for i in range(balanced.lineCount()):
            line=balanced.lineAt(i)
            self.assertGreaterEqual(line.x(),0)
            self.assertLessEqual(line.x()+line.width(),180.1)

    def test_art_background_not_rejected_by_old_filter(self):
        reader=Reader.__new__(Reader)
        polygon=np.array([[20,20],[140,20],[140,100],[20,100]],float)
        reader.detect=lambda image:[(polygon,'I HAVE A FEELING',.99)]
        image=np.random.default_rng(2).integers(0,256,(120,180,3),dtype=np.uint8)
        blocks=reader.read(image,'manga','English',True)
        self.assertEqual(len(blocks),1)
        self.assertEqual(blocks[0].text,'I HAVE A FEELING')

if __name__ == '__main__': unittest.main()
