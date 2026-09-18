import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
import unittest
import numpy as np
from PySide6.QtWidgets import QApplication
from app import OBSOutput,Overlay
from core import Block

class OBSOutputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_composite_and_clear(self):
        output = OBSOutput()
        overlay = Overlay()
        overlay.mode = 'manga'
        overlay.scale = 1.5
        overlay.background_mode = 'custom'
        overlay.custom_color = '#ff0000'
        overlay.blocks = [Block((30,30,120,90),'hello','สวัสดี')]
        source = np.full((150,200,3),255,np.uint8)
        output.set_frame(source,overlay)
        self.assertEqual(output.image.width(),200)
        self.assertEqual(output.image.pixelColor(5,5).name(),'#ffffff')
        self.assertEqual(output.image.pixelColor(31,31).name(),'#ff0000')
        self.assertTrue(np.all(source==255))
        output.set_frame(source)
        self.assertEqual(output.image.pixelColor(31,31).name(),'#ffffff')
        output.clear()
        self.assertTrue(output.image.isNull())
        output.close();overlay.close()

if __name__ == '__main__': unittest.main()
