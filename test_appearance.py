"""Offline UI regressions for reference-inspired appearance and region selection."""
import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
import tempfile
from pathlib import Path
import unittest
import numpy as np
from PySide6.QtCore import QObject,Signal,Qt,QRect,QPoint
from PySide6.QtGui import QImage
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication
import app as ui
from core import Block

class FakeWorker(QObject):
    ready=Signal(str);result=Signal(object,object,float);failed=Signal(object,str);stage=Signal(object,str)
    prepared=Signal(object,str)
    def cancel(self):pass
    def submit(self,job):self.last_job=job

class AppearanceTests(unittest.TestCase):
    def test_warmup_waits_for_matching_model_and_survives_region_selection(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as directory,patch.object(ui,'ROOT',Path(directory)),patch.object(ui,'Worker',FakeWorker):
            window=ui.MainWindow()
            window.backend_ready=True
            window.prepare_selected_model()
            self.assertFalse(window.engine_ready)
            self.assertEqual(window.worker.last_job[0],'prepare')
            key=window.worker.last_job[1:]
            window.on_prepared(('stale-model',key[1]),'')
            self.assertFalse(window.engine_ready)
            from unittest.mock import Mock
            window.worker.cancel=Mock()
            window.stop()
            window.worker.cancel.assert_not_called()
            window.on_prepared(key,'')
            self.assertTrue(window.engine_ready)
            window.close()

    @classmethod
    def setUpClass(cls):
        cls.qt=QApplication.instance() or QApplication([])
        ui.load_fonts()

    def pixels(self,overlay):
        image=overlay.grab().toImage().convertToFormat(QImage.Format_RGBA8888)
        return np.frombuffer(image.bits(),np.uint8).reshape(image.height(),image.bytesPerLine())[:,:image.width()*4].reshape(image.height(),image.width(),4).copy()

    def test_manga_keeps_narrow_source_bounds(self):
        overlay=ui.Overlay();overlay.resize(300,200);overlay.mode='manga'
        overlay.blocks=[Block((80,30,120,170),'text','นี่คือประโยคทดสอบภาษาไทยที่มีวรรณยุกต์')]
        alpha=self.pixels(overlay)[:,:,3]
        alpha[30:171,80:121]=0
        self.assertFalse(alpha.any())
        overlay.close()

    def test_compact_plate_and_empty_frame(self):
        overlay=ui.Overlay();overlay.resize(600,180)
        self.assertFalse(self.pixels(overlay)[:,:,3].any())
        overlay.blocks=[Block((0,0,600,180),'test','สวัสดี')]
        pixels=self.pixels(overlay)
        self.assertEqual(int(pixels[5,5,3]),0)
        self.assertGreater(int(pixels[90,300,3]),0)
        overlay.close()

    def test_selector_keyboard_and_confirmation(self):
        selector=ui.Selector(self.qt.primaryScreen(),QRect(80,90,200,100));selector.show()
        selected=[];selector.selected.connect(selected.append)
        QTest.keyClick(selector,Qt.Key_Right)
        QTest.keyClick(selector,Qt.Key_Down,Qt.ShiftModifier)
        self.assertEqual(selector.selection,QRect(81,100,200,100))
        QTest.keyClick(selector,Qt.Key_Return)
        self.assertEqual(selected,[QRect(81,100,200,100)])

    def test_preferences_and_live_preview(self):
        old_root,old_worker=ui.ROOT,ui.Worker
        with tempfile.TemporaryDirectory() as directory:
            ui.ROOT=Path(directory);ui.Worker=FakeWorker
            try:
                window=ui.MainWindow()
                self.assertFalse(window.bubble_only.isChecked())
                window.font_size.setValue(29)
                window.background_opacity.setValue(67)
                window.text_mode.setCurrentIndex(3)
                window.on_models_loaded([ui.MODEL,'test-local-model'],'')
                window.model_picker.setCurrentText('test-local-model')
                window.fragment_picker.setCurrentIndex(1)
                self.assertEqual(window.preview.font_size,29)
                self.assertEqual(window.preview.text_color,'#ffe38a')
                window.close()
                again=ui.MainWindow()
                self.assertEqual(again.overlay.font_size,29)
                self.assertEqual(again.overlay.opacity,67)
                self.assertEqual(again.overlay.text_color,'#ffe38a')
                self.assertEqual(again.model_picker.currentText(),'test-local-model')
                self.assertEqual(again.fragment_picker.currentData(),'skip')
                again.close()
            finally:
                ui.ROOT,ui.Worker=old_root,old_worker

if __name__=='__main__':unittest.main(verbosity=2)
