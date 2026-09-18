"""MoonTranslate — Windows local realtime screen translation overlay."""
import ctypes
from ctypes import wintypes
import json
import os
from pathlib import Path
import queue
import sys
import threading
import time
import traceback
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
from pythainlp.tokenize import word_tokenize

import cv2
import mss
import numpy as np
from PySide6.QtCore import (Qt, QObject, Signal, QThread, QTimer, QPoint, QPointF, QRect,
                            QRectF, QAbstractNativeEventFilter)
from PySide6.QtGui import (QColor, QPainter, QPen, QFont, QTextLayout, QTextOption,
                           QCursor, QFontDatabase)
from PySide6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QComboBox, QSpinBox, QFormLayout, QPlainTextEdit,
    QGroupBox, QMessageBox, QCheckBox, QColorDialog, QScrollArea)

from core import Reader, Translator, Block, fingerprint, changed, merge_cleaned
from manga_inpaint import warmup as warmup_inpainting,backend as inpainting_backend
from PySide6.QtGui import QImage
from prepare_engine import ROOT, MODEL, ensure_server, request
from local_models import local_models

APP_DIR = Path(__file__).resolve().parent


class Selector(QWidget):
    selected = Signal(QRect)
    cancelled = Signal()

    def __init__(self, screen, initial=None):
        super().__init__()
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setGeometry(screen.geometry())
        self.setCursor(Qt.CrossCursor)
        self.start = None
        self.end = None
        self.selection = QRect(initial) if initial is not None else QRect(self.width()//5,self.height()*3//4,self.width()*3//5,self.height()//6)
        self.moving = False
        self.origin = QPoint()

    def paintEvent(self, event):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(8, 12, 25, 145))
        p.setPen(QColor('white'))
        p.setFont(QFont('Leelawadee UI', 16))
        p.drawText(self.rect().adjusted(24,24,-24,-24), Qt.AlignTop | Qt.AlignHCenter,
                   'ลากนอกกรอบเพื่อวาดใหม่ • ลากในกรอบเพื่อย้าย • ลูกศรขยับ • Enter ยืนยัน • Esc ยกเลิก')
        if self.selection.isValid():
            rect = self.selection
            p.setCompositionMode(QPainter.CompositionMode_Clear)
            p.fillRect(rect, Qt.transparent)
            p.setCompositionMode(QPainter.CompositionMode_SourceOver)
            p.setPen(QPen(QColor('#7dd3fc'), 2))
            p.drawRect(rect)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.start = event.position().toPoint()
            self.end = self.start
            self.moving = self.selection.contains(self.start)
            self.origin = self.selection.topLeft()
            self.update()

    def mouseMoveEvent(self, event):
        if self.start is not None:
            self.end = event.position().toPoint()
            if self.moving:
                self.selection.moveTopLeft(self.origin+self.end-self.start)
                self.clamp_selection()
            else:
                self.selection = QRect(self.start,self.end).normalized().intersected(self.rect())
            self.update()

    def mouseReleaseEvent(self, event):
        if self.start is None:
            return
        self.start = None
        self.update()

    def clamp_selection(self):
        self.selection.moveLeft(max(0,min(self.width()-self.selection.width(),self.selection.x())))
        self.selection.moveTop(max(0,min(self.height()-self.selection.height(),self.selection.y())))

    def accept_selection(self):
        rect = self.selection.intersected(self.rect())
        self.hide()
        if rect.width() >= 40 and rect.height() >= 25:
            self.selected.emit(rect)
        else:
            self.cancelled.emit()
        self.close()

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key_Return,Qt.Key_Enter):
            self.accept_selection()
            return
        steps = {Qt.Key_Left:(-1,0),Qt.Key_Right:(1,0),Qt.Key_Up:(0,-1),Qt.Key_Down:(0,1)}
        if event.key() in steps:
            dx,dy=steps[event.key()]
            step=10 if event.modifiers() & Qt.ShiftModifier else 1
            self.selection.translate(dx*step,dy*step)
            self.clamp_selection();self.update();return
        if event.key() == Qt.Key_C:
            self.selection.moveCenter(self.rect().center());self.clamp_selection();self.update();return
        if event.key() == Qt.Key_Escape:
            self.hide()
            self.cancelled.emit()
            self.close()


# Thai vowel signs / tone marks that attach to the PRECEDING consonant and
# must never start a new line on their own (e.g. เมา must not split into
# "เม" + "า"). Includes the spacing vowels ะ า ำ as well as the combining marks.
_THAI_TRAILING = '\u0e30\u0e31\u0e32\u0e33\u0e34\u0e35\u0e36\u0e37\u0e38\u0e39\u0e47\u0e48\u0e49\u0e4a\u0e4b\u0e4c\u0e4d\u0e4e'
# Leading vowels that attach to the FOLLOWING consonant and must never end a
# line on their own (e.g. เก must not split into "เ" + "ก").
_THAI_LEADING = '\u0e40\u0e41\u0e42\u0e43\u0e44'


def protect_thai_clusters(text):
    """Insert a word joiner wherever breaking the line would separate a Thai
    base consonant from a vowel/tone mark that must stay attached to it."""
    out = []
    for i, ch in enumerate(text):
        if i and (ch in _THAI_TRAILING or text[i-1] in _THAI_LEADING):
            out.append('\u2060')  # WORD JOINER: allowed to sit here, forbids a line break here
        out.append(ch)
    return ''.join(out)


def numpy_to_qimage(bgr):
    rgb = np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    image = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.shape[1]*3, QImage.Format_RGB888)
    return image.copy()  # detach from the numpy buffer before it goes out of scope


@lru_cache(maxsize=1000)
def word_wrapped_text(text):
    # Explicit dictionary boundaries: Qt on Windows may otherwise wrap Thai
    # between arbitrary consonants. Joiners keep each dictionary word intact.
    tokens = word_tokenize(' '.join(text.split()),engine='newmm',keep_whitespace=True)
    return '\u200b'.join('\u2060'.join(token) if not token.isspace() else token for token in tokens)


def layout_text(text, font, width):
    text = word_wrapped_text(text)
    layout = QTextLayout(text, font)
    option = QTextOption()
    option.setWrapMode(QTextOption.WordWrap)
    option.setAlignment(Qt.AlignHCenter)
    layout.setTextOption(option)
    layout.beginLayout()
    y = 0.0
    while True:
        line = layout.createLine()
        if not line.isValid():
            break
        line.setLineWidth(max(1,width))
        line.setPosition(QPoint(0, round(y)))
        y += line.height() + 1
    layout.endLayout()
    return layout, y


def fit_text(text, family, maximum, width, height, max_lines=None):
    for size in range(maximum,5,-1):
        font = QFont(family)
        font.setPixelSize(size)
        layout,used_height = layout_text(text,font,width)
        used_width = max((layout.lineAt(i).naturalTextWidth() for i in range(layout.lineCount())),default=0)
        if (used_height<=height and used_width<=width+.1 and
            (max_lines is None or layout.lineCount()<=max_lines)):
            # Balance a paragraph without changing its font size, line count,
            # or dictionary word boundaries. Avoid a lone short last line.
            if layout.lineCount()>1:
                def raggedness(candidate):
                    widths=[candidate.lineAt(i).naturalTextWidth() for i in range(candidate.lineCount())]
                    mean=sum(widths)/len(widths)
                    return sum((w-mean)**2 for w in widths)/max(1,mean**2*len(widths))
                best=raggedness(layout)
                for ratio in (.95,.90,.85,.80,.75,.70):
                    candidate,ch=layout_text(text,font,width*ratio)
                    if candidate.lineCount()!=layout.lineCount() or ch>height:continue
                    if any(candidate.lineAt(i).naturalTextWidth()>width*ratio+.1 for i in range(candidate.lineCount())):continue
                    score=raggedness(candidate)
                    if score<best-.015:
                        for i in range(candidate.lineCount()):
                            line=candidate.lineAt(i)
                            line.setPosition(line.position()+QPointF(width*(1-ratio)/2,0))
                        layout,used_height,best=candidate,ch,score
            return layout,used_height
    return layout,used_height


class Overlay(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool |
                            Qt.WindowTransparentForInput | Qt.WindowDoesNotAcceptFocus)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_ShowWithoutActivating)
        self.blocks = []
        self.scale = 1.0
        self.mode = 'game'
        self.font_size = 22
        self.enabled = True
        self.background_mode = 'auto'
        self.custom_color = '#fffdf6'
        self.opacity = 100
        self.font_family = 'Leelawadee UI'
        self.text_color = None
        self.outline = 1
        self.compact = True

    def exclude_from_capture(self):
        # Required to avoid OCR repeatedly reading our own translated overlay.
        fn = ctypes.windll.user32.SetWindowDisplayAffinity
        fn.argtypes = [wintypes.HWND, wintypes.DWORD]
        fn.restype = wintypes.BOOL
        return bool(fn(int(self.winId()), 0x11))

    def clear(self):
        self.blocks = []
        self.update()

    def paintEvent(self, event):
        if not self.enabled:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        for block in self.blocks:
            x1,y1,x2,y2 = [v/self.scale for v in block.rect]
            if self.mode == 'game':
                box = QRectF(2, 2, self.width()-4, self.height()-4)
            else:
                # Never widen a manga text box beyond the detected source.
                # Narrow columns must fit by wrapping/shrinking, not by
                # painting over adjacent bubble borders or artwork.
                box = QRectF(x1,y1,x2-x1,y2-y1).intersected(QRectF(self.rect()))
            if self.background_mode == 'auto':
                background = QColor(*block.background) if self.mode == 'manga' else QColor('#101827')
            else:
                background = QColor({'white':'#ffffff','dark':'#101827',
                                     'custom':self.custom_color}.get(self.background_mode,self.custom_color))
            light = .2126*background.red()+.7152*background.green()+.0722*background.blue() > 145
            background.setAlpha(round(255*self.opacity/100))
            if self.mode == 'manga' and block.layout_rect is not None:
                a,b,c,d = [v/self.scale for v in block.layout_rect]
                text_box = QRectF(a,b,c-a,d-b)
            else:
                text_box = box
            inside = text_box.adjusted(2,2,-2,-2)
            if inside.width() < 1 or inside.height() < 1:
                continue
            # Thai and English need different line counts; the real balloon
            # bounds, not the source newline count, constrain the paragraph.
            max_lines = None
            layout,height = fit_text(block.translated,self.font_family,self.font_size,
                                     inside.width(),inside.height(),max_lines)
            if self.mode == 'game' and self.compact:
                content_width = max((layout.lineAt(i).naturalTextWidth() for i in range(layout.lineCount())),default=0)
                tight = QRectF(0,0,min(box.width(),content_width+18),min(box.height(),height+12))
                tight.moveCenter(box.center())
                box = tight
            painter.setPen(Qt.NoPen)
            if self.mode == 'manga' and self.background_mode == 'auto' and block.background_image is not None:
                painter.setOpacity(self.opacity/100)
                painter.drawImage(box, numpy_to_qimage(block.background_image))
                painter.setOpacity(1.0)
            else:
                painter.setBrush(background)
                painter.drawRect(box)
            painter.save()
            painter.setClipRect(inside)
            position = inside.topLeft() + QPointF(0,max(0,(inside.height()-height)/2))
            ink=QColor(self.text_color) if self.text_color else QColor('#172033' if light else '#f8fafc')
            if self.outline:
                painter.setPen(QColor('#101827' if ink.lightness()>128 else '#ffffff'))
                for dx,dy in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(1,-1),(-1,1),(1,1)]:
                    layout.draw(painter,position+QPointF(dx*self.outline,dy*self.outline))
            painter.setPen(ink)
            layout.draw(painter,position)
            painter.restore()


class Capture(QThread):
    invalidated = Signal(object)
    stable = Signal(object, object)
    failed = Signal(str)

    def __init__(self, region, session, settle_ms):
        super().__init__()
        self.region = region
        self.session = session
        self.settle = settle_ms / 1000
        self.stopping = threading.Event()
        self.force = threading.Event()

    def run(self):
        revision = 0
        previous = None
        anchor = None
        last_motion = time.monotonic()
        sent = False
        try:
            with mss.mss() as screen:
                while not self.stopping.is_set():
                    frame = np.asarray(screen.grab(self.region))[:,:,:3].copy()
                    current = fingerprint(frame)
                    force = self.force.is_set()
                    if force:
                        self.force.clear()
                    # Compare consecutive frames and the last submitted frame:
                    # slow typewriter animations must not escape detection.
                    motion = changed(previous,current)
                    drift = sent and changed(anchor,current)
                    if motion or drift or force:
                        revision += 1
                        self.invalidated.emit((self.session,revision))
                        last_motion = time.monotonic()
                        sent = False
                    previous = current
                    if not sent and time.monotonic()-last_motion >= self.settle:
                        self.stable.emit((self.session,revision),frame)
                        anchor = current.copy()
                        sent = True
                    self.stopping.wait(.20)
        except Exception as exc:
            self.failed.emit(str(exc))


class Worker(QObject):
    ready = Signal(str)
    prepared = Signal(object,str)
    result = Signal(object, object, float)
    failed = Signal(object, str)
    stage = Signal(object, str)

    def __init__(self):
        super().__init__()
        self.jobs = queue.Queue(maxsize=1)
        self.active_cancel = threading.Event()
        self.closed = False
        threading.Thread(target=self.run,daemon=True).start()

    def cancel(self):
        self.active_cancel.set()
        try:
            while True:
                self.jobs.get_nowait()
        except queue.Empty:
            pass

    def submit(self, job):
        self.cancel()
        cancel = threading.Event()
        self.active_cancel = cancel
        self.jobs.put_nowait((job,cancel))

    def run(self):
        try:
            available = local_models()
            try:
                ensure_server()
                with request('/api/tags') as response:
                    available += [m['name'] for m in json.load(response)['models']]
            except Exception:
                if not available: raise
            if not available:
                raise RuntimeError('ยังไม่มีโมเดลแปลภาษา ให้เปิด Setup.cmd ก่อน')
            reader = Reader()
            translator = Translator()
            cleanup_pool=ThreadPoolExecutor(max_workers=1,thread_name_prefix='manga-background')
            self.ready.emit('พร้อมใช้งาน • แปลบนเครื่อง')
        except Exception as exc:
            self.ready.emit('ERROR: '+str(exc))
            return
        while not self.closed:
            try:
                job,cancel = self.jobs.get(timeout=.5)
            except queue.Empty:
                continue
            if len(job)==3 and job[0]=='prepare':
                _,model,language=job
                try:
                    reader.prepare(language)
                    background_ready=cleanup_pool.submit(warmup_inpainting)
                    translator.model=model
                    translator.warmup(language,lambda:cancel.is_set() or self.closed)
                    background_ready.result()
                    print('INPAINT_BACKEND '+inpainting_backend(),flush=True)
                    if not cancel.is_set() and not self.closed:self.prepared.emit((model,language),'')
                except Exception as exc:
                    if not cancel.is_set() and not self.closed:self.prepared.emit((model,language),str(exc))
                continue
            token,image,mode,language,glossary,bubble_only,model,fragment_policy = job
            if cancel.is_set():
                continue
            started = time.monotonic()
            try:
                self.stage.emit(token,'กำลังอ่านข้อความ…')
                blocks = reader.read(image,mode,language,bubble_only,clean_background=(mode!='manga'))
                read_finished=time.monotonic()
                if cancel.is_set():
                    continue
                original_blocks=blocks
                cleanup=(cleanup_pool.submit(reader.clean,image,blocks,lambda cancel=cancel:cancel.is_set() or self.closed)
                         if mode=='manga' else None)
                self.stage.emit(token,f'กำลังแปล {len(blocks)} กลุ่มและเติมพื้นหลังพร้อมกัน…')
                translator.model = model
                blocks = translator.translate(blocks,language,glossary,cancel.is_set,fragment_policy)
                translation_finished=time.monotonic()
                if cleanup is not None:
                    cleaned=cleanup.result()
                    if cancel.is_set() or self.closed:continue
                    blocks=merge_cleaned(original_blocks,cleaned,blocks)
                if not cancel.is_set() and not self.closed:
                    elapsed=time.monotonic()-started
                    print('PERFORMANCE '+json.dumps({'groups':len(blocks),'ocr_seconds':round(read_finished-started,3),
                          'ocr_refinement':getattr(reader,'refinement_stats',{}),
                          'translation_seconds':round(translation_finished-read_finished,3),
                          'remaining_background_seconds':round(time.monotonic()-translation_finished,3),
                          'total_seconds':round(elapsed,3),'model':model}),flush=True)
                    self.result.emit(token,blocks,elapsed)
            except Exception as exc:
                if not cancel.is_set() and not self.closed:
                    self.failed.emit(token,str(exc))
        translator.close()
        cleanup_pool.shutdown(wait=False,cancel_futures=True)


class Hotkeys(QAbstractNativeEventFilter):
    def __init__(self, owner):
        super().__init__()
        self.owner = owner

    def nativeEventFilter(self, event_type, message):
        msg = wintypes.MSG.from_address(int(message))
        if msg.message == 0x0312:
            if msg.wParam == 1:
                QTimer.singleShot(0,self.owner.toggle)
            elif msg.wParam == 2:
                QTimer.singleShot(0,self.owner.select_region)
            elif msg.wParam == 3:
                QTimer.singleShot(0,self.owner.refresh)
            return True,0
        return False,0


class OBSOutput(QWidget):
    """Ordinary capturable window; never disable exclusion on the screen overlay."""
    def __init__(self):
        super().__init__()
        self.setWindowTitle('MoonTranslate OBS')
        self.resize(640,480)
        self.image = QImage()

    def set_frame(self, frame, overlay=None):
        self.image = numpy_to_qimage(frame)
        if overlay is not None:
            # Render through an offscreen widget so source pixels and OCR
            # coordinates match even when Windows display scaling is enabled.
            render = Overlay()
            render.setWindowFlags(Qt.Widget)
            render.resize(self.image.size())
            for key in ('mode','font_size','font_family','text_color','outline','compact',
                        'background_mode','custom_color','opacity','blocks'):
                setattr(render,key,getattr(overlay,key))
            render.font_size = max(6,round(overlay.font_size*overlay.scale))
            render.scale = 1
            painter = QPainter(self.image)
            painter.drawPixmap(0,0,render.grab())
            painter.end()
            render.deleteLater()
        self.update()

    def clear(self):
        self.image = QImage()
        self.update()

    def paintEvent(self,event):
        painter = QPainter(self)
        painter.fillRect(self.rect(),QColor('black'))
        if self.image.isNull():
            painter.setPen(QColor('white'))
            painter.drawText(self.rect(),Qt.AlignCenter,'MoonTranslate OBS • รอภาพแปล')
            return
        size = self.image.size().scaled(self.size(),Qt.KeepAspectRatio)
        target = QRect(QPoint(0,0),size)
        target.moveCenter(self.rect().center())
        painter.setRenderHint(QPainter.SmoothPixmapTransform)
        painter.drawImage(target,self.image)


class MainWindow(QWidget):
    models_loaded = Signal(object,str)
    def __init__(self):
        super().__init__()
        self.setWindowTitle('MoonTranslate • แปลเกมและมังงะ')
        self.resize(550,850)
        self.overlay = Overlay()
        self.capture = None
        self.obs_output = OBSOutput()
        self.obs_frame = None
        self.session = 0
        self.token = None
        self.region = None
        self.engine_ready = False
        self.backend_ready = False
        self.prepared_key = None
        self.resume_after_prepare = False
        self.selector = None
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0,0,0,0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        scroll.setWidget(content)
        outer.addWidget(scroll)
        root = QVBoxLayout(content)
        root.setContentsMargins(24,22,24,22)
        root.setSpacing(13)
        title = QLabel('MoonTranslate')
        title.setObjectName('title')
        root.addWidget(title)
        sub = QLabel('อังกฤษ / ญี่ปุ่น → ไทย\nแปลทับหน้าจออัตโนมัติ • ประมวลผลในเครื่อง')
        sub.setObjectName('subtitle')
        root.addWidget(sub)
        self.preview = Overlay()
        self.preview.setWindowFlags(Qt.Widget)
        self.preview.setFixedHeight(100)
        self.preview.blocks = [Block((0,0,470,90),'ตัวอย่าง','ซื้อของเพื่อผู้ที่อยู่ข้างฉัน')]
        root.addWidget(self.preview)
        preview_note = QLabel('ตัวอย่างฟอนต์ สี และพื้นหลัง • เปลี่ยนแล้วเห็นผลทันที')
        preview_note.setObjectName('subtitle')
        root.addWidget(preview_note)
        group = QGroupBox('ตั้งค่าการอ่าน')
        form = QFormLayout(group)
        self.mode = QComboBox()
        self.mode.addItem('เกม Type-Moon / Visual Novel','game')
        self.mode.addItem('มังงะ / MangaDex','manga')
        self.mode.setCurrentIndex(1)
        self.language = QComboBox()
        self.language.addItem('อัตโนมัติ (อังกฤษ / ญี่ปุ่น)','Japanese or English')
        self.language.addItem('ญี่ปุ่น','Japanese')
        self.language.addItem('อังกฤษ','English')
        self.language.setToolTip('เลือกอังกฤษสำหรับมังงะอังกฤษ เพื่อใช้ OCR อังกฤษเฉพาะทาง; อัตโนมัติและญี่ปุ่นใช้ OCR หลายภาษา')
        self.font_size = QSpinBox()
        self.font_size.setRange(12,48)
        self.font_size.setValue(22)
        self.settle = QSpinBox()
        self.settle.setRange(200,2000)
        self.settle.setSingleStep(100)
        self.settle.setValue(500)
        self.settle.setSuffix(' ms')
        form.addRow('โหมด',self.mode)
        form.addRow('ภาษาต้นฉบับ',self.language)
        self.model_picker = QComboBox()
        self.model_picker.addItem(MODEL)
        self.model_picker.setToolTip('โมเดล Ollama ของ MoonTranslate และ Gemma GGUF ใน C:/local-game-subs เปลี่ยนแล้วอาจต้องรอโหลดโมเดล')
        self.model_picker.currentTextChanged.connect(self.translation_options_changed)
        form.addRow('โมเดลแปลภาษา',self.model_picker)
        self.refresh_models_button=QPushButton('รีเฟรชรายชื่อโมเดลในเครื่อง')
        self.refresh_models_button.clicked.connect(self.refresh_models)
        form.addRow('',self.refresh_models_button)
        self.fragment_picker=QComboBox()
        self.fragment_picker.addItem('แปลคำขาดเท่าที่เข้าใจ และคง …','infer')
        self.fragment_picker.addItem('ข้ามข้อความที่ขาด / ลงท้าย …','skip')
        self.fragment_picker.setToolTip('โหมดข้ามจะข้ามทั้งกลุ่มที่ลงท้าย … หรือขีด ไม่เดาว่าส่วนที่หายไปคืออะไร')
        self.fragment_picker.currentIndexChanged.connect(self.translation_options_changed)
        form.addRow('คำไม่ครบ เช่น Big si…',self.fragment_picker)
        form.addRow('ขนาดตัวอักษรไทย',self.font_size)
        self.font_picker = QComboBox()
        families = QFontDatabase.families(QFontDatabase.WritingSystem.Thai)
        self.font_picker.addItems(sorted(set(families+['Leelawadee UI'])))
        self.font_picker.setCurrentText('Leelawadee UI')
        self.font_picker.currentTextChanged.connect(self.change_font_family)
        form.addRow('ฟอนต์ภาษาไทย',self.font_picker)
        self.text_mode = QComboBox()
        for label,value in [('อัตโนมัติ',None),('ขาว','#ffffff'),('ดำ','#111111'),('เหลือง','#ffe38a'),('เขียว','#a6ffb1')]:
            self.text_mode.addItem(label,value)
        self.text_mode.currentIndexChanged.connect(self.change_text_color)
        form.addRow('สีตัวอักษร',self.text_mode)
        self.text_color_button=QPushButton('เลือกสีตัวอักษรเอง…')
        self.text_color_button.clicked.connect(self.pick_text_color)
        form.addRow('',self.text_color_button)
        self.outline_size=QSpinBox();self.outline_size.setRange(0,3);self.outline_size.setValue(1)
        self.outline_size.valueChanged.connect(self.change_outline)
        form.addRow('เส้นขอบตัวอักษร',self.outline_size)
        self.compact_box=QCheckBox('เกม: กล่องซับพอดีกับข้อความ');self.compact_box.setChecked(True)
        self.compact_box.toggled.connect(self.change_compact)
        form.addRow(self.compact_box)
        form.addRow('รอข้อความนิ่งก่อนแปล',self.settle)
        self.bubble_only = QCheckBox('กรองพื้นหลัง (อาจทำให้บางคำพูดหาย)')
        self.bubble_only.setChecked(False)
        self.bubble_only.setToolTip('ปกติให้ปิด เพื่อแปลข้อความที่อ่านได้ทุกกลุ่ม เปิดเฉพาะเมื่อต้องการลดข้อความในฉากหลัง')
        self.bubble_only.toggled.connect(lambda *_: self.refresh())
        # Retained internally for old job tuples; manga no longer rejects
        # dialogue merely because its background contains artwork.
        self.bubble_only.hide()
        self.background_mode = QComboBox()
        for label,value in [('สีพื้นเดิมอัตโนมัติ','auto'),('ขาว','white'),('เข้ม','dark'),('เลือกสีเอง','custom')]:
            self.background_mode.addItem(label,value)
        self.background_mode.currentIndexChanged.connect(self.change_background)
        form.addRow('พื้นหลังคำแปล',self.background_mode)
        self.background_opacity = QSpinBox()
        self.background_opacity.setRange(0,100)
        self.background_opacity.setValue(100)
        self.background_opacity.setSuffix(' %')
        self.background_opacity.valueChanged.connect(self.change_opacity)
        form.addRow('ความทึบ (0 = ใส)',self.background_opacity)
        self.color_button = QPushButton('เลือกสีพื้นหลัง…')
        self.color_button.clicked.connect(self.pick_background)
        form.addRow('',self.color_button)
        root.addWidget(group)
        self.select_button = QPushButton('1. เลือกพื้นที่บนจอหลัก  ·  Ctrl+Alt+R')
        self.select_button.clicked.connect(self.select_region)
        root.addWidget(self.select_button)
        self.region_label = QLabel('เกม: ครอบเฉพาะกล่องบทสนทนา\nมังงะ: ครอบกรอบคำพูดหรือบริเวณที่กำลังอ่าน')
        self.region_label.setWordWrap(True)
        root.addWidget(self.region_label)
        self.start_button = QPushButton('2. เริ่มแปลอัตโนมัติ  ·  Ctrl+Alt+T')
        self.start_button.setObjectName('primary')
        self.start_button.setEnabled(False)
        self.start_button.clicked.connect(self.toggle)
        root.addWidget(self.start_button)
        row = QHBoxLayout()
        self.refresh_button = QPushButton('แปลใหม่  ·  Ctrl+Alt+F')
        self.refresh_button.clicked.connect(self.refresh)
        row.addWidget(self.refresh_button)
        self.show_overlay = QCheckBox('แสดงคำแปลทับ')
        self.show_overlay.setChecked(True)
        self.show_overlay.toggled.connect(self.overlay_visibility)
        row.addWidget(self.show_overlay)
        root.addLayout(row)
        self.obs_button = QPushButton('เปิดหน้าต่างภาพแปลสำหรับ OBS')
        self.obs_button.clicked.connect(self.open_obs_output)
        root.addWidget(self.obs_button)
        self.status = QLabel('กำลังเตรียม OCR และโมเดลในเครื่อง…')
        self.status.setWordWrap(True)
        self.status.setObjectName('status')
        root.addWidget(self.status)
        root.addWidget(QLabel('ข้อความล่าสุดและคำแปล'))
        self.history = QPlainTextEdit()
        self.history.setReadOnly(True)
        self.history.setPlaceholderText('คำแปลจะปรากฏที่นี่และทับพื้นที่ที่เลือก')
        self.history.setMaximumBlockCount(300)
        root.addWidget(self.history,1)
        self.glossary_button = QPushButton('เปิดคำศัพท์เฉพาะ / ชื่อตัวละคร')
        self.glossary_button.clicked.connect(self.edit_glossary)
        root.addWidget(self.glossary_button)
        note = QLabel('ใช้เกมแบบหน้าต่างหรือไร้ขอบ • ย้ายหน้าต่างแล้วเลือกพื้นที่ใหม่\nภาพเคลื่อนไหวหรือข้อความแนวตั้งซับซ้อนอาจอ่านคลาดเคลื่อน')
        note.setObjectName('subtitle')
        note.setWordWrap(True)
        root.addWidget(note)
        # Main actions remain reachable while appearance settings scroll.
        for control in (self.select_button,self.start_button,self.status):
            root.removeWidget(control)
            outer.addWidget(control)
        self.font_size.valueChanged.connect(self.change_font)
        self.mode.currentIndexChanged.connect(self.settings_changed)
        self.language.currentIndexChanged.connect(self.settings_changed)
        self.settle.valueChanged.connect(self.settings_changed)
        self.worker = Worker()
        self.worker.ready.connect(self.on_ready)
        self.worker.prepared.connect(self.on_prepared)
        self.worker.result.connect(self.on_result)
        self.worker.stage.connect(self.on_stage)
        self.worker.failed.connect(self.on_error)
        self.hotkeys = Hotkeys(self)
        QApplication.instance().installNativeEventFilter(self.hotkeys)
        user32 = ctypes.windll.user32
        user32.RegisterHotKey.argtypes = [wintypes.HWND,ctypes.c_int,wintypes.UINT,wintypes.UINT]
        user32.UnregisterHotKey.argtypes = [wintypes.HWND,ctypes.c_int]
        self.registered = []
        for ident,key in [(1,ord('T')),(2,ord('R')),(3,ord('F'))]:
            if user32.RegisterHotKey(int(self.winId()),ident,0x4003,key):
                self.registered.append(ident)
        if len(self.registered) < 3:
            self.history.appendPlainText('ปุ่มลัดบางปุ่มถูกโปรแกรมอื่นใช้: ใช้ปุ่มในหน้าต่างนี้แทนได้')
        self.load_preferences()
        self.models_loaded.connect(self.on_models_loaded)
        self.sync_preview()

    def on_ready(self, message):
        self.backend_ready = not message.startswith('ERROR:')
        self.status.setText(message)
        if self.backend_ready:
            self.refresh_models()
            self.prepare_selected_model()

    def prepare_selected_model(self):
        if not self.backend_ready:return
        key=(self.model_picker.currentText(),self.language.currentData())
        if self.engine_ready and key==self.prepared_key:return
        if self.capture is not None:
            self.stop()
            self.resume_after_prepare=True
        self.engine_ready=False
        self.start_button.setEnabled(False)
        self.status.setText('กำลังโหลดโมเดลล่วงหน้า… รอพร้อมก่อนเริ่มแปล')
        self.worker.submit(('prepare',*key))

    def on_prepared(self,key,error):
        if key!=(self.model_picker.currentText(),self.language.currentData()):return
        self.engine_ready=not error
        if error:
            self.resume_after_prepare=False
            self.status.setText('เตรียมโมเดลไม่สำเร็จ: '+error)
            return
        self.prepared_key=key
        self.start_button.setEnabled(self.region is not None)
        self.status.setText('โมเดลพร้อมแล้ว • เลือกพื้นที่แล้วเริ่มแปลได้')
        if self.resume_after_prepare:
            self.resume_after_prepare=False
            self.toggle()

    def refresh_models(self):
        self.refresh_models_button.setEnabled(False)
        def run():
            try:
                with request('/api/tags',timeout=5) as response:
                    names=sorted(m['name'] for m in json.load(response)['models']
                                 if 'capabilities' not in m or 'completion' in m['capabilities'])
                self.models_loaded.emit(names+local_models(),'')
            except Exception as exc:
                names = local_models()
                self.models_loaded.emit(names,'' if names else str(exc))
        threading.Thread(target=run,daemon=True).start()

    def on_models_loaded(self,names,error):
        self.refresh_models_button.setEnabled(True)
        if error:
            self.status.setText('อ่านรายชื่อโมเดลไม่ได้: '+error);return
        previous=self.model_picker.currentText()
        self.model_picker.blockSignals(True)
        self.model_picker.clear();self.model_picker.addItems(names)
        if previous in names:self.model_picker.setCurrentText(previous)
        elif MODEL in names:self.model_picker.setCurrentText(MODEL)
        self.model_picker.blockSignals(False)
        if previous != self.model_picker.currentText():self.translation_options_changed()

    def translation_options_changed(self,*_):
        if hasattr(self,'worker'):
            self.token=None
            if self.engine_ready and self.prepared_key==(self.model_picker.currentText(),self.language.currentData()):
                self.refresh()
            else:self.prepare_selected_model()

    def select_region(self):
        self.stop()
        self.hide()
        QTimer.singleShot(200,self.open_selector)

    def open_selector(self):
        self.selector = Selector(QApplication.primaryScreen(),self.region)
        self.selector.selected.connect(self.set_region)
        self.selector.cancelled.connect(self.show)
        self.selector.show()
        self.selector.activateWindow()

    def set_region(self, rect):
        self.region = rect
        self.region_label.setText(f'เลือกแล้ว: {rect.width()} × {rect.height()} • บนจอหลัก • Enter ยืนยันกรอบ')
        self.start_button.setEnabled(self.engine_ready)
        self.show()
        self.status.setText('เลือกพื้นที่แล้ว • กดเริ่มแปล แล้วกลับไปที่เกมหรือมังงะ')

    def toggle(self):
        if self.capture is not None:
            self.stop()
            self.status.setText('พักการแปลแล้ว')
            return
        if not self.engine_ready or self.region is None:
            self.status.setText('รอระบบพร้อม แล้วเลือกพื้นที่ก่อนเริ่มแปล')
            return
        if not self.model_picker.currentText():
            self.status.setText('ยังไม่มีโมเดลให้เลือก • กดรีเฟรชรายชื่อโมเดล');return
        screen = QApplication.primaryScreen()
        with mss.mss() as sct:
            monitor = sct.monitors[1]
        sx = monitor['width']/screen.geometry().width()
        sy = monitor['height']/screen.geometry().height()
        self.scale_x = sx
        self.scale_y = sy
        r = self.region
        physical = {'left':monitor['left']+round(r.x()*sx), 'top':monitor['top']+round(r.y()*sy),
                    'width':round(r.width()*sx), 'height':round(r.height()*sy)}
        self.overlay.setGeometry(QRect(screen.geometry().topLeft()+r.topLeft(), r.size()))
        self.overlay.scale = sx
        self.overlay.mode = self.mode.currentData()
        self.overlay.font_size = self.font_size.value()
        self.overlay.show()
        if not self.overlay.exclude_from_capture():
            self.overlay.hide()
            self.status.setText('Windows ไม่รองรับการแยกภาพ overlay ออกจากภาพจับหน้าจอ จึงยังไม่เริ่มแปล')
            return
        self.session += 1
        self.capture = Capture(physical,self.session,self.settle.value())
        self.capture.invalidated.connect(self.on_invalidated)
        self.capture.stable.connect(self.on_stable)
        self.capture.failed.connect(self.capture_error)
        self.capture.start()
        self.start_button.setText('พักการแปล  ·  Ctrl+Alt+T')
        self.status.setText('กำลังตรวจข้อความ • กลับไปที่เกมหรือหน้า MangaDex ได้เลย')
        self.mode.setEnabled(False)
        self.language.setEnabled(False)
        self.settle.setEnabled(False)

    def stop(self):
        self.session += 1
        self.token = None
        self.obs_frame = None
        self.obs_output.clear()
        if self.capture is not None:
            self.capture.stopping.set()
            self.capture.wait(3000)
            self.capture = None
        if self.engine_ready:self.worker.cancel()
        self.overlay.clear()
        self.overlay.hide()
        self.start_button.setText('2. เริ่มแปลอัตโนมัติ  ·  Ctrl+Alt+T')
        self.mode.setEnabled(True)
        self.language.setEnabled(True)
        self.settle.setEnabled(True)

    def capture_error(self, message):
        self.stop()
        self.status.setText('จับภาพไม่ได้: '+message)

    def settings_changed(self, *_):
        if self.capture is not None:
            self.stop()
        if hasattr(self,'worker'):self.prepare_selected_model()

    def refresh(self):
        if self.capture is not None:
            self.worker.cancel()
            self.overlay.clear()
            self.capture.force.set()

    def overlay_visibility(self, visible):
        self.overlay.enabled = visible
        self.overlay.update()

    def change_font(self, value):
        self.overlay.font_size = value
        self.overlay.update()
        self.sync_preview()

    def sync_preview(self):
        for key in ('font_size','font_family','text_color','outline','compact','background_mode','custom_color','opacity'):
            setattr(self.preview,key,getattr(self.overlay,key))
        self.preview.update()

    def change_font_family(self,value):
        self.overlay.font_family=value;self.overlay.update();self.sync_preview()

    def change_text_color(self,*_):
        self.overlay.text_color=self.text_mode.currentData();self.overlay.update();self.sync_preview()

    def pick_text_color(self):
        color=QColorDialog.getColor(QColor(self.overlay.text_color or '#ffffff'),self,'เลือกสีตัวอักษร')
        if color.isValid():
            self.text_mode.addItem(color.name(),color.name());self.text_mode.setCurrentIndex(self.text_mode.count()-1)

    def change_outline(self,value):
        self.overlay.outline=value;self.overlay.update();self.sync_preview()

    def change_compact(self,value):
        self.overlay.compact=value;self.overlay.update();self.sync_preview()

    def change_background(self, *_):
        self.overlay.background_mode = self.background_mode.currentData()
        self.overlay.update()
        self.sync_preview()

    def change_opacity(self, value):
        self.overlay.opacity = value
        self.overlay.update()
        self.sync_preview()

    def pick_background(self):
        color = QColorDialog.getColor(QColor(self.overlay.custom_color),self,'เลือกสีพื้นหลังคำแปล')
        if color.isValid():
            self.overlay.custom_color = color.name()
            self.background_mode.setCurrentIndex(self.background_mode.findData('custom'))
            self.overlay.update()
            self.sync_preview()

    def load_preferences(self):
        try:
            data=json.loads((ROOT/'appearance.json').read_text(encoding='utf-8'))
            self.font_size.setValue(int(data.get('font_size',22)))
            saved_model=data.get('model',MODEL)
            if saved_model != MODEL:self.model_picker.addItem(saved_model)
            self.model_picker.setCurrentText(saved_model)
            index=self.language.findData(data.get('source_language','Japanese or English'))
            if index>=0:self.language.setCurrentIndex(index)
            index=self.fragment_picker.findData(data.get('fragment_policy','infer'))
            if index>=0:self.fragment_picker.setCurrentIndex(index)
            self.font_picker.setCurrentText(data.get('font_family','Leelawadee UI'))
            self.outline_size.setValue(int(data.get('outline',1)))
            self.compact_box.setChecked(bool(data.get('compact',True)))
            self.overlay.custom_color=data.get('custom_color','#fffdf6')
            index=self.background_mode.findData(data.get('background_mode','auto'))
            if index>=0:self.background_mode.setCurrentIndex(index)
            self.background_opacity.setValue(int(data.get('opacity',100)))
            color=data.get('text_color')
            index=self.text_mode.findData(color)
            if index<0 and QColor(color).isValid():
                self.text_mode.addItem(color,color);index=self.text_mode.count()-1
            if index>=0:self.text_mode.setCurrentIndex(index)
        except (OSError,ValueError,TypeError):
            pass

    def save_preferences(self):
        data={k:getattr(self.overlay,k) for k in ('font_size','font_family','outline','compact','custom_color','background_mode','opacity','text_color')}
        data.update(model=self.model_picker.currentText(),fragment_policy=self.fragment_picker.currentData(),
                    source_language=self.language.currentData())
        try:
            temporary=ROOT/'appearance.json.tmp'
            temporary.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
            temporary.replace(ROOT/'appearance.json')
        except OSError:
            pass

    def on_invalidated(self, token):
        if self.capture is None or token[0] != self.session:
            return
        self.token = token
        self.worker.cancel()
        self.overlay.clear()
        self.obs_frame = None
        self.obs_output.clear()
        self.status.setText('ข้อความเปลี่ยน • รอภาพนิ่ง…')

    def glossary(self):
        path = ROOT/'glossary.json'
        if not path.exists():
            path.write_text('{}\n',encoding='utf-8')
        value = json.loads(path.read_text(encoding='utf-8-sig'))
        if not isinstance(value,dict) or not all(isinstance(k,str) and isinstance(v,str) for k,v in value.items()):
            raise ValueError('glossary.json ต้องเป็นคู่ข้อความต้นฉบับกับคำแปล')
        return value

    def edit_glossary(self):
        try:
            self.glossary()
            import subprocess
            subprocess.Popen(['notepad.exe',str(ROOT/'glossary.json')])
        except Exception as exc:
            QMessageBox.warning(self,'คำศัพท์เฉพาะ',str(exc))

    def on_stable(self, token, frame):
        if self.capture is None or token[0] != self.session or token != self.token:
            return
        try:
            self.obs_frame = frame.copy()
            self.obs_output.set_frame(frame)
            self.worker.submit((token,frame,self.mode.currentData(),self.language.currentData(),self.glossary(),self.bubble_only.isChecked(),self.model_picker.currentText(),self.fragment_picker.currentData()))
        except Exception as exc:
            self.on_error(token,str(exc))

    def on_stage(self, token, message):
        if token == self.token:
            self.status.setText(message)

    def on_result(self, token, blocks, elapsed):
        if token != self.token or self.capture is None:
            return
        self.overlay.blocks = blocks
        self.overlay.update()
        if self.obs_frame is not None:
            self.obs_output.set_frame(self.obs_frame,self.overlay)
        if not blocks:
            self.status.setText('ไม่มีคำแปลรอบนี้ • อาจไม่พบข้อความ หรือข้ามคำที่อ่านไม่ครบ')
            return
        self.status.setText(f'แปลแล้ว {len(blocks)} กลุ่ม • {elapsed:.1f} วินาที • รอข้อความถัดไป')
        for block in blocks:
            self.history.appendPlainText(block.text+'\n→ '+block.translated+'\n')

    def on_error(self, token, message):
        if token == self.token:
            self.overlay.clear()
            if self.obs_frame is not None:
                self.obs_output.set_frame(self.obs_frame)
            self.status.setText('แปลไม่สำเร็จ: '+message+' • กดแปลใหม่เพื่อลองอีกครั้ง')

    def closeEvent(self, event):
        self.save_preferences()
        self.stop()
        self.worker.closed = True
        self.worker.cancel()
        for ident in self.registered:
            ctypes.windll.user32.UnregisterHotKey(int(self.winId()),ident)
        QApplication.instance().removeNativeEventFilter(self.hotkeys)
        self.overlay.close()
        self.obs_output.close()
        event.accept()

    def open_obs_output(self):
        self.obs_output.show()
        self.obs_output.raise_()
        self.status.setText('OBS: เพิ่ม Window Capture → MoonTranslate OBS • วางหน้าต่างนี้นอกพื้นที่ OCR และอย่าย่อหน้าต่าง')


STYLE = '''
QWidget { background:#101827; color:#e7edf8; font-family:'Leelawadee UI'; font-size:14px; }
QLabel#title { font-size:30px; font-weight:700; color:#c4b5fd; }
QLabel#subtitle { color:#a8b6cc; font-size:12px; }
QGroupBox { border:1px solid #334155; border-radius:10px; margin-top:12px; padding:16px 10px 8px; }
QGroupBox::title { subcontrol-origin:margin; left:12px; color:#a8b6cc; }
QPushButton { background:#22314a; border:1px solid #3b4e6d; border-radius:8px; padding:10px; }
QPushButton:hover { background:#304768; }
QPushButton:disabled { color:#73819a; background:#1a2436; }
QPushButton#primary { background:#7860d7; color:white; font-weight:600; border:0; padding:14px; }
QPushButton#primary:hover { background:#8b74e5; }
QPushButton#primary:disabled { background:#343249; color:#8e89a5; }
QComboBox,QSpinBox,QPlainTextEdit { background:#182438; border:1px solid #3b4e6d; border-radius:5px; padding:5px; }
QLabel#status { color:#7dd3fc; padding:8px; background:#16243b; border-radius:6px; }
'''


def load_fonts():
    fonts = Path(os.environ.get('WINDIR','C:/Windows'))/'Fonts'
    for name in ('LeelawUI.ttf','LeelaUIb.ttf'):
        QFontDatabase.addApplicationFont(str(fonts/name))


def main():
    ROOT.mkdir(parents=True,exist_ok=True)
    app = QApplication(sys.argv)
    load_fonts()
    app.setStyle('Fusion')
    app.setStyleSheet(STYLE)
    window = MainWindow()
    window.show()
    print('Main window shown',flush=True)
    return app.exec()


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception:
        ROOT.mkdir(parents=True,exist_ok=True)
        (ROOT/'crash.log').write_text(traceback.format_exc(),encoding='utf-8')
        raise
