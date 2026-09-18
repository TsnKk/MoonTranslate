"""Conservative balloon fitting and OCR-local erasure, entirely on device."""
import cv2
import numpy as np
from manga_inpaint import inpaint


def prepare_patch(image, rect, polygons, other_rects):
    height, width = image.shape[:2]
    x1,y1,x2,y2 = rect
    # Balloon boundaries may be far from short, off-centre dialogue. Layout
    # needs wider context; LaMa separately crops tightly around its own mask.
    pad = max(512, min(768, max(x2-x1,y2-y1)*3))
    left,top,right,bottom = max(0,x1-pad),max(0,y1-pad),min(width,x2+pad),min(height,y2+pad)
    crop = image[top:bottom,left:right].copy()
    support = np.zeros(crop.shape[:2], np.uint8)
    for polygon in polygons:
        cv2.fillPoly(support, [np.round(polygon-[left,top]).astype(np.int32)], 255)
    # Outline fonts have a white halo *outside* the OCR polygon. Sampling that
    # halo as background produced the white patches seen on shaded narration.
    heights = [min(np.ptp(p[:,0]),np.ptp(p[:,1])) for p in polygons]
    halo = max(2,min(8,round(float(np.median(heights))*.18))) if heights else 2
    expanded = cv2.dilate(support,np.ones((halo*2+1,halo*2+1),np.uint8))
    ring = cv2.dilate(expanded,np.ones((13,13),np.uint8)) > 0
    ring &= expanded == 0
    pixels = crop[ring]
    if not len(pixels): pixels = crop.reshape(-1,3)
    bins = pixels.astype(np.int32)//16
    ids = bins[:,0]*256+bins[:,1]*16+bins[:,2]
    common = np.bincount(ids,minlength=4096).argmax()
    background = np.median(pixels[ids==common],axis=0).astype(np.uint8)
    diff = np.max(np.abs(crop.astype(np.int16)-background.astype(np.int16)),axis=2)
    textured = np.mean(np.max(np.abs(pixels.astype(np.int16)-background.astype(np.int16)),axis=1)>28) > .15
    if textured:
        # Remove both ink and its outline. The previous glyph-only mask left
        # white islands which then contaminated the inpaint boundary.
        ink = expanded.copy()
    else:
        ink = ((diff > 32) & (expanded > 0)).astype(np.uint8)*255
        ink = cv2.dilate(ink,np.ones((3,3),np.uint8))
        ink[expanded == 0] = 0
    # A neighbouring OCR group is never part of this erasure.
    for a,b,c,d in other_rects:
        aa,bb,cc,dd=max(0,a-left-1),max(0,b-top-1),min(crop.shape[1],c-left+1),min(crop.shape[0],d-top+1)
        if cc>aa and dd>bb: ink[bb:dd,aa:cc]=0
    cleaned = (inpaint(crop,ink) if textured else cv2.inpaint(crop,ink,3,cv2.INPAINT_TELEA)) if ink.any() else crop
    # A closed, mostly uniform region is required before enlarging text layout.
    uniform = (diff < 28).astype(np.uint8)
    uniform[support > 0] = 1
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(uniform,8)
    source = labels[y1-top:y2-top,x1-left:x2-left]
    candidates = np.bincount(source.ravel(),minlength=count)
    candidates[0] = 0
    label = int(candidates.argmax())
    safe = labels == label
    region = None
    if label and candidates[label] >= source.size*.85:
        sx,sy,sw,sh,area = stats[label]
        enclosed = sx>0 and sy>0 and sx+sw<crop.shape[1] and sy+sh<crop.shape[0]
        if enclosed and area > (x2-x1)*(y2-y1)*1.25:
            safe = cv2.erode(safe.astype(np.uint8),np.ones((9,9),np.uint8)) > 0
            invalid = cv2.integral((~safe).astype(np.uint8))
            def fits(a,b,c,d):
                return invalid[d,c]-invalid[b,c]-invalid[d,a]+invalid[b,a] == 0
            # Expand only rectangular strips wholly inside the balloon.
            a,b,c,d = x1-left,y1-top,x2-left,y2-top
            if fits(a,b,c,d):
                for _ in range(pad):
                    changed = False
                    for side in range(4):
                        aa,bb,cc,dd = a-(side==0),b-(side==1),c+(side==2),d+(side==3)
                        global_rect = (aa+left,bb+top,cc+left,dd+top)
                        collision = any(min(global_rect[2],q[2])>max(global_rect[0],q[0]) and
                                        min(global_rect[3],q[3])>max(global_rect[1],q[1]) for q in other_rects)
                        if aa>=0 and bb>=0 and cc<=safe.shape[1] and dd<=safe.shape[0] and not collision and fits(aa,bb,cc,dd):
                            a,b,c,d = aa,bb,cc,dd
                            changed = True
                    if not changed: break
                region = (a+left,b+top,c+left,d+top)
            # A large closed balloon can have off-centre source lettering.
            # Prefer a safe centred rectangle rather than retaining that drift.
            cx,cy=centroids[label]
            best_area=(region[2]-region[0])*(region[3]-region[1]) if region else 0
            for fx in (.95,.9,.85,.8,.75,.7,.65,.6,.55,.5):
                for fy in (.95,.9,.85,.8,.75,.7,.65,.6,.55,.5):
                    rw,rh=max(1,round(sw*fx)),max(1,round(sh*fy))
                    aa,bb=round(cx-rw/2),round(cy-rh/2);cc,dd=aa+rw,bb+rh
                    if rw*rh<best_area or aa<0 or bb<0 or cc>safe.shape[1] or dd>safe.shape[0]:continue
                    candidate=(aa+left,bb+top,cc+left,dd+top)
                    collision=any(min(candidate[2],q[2])>max(candidate[0],q[0]) and
                                  min(candidate[3],q[3])>max(candidate[1],q[1]) for q in other_rects)
                    if not collision and fits(aa,bb,cc,dd):region=candidate;best_area=rw*rh
    # Keep a small margin for anti-aliased source glyphs even without a balloon.
    paint_rect = (max(left,x1-halo),max(top,y1-halo),min(right,x2+halo),min(bottom,y2+halo))
    if region:
        paint_rect = (min(paint_rect[0],region[0]),min(paint_rect[1],region[1]),
                      max(paint_rect[2],region[2]),max(paint_rect[3],region[3]))
    a,b,c,d = paint_rect
    return paint_rect, region or rect, cleaned[b-top:d-top,a-left:c-left], tuple(int(v) for v in background[::-1])
