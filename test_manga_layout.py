import unittest
import cv2
import numpy as np
from manga_layout import prepare_patch


class MangaLayoutTests(unittest.TestCase):
    def test_patch_can_be_converted_for_overlay(self):
        from app import numpy_to_qimage
        pixels = np.zeros((16,24,3),np.uint8)
        pixels[:,:,2] = 255
        rendered = numpy_to_qimage(pixels)
        self.assertEqual(rendered.pixelColor(0,0).red(),255)
        self.assertEqual(rendered.width(),24)

    def fixture(self):
        image = np.full((200,240,3),150,np.uint8)
        cv2.ellipse(image,(120,100),(85,80),0,0,360,(255,255,255),-1)
        cv2.ellipse(image,(120,100),(85,80),0,0,360,(0,0,0),3)
        cv2.putText(image,'HELLO',(80,100),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,0,0),1)
        rect = (78,87,133,103)
        polygon = np.array([[78,87],[133,87],[133,103],[78,103]])
        return image,rect,(polygon,)

    def test_closed_balloon_grows_and_preserves_border(self):
        image,rect,polygons = self.fixture()
        paint,layout,patch,_ = prepare_patch(image,rect,polygons,[])
        self.assertGreater(layout[2]-layout[0],rect[2]-rect[0])
        result = image.copy()
        a,b,c,d = paint; result[b:d,a:c] = patch
        outside = np.ones(image.shape[:2],bool);outside[85:106,76:136]=False
        np.testing.assert_array_equal(image[outside],result[outside])
        self.assertGreater(result[90:99,80:130].mean(),245)

    def test_open_background_does_not_expand(self):
        image,rect,polygons = self.fixture()
        image[:] = 255
        _,layout,_,_ = prepare_patch(image,rect,polygons,[])
        self.assertEqual(layout,rect)

    def test_closed_rectangle_recentres_off_centre_source(self):
        image=np.full((220,440,3),150,np.uint8)
        cv2.rectangle(image,(20,20),(420,200),(255,255,255),-1)
        cv2.rectangle(image,(20,20),(420,200),(0,0,0),3)
        cv2.putText(image,'Hi',(55,105),cv2.FONT_HERSHEY_SIMPLEX,.7,(0,0,0),1)
        rect=(52,85,83,110);polygon=np.array([[52,85],[83,85],[83,110],[52,110]])
        _,layout,_,_=prepare_patch(image,rect,(polygon,),[])
        self.assertAlmostEqual((layout[0]+layout[2])/2,220,delta=3)
        self.assertAlmostEqual((layout[1]+layout[3])/2,110,delta=3)
        self.assertGreaterEqual(layout[0],20)
        self.assertLessEqual(layout[2],420)
        self.assertGreaterEqual(layout[1],20)
        self.assertLessEqual(layout[3],200)

    def test_no_expansion_into_other_text(self):
        image,rect,polygons = self.fixture()
        other = (140,85,155,110)
        _,layout,_,_ = prepare_patch(image,rect,polygons,[other])
        self.assertLessEqual(layout[2],other[0])


if __name__ == '__main__': unittest.main()
