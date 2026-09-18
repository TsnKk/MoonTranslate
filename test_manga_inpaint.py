import unittest
from unittest.mock import patch,Mock
import numpy as np
import manga_inpaint

class InpaintTests(unittest.TestCase):
    def test_restores_only_mask_pixels_after_resizing(self):
        source=np.full((600,700,3),123,np.uint8)
        mask=np.zeros((600,700),np.uint8);mask[100:140,160:190]=255
        session=Mock()
        session.run.return_value=[np.zeros((1,3,512,512),np.float32)]
        with patch.object(manga_inpaint,'available',return_value=True),patch.object(manga_inpaint,'_session',session):
            result=manga_inpaint.inpaint(source,mask)
        np.testing.assert_array_equal(result[mask==0],source[mask==0])
        self.assertTrue(np.all(result[mask>0]==0))
        self.assertTrue(np.all(source==123))
        inputs=session.run.call_args.args[1]
        self.assertEqual(inputs['image'].shape,(1,3,512,512))
        self.assertEqual(inputs['mask'].shape,(1,1,512,512))

    def test_empty_mask_never_loads_model(self):
        image=np.zeros((10,10,3),np.uint8)
        with patch.object(manga_inpaint,'available',side_effect=AssertionError):
            result=manga_inpaint.inpaint(image,np.zeros((10,10),np.uint8))
        np.testing.assert_array_equal(result,image)

    def test_distant_text_is_excluded_from_model_context(self):
        source=np.full((700,700,3),200,np.uint8)
        source[:80]=17
        mask=np.zeros((700,700),np.uint8);mask[320:380,320:380]=255
        session=Mock();session.run.return_value=[np.zeros((1,3,512,512),np.float32)]
        with patch.object(manga_inpaint,'available',return_value=True),patch.object(manga_inpaint,'_session',session):
            result=manga_inpaint.inpaint(source,mask)
        self.assertTrue(np.all(session.run.call_args.args[1]['image']>.7))
        np.testing.assert_array_equal(result[mask==0],source[mask==0])

if __name__=='__main__':unittest.main()
