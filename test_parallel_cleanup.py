import unittest
import numpy as np
from core import Block,Reader,merge_cleaned
from unittest.mock import patch

class ParallelCleanupTests(unittest.TestCase):
    def test_skipped_translation_stays_with_correct_cleaned_box(self):
        original=[Block((0,0,10,10),'skip'),Block((20,0,30,10),'keep')]
        cleaned=[Block((0,0,12,12),'skip'),Block((18,0,32,12),'keep')]
        translated=[Block((20,0,30,10),'keep','คำแปล')]
        result=merge_cleaned(original,cleaned,translated)
        self.assertEqual(result[0].rect,(18,0,32,12))
        self.assertEqual(result[0].translated,'คำแปล')
        self.assertEqual(cleaned[1].translated,'')

    def test_cleaning_copies_input_and_cache(self):
        reader=Reader.__new__(Reader)
        image=np.full((50,50,3),255,np.uint8)
        original=[Block((10,10,20,20),'hello')]
        patch_result=((9,9,21,21),(10,10,20,20),np.full((12,12,3),255,np.uint8),(255,255,255))
        with patch('core.prepare_patch',return_value=patch_result) as prepare:
            first=reader.clean(image,original)
            first[0].background_image[:]=0
            second=reader.clean(image,original)
            self.assertEqual(prepare.call_count,1)
            self.assertTrue(np.all(second[0].background_image==255))
        self.assertEqual(original[0].rect,(10,10,20,20))
        self.assertIsNone(original[0].background_image)

    def test_cancelled_cleanup_returns_no_partial_page(self):
        reader=Reader.__new__(Reader)
        result=reader.clean(np.zeros((20,20,3),np.uint8),[Block((1,1,9,9),'text')],lambda:True)
        self.assertEqual(result,[])

if __name__=='__main__':unittest.main()
