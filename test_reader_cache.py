import unittest
from unittest.mock import Mock
from types import SimpleNamespace
import numpy as np
from core import Reader,Block

class ReaderCacheTests(unittest.TestCase):
    def test_cache_copies_results_and_invalidates_pixels_and_language(self):
        reader=Reader.__new__(Reader)
        reader._read_image=Mock(return_value=[Block((0,0,20,20),'THEM TO YOU')])
        image=np.zeros((25,25,3),np.uint8)
        first=reader.read(image,'manga','English')
        first[0].text='modified by caller'
        second=reader.read(image,'manga','English')
        self.assertTrue(reader.cache_hit)
        self.assertEqual(second[0].text,'THEM TO YOU')
        self.assertEqual(reader._read_image.call_count,1)
        changed=image.copy();changed[0,0]=1
        reader.read(changed,'manga','English')
        reader.read(image,'manga','Japanese')
        self.assertEqual(reader._read_image.call_count,3)

    def test_manga_does_not_rotate_upright_italic_text(self):
        reader=Reader.__new__(Reader)
        def engine(image,**kwargs):
            text='WEHL' if kwargs['use_cls'] else 'THEM'
            return SimpleNamespace(boxes=[np.array([[2,2],[18,2],[18,12],[2,12]])],txts=[text],scores=[.99])
        reader.engine=engine
        image=np.full((30,40,3),255,np.uint8)
        self.assertEqual(reader.read(image,'manga','English')[0].text,'THEM')

    def test_cache_has_bounded_entry_count(self):
        reader=Reader.__new__(Reader)
        reader._read_image=Mock(return_value=[])
        for value in range(16):reader.read(np.full((2,2,3),value,np.uint8),'manga','English')
        self.assertEqual(len(reader.read_cache),12)

if __name__=='__main__':unittest.main()
