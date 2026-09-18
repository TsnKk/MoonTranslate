import unittest
from unittest.mock import Mock
from types import SimpleNamespace
import numpy as np
from ocr_refine import choose_reading,refine_detections,english_spacing,non_latin_consensus


class RefinementTests(unittest.TestCase):
    def test_spacing_keeps_letters_and_does_not_split_names(self):
        source='I THINK\nYOUCAN\nHELP SHIKI'
        result=english_spacing(source)
        self.assertIn('YOU CAN',result)
        self.assertEqual(''.join(result.split()),''.join(source.split()))
        self.assertEqual(english_spacing('I SAW SHIKI TODAY'),'I SAW SHIKI TODAY')

    def test_short_names_are_not_segmented_out_of_context(self):
        self.assertEqual(english_spacing('IKNOW'),'IKNOW')

    def test_foreign_noise_requires_low_confidence_and_agreement(self):
        self.assertTrue(non_latin_consensus(.5,[('セキ',.9),('セキ',.92)]))
        self.assertFalse(non_latin_consensus(.98,[('セキ',.9),('セキ',.92)]))
        self.assertFalse(non_latin_consensus(.5,[('セキ',.9),('セイ',.92)]))

    def test_agreement_with_clear_gain_repairs_a_character(self):
        self.assertEqual(choose_reading('I CAN7',.72,[("I CAN'T",.95),("I CAN'T",.94)])[0],"I CAN'T")

    def test_disagreement_does_not_overwrite_a_name(self):
        self.assertEqual(choose_reading('Shiki',.88,[('Shiro',.93),('Shige',.94)])[0],'Shiki')

    def test_unrelated_high_confidence_is_rejected(self):
        self.assertEqual(choose_reading('WAIT',.5,[('ありがとう',.99)])[0],'WAIT')

    def test_matching_reading_preserves_word_boundaries(self):
        self.assertEqual(choose_reading('CANTGO',.7,[("CAN'T GO",.96),("CAN'TGO",.98)])[0],"CAN'T GO")

    def test_high_confidence_lines_are_not_reread_and_work_is_bounded(self):
        engine=Mock(return_value=SimpleNamespace(txts=['HELLO'],scores=[.91]))
        box=np.array([[10,10],[80,10],[80,30],[10,30]])
        _,stats=refine_detections(np.full((60,100,3),255,np.uint8),[(box,'HELLO',.99)]*10,engine,'English')
        self.assertEqual(stats['checked'],0);engine.assert_not_called()
        _,stats=refine_detections(np.full((60,100,3),255,np.uint8),[(box,'HELLO',.7)]*10,engine,'English',limit=3,budget=10)
        self.assertEqual(stats['checked'],3);self.assertEqual(engine.call_count,6)


if __name__=='__main__':unittest.main()
