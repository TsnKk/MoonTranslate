import unittest
from unittest.mock import Mock, patch
from core import Block, Translator
from local_models import GEMMA, local_models
from prepare_engine import MODEL


class LocalModelsTest(unittest.TestCase):
    def test_missing_files_not_listed(self):
        with patch('local_models.MODEL_PATH') as model:
            model.is_file.return_value = False
            self.assertEqual(local_models(), [])

    def test_gemma_routing_cache_and_switch_cleanup(self):
        translator = Translator(GEMMA)
        translator.gemma = Mock()
        translator.gemma.chat.return_value = '{"translations":["สวัสดี"]}'
        translator.session = Mock()
        blocks = [Block((0, 0, 20, 20), 'Hello')]
        self.assertEqual(translator.translate(blocks, 'English', {})[0].translated, 'สวัสดี')
        translator.translate(blocks, 'English', {})
        translator.gemma.chat.assert_called_once()
        translator.session.post.assert_not_called()
        translator.model = MODEL
        translator.translate([], 'English', {})
        translator.gemma.close.assert_called_once()


if __name__ == '__main__':
    unittest.main()
