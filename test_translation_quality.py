import json
import unittest
from unittest.mock import Mock
from core import Translator,Block,added_question


class TranslationQualityTests(unittest.TestCase):
    def test_duplicate_dialogue_is_translated_once_and_keeps_all_boxes(self):
        translator=Translator('test');translator.ollama_chat=Mock(return_value=json.dumps({'translations':['ขอบคุณนะ']}))
        blocks=[Block((i*20,0,i*20+10,10),'Thank you.') for i in range(3)]
        result=translator.translate(blocks,'English',{})
        self.assertEqual(len(result),3)
        self.assertEqual([b.rect for b in result],[b.rect for b in blocks])
        self.assertEqual(json.loads(translator.ollama_chat.call_args.args[0][1]['content']),['Thank you.'])

    def test_question_guard_preserves_real_questions(self):
        self.assertFalse(added_question('Do you know why?','รู้ไหมว่าทำไม'))
        self.assertFalse(added_question('Can you hear me','ได้ยินฉันใช่ไหม'))
        self.assertTrue(added_question('I warned you!','เตือนแล้วใช่ไหม!'))

    def test_only_suspicious_item_is_reviewed_and_cached(self):
        translator=Translator('test');translator.ollama_chat=Mock(side_effect=[
            json.dumps({'translations':['เตือนแล้วใช่ไหม!','ขอบคุณนะ']}),
            json.dumps({'translations':['เตือนแล้วนะ!']})])
        blocks=[Block((0,0,10,10),'I warned you!'),Block((20,0,30,10),'Thank you.')]
        result=translator.translate(blocks,'English',{})
        self.assertEqual([b.translated for b in result],['เตือนแล้วนะ!','ขอบคุณนะ'])
        messages=translator.ollama_chat.call_args.args[0]
        self.assertEqual(len(json.loads(messages[1]['content'])),1)
        translator.translate(blocks,'English',{})
        self.assertEqual(translator.ollama_chat.call_count,2)


if __name__=='__main__':unittest.main()
