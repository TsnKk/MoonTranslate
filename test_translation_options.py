import unittest
from core import Translator,Block

class TranslationOptionsTests(unittest.TestCase):
    def test_warmup_contacts_model_even_after_switching_back(self):
        translator=self.translator()
        translator.warmup('English')
        translator.warmup('English')
        self.assertEqual(len(translator.session.calls),2)

    def test_review_is_cached_separately_and_uses_revised_output(self):
        from unittest.mock import Mock
        translator=Translator()
        translator.ollama_chat=Mock(side_effect=[
            '{"translations":["คำแปลเดิม"]}',
            '{"translations":["คำแปลร่าง"]}',
            '{"translations":["คำแปลตรวจทาน"]}'])
        blocks=[Block((0,0,100,40),'A complete sentence')]
        translator.translate(blocks,'English',{})
        result=translator.translate(blocks,'English',{},review=True)
        self.assertEqual(result[0].translated,'คำแปลตรวจทาน')
        translator.translate(blocks,'English',{},review=True)
        self.assertEqual(translator.ollama_chat.call_count,3)

    def translator(self):
        translator=Translator()
        class Response:
            def raise_for_status(self):pass
            def json(self):return {'message':{'content':'{"translations":["พี่สา"]}'}}
        class Session:
            calls=[]
            def post(self,*args,**kwargs):self.calls.append(kwargs['json']);return Response()
        translator.session=Session()
        return translator

    def test_skip_fragment_sends_nothing(self):
        translator=self.translator()
        for text in ['Big si...','Big si…','Big si—']:
            self.assertEqual(translator.translate([Block((0,0,100,40),text)],'English',{},fragment_policy='skip'),[])
        self.assertEqual(translator.session.calls,[])

    def test_infer_preserves_ellipsis(self):
        translator=self.translator()
        blocks=translator.translate([Block((0,0,100,40),'Big si...')],'English',{})
        self.assertEqual(blocks[0].translated,'พี่สา…')

    def test_cache_separates_models_and_policies(self):
        translator=self.translator();blocks=[Block((0,0,100,40),'Big sister')]
        translator.translate(blocks,'English',{})
        translator.translate(blocks,'English',{})
        self.assertEqual(len(translator.session.calls),1)
        translator.model='another-installed-model'
        translator.translate(blocks,'English',{})
        translator.translate(blocks,'English',{},fragment_policy='skip')
        self.assertEqual(len(translator.session.calls),3)
        self.assertEqual(translator.session.calls[-1]['model'],'another-installed-model')

    def test_unreadable_empty_answer_is_omitted(self):
        translator=self.translator()
        class Response:
            def raise_for_status(self):pass
            def json(self):return {'message':{'content':'{"translations":[""]}'}}
        translator.session.post=lambda *a,**k:Response()
        self.assertEqual(translator.translate([Block((0,0,100,40),'zxqv...')],'English',{}),[])

if __name__=='__main__':unittest.main(verbosity=2)
