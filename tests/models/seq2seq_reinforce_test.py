import json

from tests import ModelTestCase

class Seq2SeqREINFORCETest(ModelTestCase):
    def setup_method(self):
        super().setup_method()
        self.set_up_model(
            ModelTestCase.FIXTURES_ROOT / "natural_lang" / "seq2seq_reinforce.jsonnet",
            ModelTestCase.FIXTURES_ROOT / "natural_lang" / "seq2seq_copy.tsv",
        )

    def test_model_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(self.param_file, tolerance=1e-2)
