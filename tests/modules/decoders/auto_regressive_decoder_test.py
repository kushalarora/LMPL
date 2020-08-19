from tests import ModelTestCase
import json 

class TestAutoRegressiveDecoder(ModelTestCase):
    def setup_method(self):
      super().setup_method()
      self.set_up_model(
          ModelTestCase.FIXTURES_ROOT / "natural_lang" / "ptb_lm.jsonnet",
          ModelTestCase.FIXTURES_ROOT / "natural_lang" / "sentences.txt",
      )

    def test_model_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(self.param_file, tolerance=1e-2)

    def test_epoch_and_batch_number_is_updated(self):
        param_overrides = json.dumps({"trainer": {"num_epochs": 2}})
        model, loaded_model = self.ensure_model_can_train_save_and_load(
                                    self.param_file, tolerance=1e-2, overrides=param_overrides)
        assert model.epoch > 0
        assert model.batch_number > 0