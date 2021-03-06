import json

from tests import ModelTestCase

class Seq2SeqMLETest(ModelTestCase):
    def setup_method(self):
        super().setup_method()
        self.set_up_model(
            ModelTestCase.FIXTURES_ROOT / "natural_lang" / "seq2seq_mle.jsonnet",
            ModelTestCase.FIXTURES_ROOT / "natural_lang" / "seq2seq_copy.tsv",
        )

    def test_model_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(self.param_file, tolerance=1e-2)

    def test_bidirectional_model_can_train_save_and_load(self):

        param_overrides = json.dumps(
            {
                "model": {
                    "encoder": {"bidirectional": True},
                    "decoder": {"decoder_net": {"decoding_dim": 24}},
                }
            }
        )
        self.ensure_model_can_train_save_and_load(
            self.param_file, tolerance=1e-2, overrides=param_overrides
        )

    def test_no_attention_model_can_train_save_and_load(self):
        param_overrides = json.dumps({"model": {"decoder": {"decoder_net": {"attention": None}}}})
        self.ensure_model_can_train_save_and_load(
            self.param_file, tolerance=1e-2, overrides=param_overrides
        )

    def test_decode_runs_correctly(self):
        self.model.eval()
        training_tensors = self.dataset.as_tensor_dict()
        output_dict = self.model(**training_tensors)
        decode_output_dict = self.model.make_output_human_readable(output_dict)
        # `make_output_human_readable` should have added a `predicted_tokens` field to
        # `output_dict`. Checking if it's there.
        assert "predicted_tokens" in decode_output_dict

        # The output of model.make_output_human_readable should still have 'predicted_tokens' after
        # using the beam search. To force the beam search, we just remove `target_tokens` from the
        # input tensors.
        del training_tensors["target_tokens"]
        output_dict = self.model(**training_tensors)
        decode_output_dict = self.model.make_output_human_readable(output_dict)
        assert "predicted_tokens" in decode_output_dict

    def test_with_tensor_based_metric(self):
        param_overrides = json.dumps({"model": {"decoder": {"tensor_based_metric": "bleu"}}})
        self.ensure_model_can_train_save_and_load(
            self.param_file, tolerance=1e-2, overrides=param_overrides
        )


    def test_epoch_and_batch_number_is_updated(self):
        param_overrides = json.dumps({"trainer": {"num_epochs": 2}})
        model, loaded_model = self.ensure_model_can_train_save_and_load(
            self.param_file, tolerance=1e-2, overrides=param_overrides)
        assert model.epoch > 0
        assert model.batch_number > 0
