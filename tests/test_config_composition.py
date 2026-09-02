from __future__ import annotations

import unittest

from scripts.sketchformer.config import compose_training_config


class ConfigCompositionTest(unittest.TestCase):
    def test_default_training_config_targets_tok_dict(self) -> None:
        config = compose_training_config("configs/train.yaml", experiment="smoke_test")

        self.assertEqual(config["data"]["format"]["type"], "tok_dict")
        self.assertEqual(config["model"]["name"], "sketchformer_tok_dict")
        model_tokens = config["model"]["input"]["token_dictionary"]
        data_tokens = config["data"]["format"]["token_dictionary"]
        for key in (
            "codebook_size",
            "motion_token_offset",
            "pad_token_id",
            "sep_token_id",
            "sos_token_id",
            "eos_token_id",
            "vocab_size",
        ):
            self.assertEqual(model_tokens[key], data_tokens[key])
        self.assertEqual(model_tokens["pad_token_id"], 0)
        self.assertEqual(model_tokens["sep_token_id"], 1001)
        self.assertEqual(model_tokens["sos_token_id"], 1002)
        self.assertEqual(model_tokens["eos_token_id"], 1003)
        self.assertEqual(model_tokens["vocab_size"], 1004)
        self.assertEqual(config["trainer"]["checkpointing"]["monitor"], "val/token_loss")


if __name__ == "__main__":
    unittest.main()
