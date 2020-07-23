local config = import "ocr.jsonnet";

config + {
  model +: {
    decoder +: {
      "type": "lmpl_searnn_decoder",
      "rollin_mode":  std.extVar("rollin_mode"),
      "rollout_mode": std.extVar("rollout_mode"),
      "rollout_cost_function": {
        "type": "hamming",
      },
    },
  },
}