"""Inkling-2-Mini on Arc Pro B70 (XPU), served with the cookbook DSpark recipes.

The kernel tests under ``test/registered/kernels/ops/*/test_inkling2_*_xpu.py``
cover the XPU operators in isolation; this one boots the real checkpoint with the
cookbook's TP2 / 512K and TP4 / 1M commands, so it catches what only shows up when
the pieces meet: a startup or graph-capture failure, a collective or expert-parallel
regression that keeps the server healthy while the answers go wrong, a draft that
stops matching the target, and breakage in the image and tool-call paths.
"""

import base64
import io
import json
import os
import unittest

import requests
import torch

from sglang.test.ci.ci_register import register_xpu_ci
from sglang.test.kits.eval_accuracy_kit import GSM8KMixin
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
    terminate_and_kill_process_tree,
    try_cached_model,
)

register_xpu_ci(est_time=2400, suite="nightly-kernel-main-test-4-gpu-xpu", nightly=True)

MODEL = os.environ.get(
    "INKLING2_MINI_TEST_MODEL_PATH", "thinkingmachines/Inkling-2-Mini"
)

# The cookbook's Intel Arc Pro B70 environment: the DSpark runtime switches, and
# oneCCL staging its eager collectives through its own registered buffers.
COOKBOOK_ENV = {
    "SGLANG_OPT_DSV41_DSPARK_MID_FOLD": "1",
    "SGLANG_OPT_DSV41_DSPARK_PREPARE_INPUTS_ONCE": "1",
    "SGLANG_OPT_DSV41_DSPARK_SKIP_PREFILLED_INPUTS": "1",
    "SGLANG_DSPARK_FUSED_ACCEPT_TAIL": "1",
    "SGLANG_OPT_DSV41_DSPARK_POST_FOLD": "1",
    "SGLANG_OPT_DSV41_DSPARK_ONE_LAUNCH_COMMIT_STAGE": "1",
    "SGLANG_OPT_DSV41_TARGET_VERIFY_METADATA_GLUE": "1",
    "CCL_SYCL_ALLREDUCE_TMP_BUF": "1",
    "CCL_SYCL_ALLGATHERV_TMP_BUF": "1",
    "CCL_SYCL_REDUCE_SCATTER_TMP_BUF": "1",
    "CCL_SYCL_BROADCAST_TMP_BUF": "1",
    "CCL_SYCL_ALLTOALL_TMP_BUF": "1",
}

# The cookbook commands minus --tp/--ep and the context sizes (see the classes).
COOKBOOK_COMMON_ARGS = [
    "--device",
    "xpu",
    "--attention-backend",
    "inkling2_xpu",
    "--page-size",
    "64",
    "--chunked-prefill-size",
    "8192",
    "--mem-fraction-static",
    "0.95",
    "--swa-full-tokens-ratio",
    "0.03",
    "--max-running-requests",
    "8",
    "--hybrid-cache-checkpoints-level",
    "2",
    "--max-mamba-cache-size",
    "64",
    "--speculative-algorithm",
    "DSPARK",
    "--speculative-num-draft-tokens",
    "8",
    "--speculative-draft-attention-backend",
    "inkling2_xpu",
    "--cuda-graph-config",
    '{"decode":{"backend":"full","bs":[1,2,4,8],"max_bs":8},"prefill":{"backend":"full",'
    '"bs":[64,128,256,512,1024,2048,4096,8192],"max_bs":8192,"full_prefill_max_req":4,'
    '"max_seq_len":8192}}',
    "--enable-multimodal",
    "--reasoning-parser",
    "inkling",
    "--tool-call-parser",
    "inkling2",
    "--grammar-backend",
    "xgrammar",
    "--enable-cache-report",
    "--skip-server-warmup",
    "--watchdog-timeout",
    "1200",
]

# Measured on B70 with these settings (200 questions): TP2 95.5%, TP4 93.5% and
# 94.5%. The floor sits about three binomial sigma below, so only a real
# accuracy drop trips it.
GSM8K_THRESHOLD = 0.88

# Measured average accept length over GSM8K: 3.6 at TP2 and TP4. A draft that
# stops matching the target falls toward 1.0.
ACCEPT_LENGTH_THRESHOLD = 3.0


def _shapes_image_url() -> str:
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (448, 448), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle([60, 60, 220, 220], fill="red")
    draw.ellipse([240, 240, 400, 400], fill="blue")
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


@unittest.skipUnless(torch.xpu.is_available(), "Intel XPU not available")
class TestInkling2MiniTP2XPU(GSM8KMixin, CustomTestCase):
    """TP2 / EP2 at 524,288 tokens: the cookbook's two-card command."""

    shape_args = [
        "--tp",
        "2",
        "--ep",
        "2",
        "--context-length",
        "524288",
        "--max-total-tokens",
        "528384",
    ]

    gsm8k_backend = "sgl_eval"
    gsm8k_score_threshold = GSM8K_THRESHOLD
    gsm8k_accept_length_thres = ACCEPT_LENGTH_THRESHOLD
    gsm8k_num_examples = 200
    gsm8k_num_threads = 8
    gsm8k_max_tokens = 8192
    gsm8k_temperature = 0.0
    gsm8k_top_p = 1.0

    @classmethod
    def setUpClass(cls):
        cls.model = try_cached_model(MODEL)
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=3600,
            other_args=cls.shape_args + COOKBOOK_COMMON_ARGS,
            env=COOKBOOK_ENV,
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process:
            terminate_and_kill_process_tree(cls.process)

    def _chat(self, messages, **kwargs) -> dict:
        response = requests.post(
            self.base_url + "/v1/chat/completions",
            json={
                "model": self.model,
                "messages": messages,
                "temperature": 0,
                "max_tokens": 1024,
                **kwargs,
            },
            timeout=600,
        )
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()["choices"][0]["message"]

    def test_image_understanding(self):
        """The vision tokens must reach the language model: a processor or
        encoder regression leaves text answers intact but loses the image."""
        message = self._chat(
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "What shapes and colors are in this image? "
                            "Answer briefly.",
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": _shapes_image_url()},
                        },
                    ],
                }
            ]
        )
        answer = message["content"].lower()
        for word in ("red", "blue", "square", "circle"):
            self.assertIn(word, answer)

    def test_tool_call(self):
        """Tools render through the chat template and the model's call parses
        back into an OpenAI tool_calls entry."""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the current weather for a city",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                },
            }
        ]
        message = self._chat(
            [
                {
                    "role": "user",
                    "content": "What's the weather in Paris right now? Use the tool.",
                }
            ],
            tools=tools,
        )
        self.assertTrue(message["tool_calls"], message)
        call = message["tool_calls"][0]["function"]
        self.assertEqual(call["name"], "get_weather")
        self.assertEqual(json.loads(call["arguments"]), {"city": "Paris"})


class TestInkling2MiniTP4XPU(TestInkling2MiniTP2XPU):
    """TP4 / EP4 at 1,048,576 tokens: the cookbook's four-card command. Two KV
    heads per rank take different split-K and packing than TP2's four."""

    shape_args = [
        "--tp",
        "4",
        "--ep",
        "4",
        "--context-length",
        "1048576",
        "--max-total-tokens",
        "1052672",
    ]


if __name__ == "__main__":
    unittest.main()
