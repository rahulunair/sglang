"""Unit tests for srt/layers/attention/inkling2_xpu_backend"""

import unittest

from sglang.srt.layers.attention.inkling2_xpu_backend import _decode_splits
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestDecodeSplits(CustomTestCase):
    def test_batch_invariant_splits_ignore_the_batch(self):
        """Deterministic inference needs a request's split count, and so its LSE
        merge order, to stay the same whatever else is in the batch."""
        for kv_heads in (1, 2, 4, 8):
            for is_swa in (False, True):
                counts = {
                    _decode_splits(
                        requests=requests,
                        kv_heads=kv_heads,
                        is_swa=is_swa,
                        batch_invariant=True,
                    )
                    for requests in (1, 2, 3, 8, 64)
                }
                self.assertEqual(len(counts), 1, (kv_heads, is_swa, counts))


if __name__ == "__main__":
    unittest.main()
