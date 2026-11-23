from sglang.test.ci.ci_register import register_xpu_ci

register_xpu_ci(est_time=5, suite="extra-test-1-gpu-xpu")

import unittest

import torch

from sglang.kernels.ops.speculative.dspark.markov_sample import MarkovSampleStep
from sglang.test.test_utils import CustomTestCase


@unittest.skipUnless(torch.xpu.is_available(), "XPU only")
class TestMarkovSampleStepXPU(CustomTestCase):
    def test_vocab_limit_mask_records_in_an_xpu_graph(self):
        """The vocab-tail mask of a padded shard must record in an XPU graph and
        replay to the eager result; boolean-index compaction cannot be recorded."""
        device = torch.device("xpu")
        g = torch.Generator(device=device).manual_seed(3)
        rows, width, limit = 8, 4096, 4000
        base = torch.randn(rows, width, generator=g, device=device)
        bias = torch.randn(rows, width, generator=g, device=device)
        temps = torch.zeros(rows, device=device)
        offset = torch.tensor([7], dtype=torch.int64, device=device)

        def sample():
            return MarkovSampleStep.torch(
                base_logits=base,
                bias=bias,
                temperatures=temps,
                key=12345,
                offset=offset,
                step=2,
                step_dtype=torch.float32,
                vocab_limit=limit,
            )

        eager = sample()
        self.assertTrue(bool((eager < limit).all()))
        stream = torch.xpu.Stream()
        stream.wait_stream(torch.xpu.current_stream())
        with torch.xpu.stream(stream):
            sample()
        torch.xpu.current_stream().wait_stream(stream)
        graph = torch.xpu.XPUGraph()
        with torch.xpu.graph(graph):
            recorded = sample()
        graph.replay()
        torch.xpu.synchronize()
        self.assertTrue(torch.equal(recorded, eager))


if __name__ == "__main__":
    unittest.main()
