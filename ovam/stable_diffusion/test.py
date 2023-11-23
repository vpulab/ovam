# Use to run without install as a package
import sys

sys.path.append("../../")

import unittest

import numpy as np
import torch
from diffusers import StableDiffusionPipeline

from ovdaam.stable_diffusion import StableDiffusionHooker
from ovdaam.utils import get_device, set_seed


class TestStableDiffusion(unittest.TestCase):
    """Basic test to check if the Stable Diffusion model is working"""

    def setUp(self) -> None:
        self.device = get_device()

    def _base_test(
        self,
        model_id: str,
        internal_size: int,
        prompt: str = "A car in an urban environment",
        n_steps: int = 10,
        seed: int = 42,
    ) -> None:
        """
        Battery of tests for the Stable Diffusion model and
        the StableDiffusionHooker.
        """
        pipe = StableDiffusionPipeline.from_pretrained(model_id)
        pipe = pipe.to(self.device)

        # Generate an image with the original pipeline
        with self.subTest("Test generation of image"):
            set_seed(seed, self.device)
            out = pipe(prompt, num_inference_steps=n_steps)

        # Repeat the generation with the hooked pipeline
        with self.subTest("Test hooked pipeline"):
            with StableDiffusionHooker(pipe) as hooker:
                set_seed(seed, self.device)
                out_hooked = pipe(prompt, num_inference_steps=n_steps)

        with self.subTest("Test hooked pipeline do not alter the result"):
            img1 = np.array(out.images[0])
            img2 = np.array(out_hooked.images[0])
            np.testing.assert_array_equal(img1, img2)

        with self.subTest("Test daam module creation"):
            # Create a daam callable
            daam_module = hooker.daam()

        with self.subTest("Generate a heatmap for the word"):
            with torch.no_grad():
                word = prompt.split(" ")[0]
                heatmap = daam_module(word)
            self.assertEqual(heatmap.shape, (1, 1, internal_size, internal_size))

        with self.subTest("Generate a heatmap for 2 words"):
            with torch.no_grad():
                sentence_2 = " ".join(prompt.split(" ")[:2])
                heatmap = daam_module(sentence_2)
            self.assertEqual(heatmap.shape, (1, 2, internal_size, internal_size))

        with self.subTest("Collect hidden states of 2 images"):
            with StableDiffusionHooker(pipe) as hooker:
                set_seed(seed, self.device)
                out_hooked = pipe(prompt, num_inference_steps=n_steps)
                out_hooked = pipe(prompt, num_inference_steps=n_steps)

        with self.subTest("Generate daam module with 2 images"):
            daam_module = hooker.daam()

        with self.subTest("Test daam with 2 images and 3 words"):
            with torch.no_grad():
                sentence_3 = " ".join(prompt.split(" ")[:3])
                heatmap = daam_module(sentence_3)

            self.assertEqual(heatmap.shape, (2, 3, internal_size, internal_size))

        del daam_module
        del pipe

    def test_stable_diffusion_15(self):
        """Test for Stable Diffusion 1.5"""
        model_id = "runwayml/stable-diffusion-v1-5"
        self._base_test(model_id=model_id, internal_size=64)

    def test_stable_diffusion_2_base(self):
        """Test for Stable Diffusion 2 Base"""
        model_id = "stabilityai/stable-diffusion-2-base"
        self._base_test(model_id, internal_size=64)

    def test_stable_diffusion_2_1(self):
        """Test for Stable Diffusion 2.1"""
        model_id = "stabilityai/stable-diffusion-2-1"
        self._base_test(model_id, internal_size=96)


if __name__ == "__main__":
    unittest.main()
