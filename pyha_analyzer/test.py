"""
Unit tests for the pyha_analyzer package
To run: `python -m pyha_analyzer.test`
"""
# Class names are self explanatory
# pylint: disable=missing-class-docstring
import os
import unittest
from pathlib import Path

import torch
import importlib_resources as pkg_resources
import wandb

from pyha_analyzer import config
from pyha_analyzer import utils
from pyha_analyzer.augmentations import (BackgroundNoise, LowpassFilter, Mixup, RandomEQ,
                                         HighpassFilter, SyntheticNoise)
from pyha_analyzer.chunking_methods.sliding_chunks import convolving_chunk
from pyha_analyzer.models.early_stopper import EarlyStopper
from pyha_analyzer.models.timm_model import TimmModel
from pyha_analyzer.dataset import get_datasets, make_dataloaders
from pyha_analyzer.train import run_batch, map_metric, save_model

cfg = config.cfg
wandb.init(mode="disabled")
dataset, valid_ds = get_datasets()


class TestAugmentations(unittest.TestCase):
    def test_augs(self):
        """ Test all augmentations and verify output is correct size """
        audio, label = utils.get_annotation(dataset.samples, 0, dataset.class_to_idx)
        TestUtils.assert_one_hot(label, dataset.num_classes)
        label_id = (label == 1.0).nonzero()[0].item()
        mixup_alpha = 0.25
        cfg.mixup_alpha_range = [mixup_alpha, mixup_alpha] # type: ignore
        cfg.mixup_ceil_interval = 1. # type: ignore
        num_clips_range = list(range(cfg.mixup_num_clips_range[0],
                                     cfg.mixup_num_clips_range[1] + 1))
        mixup = Mixup(dataset.samples, dataset.class_to_idx, cfg)
        new_audio, new_label = mixup(audio, label)
        # Assert mixup output
        assert new_audio.shape == audio.shape, "Mixup should not change shape"
        assert new_label[label_id] == 1., \
                "Mixup labels should be rounded up to nearest interval"
        assert float(new_label.sum()) in num_clips_range, "Mixup label should add to two 2"

        augs = []
        noise_colors = ["white", "pink", "brown", "blue", "violet"]
        for col in noise_colors:
            cfg.noise_type = col # type: ignore
            augs.append(SyntheticNoise(cfg))
        augs+= [RandomEQ(cfg),
                BackgroundNoise(cfg),
                LowpassFilter(cfg),
                HighpassFilter(cfg)]
        augmented_audio = [aug(audio) for aug in augs]
        for aug_audio in augmented_audio:
            assert aug_audio.shape == audio.shape, "Augmented audio should not change shape"


class TestConfig(unittest.TestCase):
    def test_config(self):
        """ Confirm config is a singleton """
        config1 = config.Config()
        config2 = config.Config()
        assert config1 == config2, "Config should be a singleton class"


class TestDataset(unittest.TestCase):
    def test_dataset(self):
        """ Verify dataset is correct size and has correct number of classes """
        assert dataset.num_classes == valid_ds.num_classes
        assert len(dataset.class_to_idx) == dataset.num_classes
        assert len(valid_ds.class_to_idx) == valid_ds.num_classes
        assert dataset.num_samples == 5 * cfg.sample_rate, "Dataset clips should be 5 seconds long"

    def test_spectrogram(self):
        """ Test to_image in dataset.py """
        spec = dataset.to_image(torch.zeros(5*cfg.sample_rate).to(cfg.prepros_device))
        assert (spec != spec.mean().item()).sum() == 0, "Spectrogram of no audio should be constant"


class TestModel(unittest.TestCase):
    def test_model(self):
        """ Test model.forward output is correct """
        batch_size = 8
        num_classes = 12
        model = TimmModel(num_classes,"tf_efficientnet_b4",True).to(cfg.device)
        model.eval()
        out = model.forward(torch.zeros((batch_size,3,224,224)).to(cfg.device))
        assert str(out.device).startswith(cfg.device), "Model output device is wrong"
        assert out.shape == (batch_size,num_classes), "Model output shape not matched"

    def test_early_stopper(self):
        """ Tests early stopper performs properly """
        # Test if stops after patience over
        early_stopper = EarlyStopper(patience=2, min_delta=0.01)
        for x, stop in [(1, False),(0.9, False),(0.9, True)]:
            assert early_stopper.early_stop(x) == stop

        # Test zero patience
        early_stopper = EarlyStopper(patience=0, min_delta=0.01)
        for x, stop in [(1, False),(1.1, False),(0.9, True)]:
            assert early_stopper.early_stop(x) == stop
        
        # Test patience reset
        early_stopper = EarlyStopper(patience=2, min_delta=0.1)
        for x, stop in [(0.7, False),(0.5, False),(0.9, False),(0.7, False),(0.7, True)]:
            assert early_stopper.early_stop(x) == stop


class TestTrain(unittest.TestCase):
    def test_train_batch(self):
        """ Tests if a training batch runs properly """
        cfg.jobs = 0 # type: ignore
        train_dl, _ = make_dataloaders(dataset, valid_ds)
        model = TimmModel(dataset.num_classes, "tf_efficientnet_b4", True).to(cfg.device)
        model.create_loss_fn(dataset)
        mels, labels = next(iter(train_dl))
        loss, outputs = run_batch(model, mels, labels)
        assert loss >= 0, "Loss should be positive"
        assert outputs.shape == labels.shape, "Model output shape should match labels shape"

    def test_map(self):
        """ Tests if macro average precision meets expected values """
        map_val = map_metric(
            torch.tensor([[0.1,0.2,0.3,0.4],[0.1,0.2,0.3,0.4]]),
            torch.tensor([[1,0,0,0],[0,0,0,1]]).to(dtype=torch.float),
            4)
        assert map_val == 0.5, "mAP expected to be 0.5"
        map_val = map_metric(
            torch.tensor([[1,0,0,0],[0,0,0,1]]).to(dtype=torch.float),
            torch.tensor([[1,0,0,0],[0,0,0,1]]).to(dtype=torch.float),
            4)
        assert map_val == 1.0, "mAP expected to be 1"

    def test_model_save(self):
        """ Tests that model saving does not crash """
        model = TimmModel(10,"tf_efficientnet_b4",True)
        save_model(model)


class TestChunking(unittest.TestCase):
    def test_yan_chunking(self):
        """ Test yan chunking methods for expected values """
        conv = convolving_chunk({"OFFSET": 4, "DURATION": 3,"CLIP LENGTH": 10}, 5, 0.4, 0.3, 0)
        assert len(conv) == 3, "convolving_chunk should return 3 chunks"
        for row in conv:
            assert row["DURATION"] == 5, "convolving_chunk should return correct duration"
        assert conv[0]["OFFSET"] == 4, "convolving_chunk should return correct offset"
        assert conv[1]["OFFSET"] == 4.5, "convolving_chunk should return correct offset"
        assert conv[2]["OFFSET"] == 2, "convolving_chunk should return correct offset"
        conv = convolving_chunk({"OFFSET": 3, "DURATION": 0.1, "CLIP LENGTH": 20}, 5, 0.4, 0.3, 0)
        assert len(conv) == 0, "convolving_chunk should return 0 chunks when under minimum"
        conv = convolving_chunk({"OFFSET": 10, "DURATION": 20, "CLIP LENGTH": 50}, 5, 0.4, 0.3, 0)
        assert len(conv) == 5, "convolving_chunk should return 5 chunks"
        for row in conv:
            assert row["DURATION"] == 5, "convolving_chunk should return correct duration"
        for i, row in enumerate(conv):
            assert row["OFFSET"] == 10 + i * 3.5, "convolving_chunk should return correct offset"

class TestUtils(unittest.TestCase):
    def test_rand(self):
        """ Test randint and rand float 1000 times """
        low, high = 0, 10
        for _ in range(1000):
            rand_int = utils.randint(low, high)
            assert low <= rand_int < high, "randint should be in range [low, high)"
            rand_float = utils.rand(float(low), float(high))
            assert low <= rand_float < high, "rand should be in range [low, high)"

    @classmethod
    def assert_one_hot(cls, one_hot: torch.Tensor, num_classes: int):
        """ Asserts that a one_hot output is correct """ 
        assert one_hot.sum() == 1.0, "one_hot should sum to 1"
        assert (one_hot==1.0).nonzero().shape[0] == 1, "one_hot should only have one 1"
        assert one_hot.shape[0] == num_classes, "one_hot should have num_classes length"

    def test_one_hot(self):
        """ Tests one_hot 100 times """
        num_classes = 20
        for _ in range(100):
            one_hot = utils.one_hot(torch.tensor(utils.randint(0, num_classes)),
                                    num_classes, on_value=1., off_value=0.)[0]
            self.assert_one_hot(one_hot, num_classes)

    def test_get_annotation(self):
        """ Tests get_annotation 100 times """
        num_samples = 5 * cfg.sample_rate
        for i in range(20):
            audio, label = utils.get_annotation(dataset.samples, i, dataset.class_to_idx)
            assert audio.shape[0] == num_samples, "audio should be num_samples long"
            self.assert_one_hot(label,dataset.num_classes)
            assert str(audio.device) == "cpu", "get annotation returned wrong device"
            assert str(label.device) == "cpu", "get annotation returned wrong device"

    def test_cropping(self):
        """ Test utils crop_audio and pad_audio """
        sample_rate = 16000
        length = 5
        num_samples = length * sample_rate
        crop_len = utils.crop_audio(torch.zeros(num_samples * 2),num_samples).shape[0]
        pad_len = utils.pad_audio(torch.zeros(num_samples // 2),num_samples).shape[0]
        assert crop_len == num_samples, "crop_audio should crop to num_samples"
        assert pad_len == num_samples, "pad_audio should pad to num_samples"


if __name__=="__main__":
    unittest.main(exit=False)
    print("Running pylint")
    main_dir = pkg_resources.files("pyha_analyzer")
    pylint_path = Path(str(main_dir), "..", ".pylintrc")
    if pylint_path.is_file():
    #if (main_dir / ".." / ".pylintrc").is_file():
        os.system(f"pylint \"{main_dir}/../entry.py\" " + \
                  f"\"{main_dir}/*.py\" \"{main_dir}/**/*.py\"" +
                  f" --rcfile=\"{pylint_path}\"")
    else:
        print("Pylintrc not found, skipping pylint")
    print("Running pyright")
    os.system(f"pyright \"{main_dir}\"")
