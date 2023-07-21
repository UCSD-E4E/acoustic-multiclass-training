"""
Unit tests for the classification folder.
Note: must run from the root directory of the project.
Example: `python -m classification.test.test`
"""
# Class names are self explanatory
# pylint: disable=missing-class-docstring
# Import order is right, sys path messes it up
# pylint: disable=wrong-import-order
# pylint: disable=wrong-import-position
import os
import sys
import unittest

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
import utils
from augmentations import (BackgroundNoise, LowpassFilter, Mixup, RandomEQ,
                           SyntheticNoise)
from models.early_stopper import EarlyStopper
from models.timm_model import TimmModel
from dataset import get_datasets, make_dataloaders
from train import run_batch, map_metric, save_model

cfg = config.cfg
dataset, valid_ds = get_datasets()
if torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"


class TestAugmentations(unittest.TestCase):
    def test_augs(self):
        """ Test all augmentations and verify output is correct size """
        audio, label = dataset.get_annotation(0)
        TestUtils.assert_one_hot(label, dataset.num_classes)
        label_id = (label == 1.0).nonzero()[0].item()
        mixup = Mixup(dataset.samples, dataset.class_to_idx,
                      cfg.sample_rate, dataset.num_classes,
                      (0.25, 0.25), 1.0)
        new_audio, new_label = mixup(audio, label)
        # Assert mixup output
        assert new_audio.shape == audio.shape, "Mixup should not change shape"
        assert new_label[label_id] == 0.75 or new_label[label_id] == 1.0, \
            "Mixup label should be equal to alpha"
        assert new_label.sum() == 1.0, "Mixup label should sum to 1"

        noise_colors = ["white", "pink", "brown", "blue", "violet"]
        synth = [SyntheticNoise(col, 0.2) for col in noise_colors]
        rand_eq = RandomEQ()
        bg_noise = BackgroundNoise(0.2, 5)
        lowpass = LowpassFilter(200, 0.707)
        augs = [*synth, rand_eq, bg_noise, lowpass]
        augmented_audio = [aug.forward(audio) for aug in augs]
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
        spec = dataset.to_image(torch.zeros(5*cfg.sample_rate).to(DEVICE))
        assert spec.sum() == 0, "Spectrogram of zero audio should be zero"

    def test_cropping(self):
        """ Test crop_audio and pad_audio in dataset.py """
        num_samples = 5 * cfg.sample_rate
        crop_len = dataset.crop_audio(torch.zeros(num_samples * 2)).shape[0]
        pad_len = dataset.pad_audio(torch.zeros(num_samples // 2)).shape[0]
        assert crop_len == num_samples, "crop_audio should crop to num_samples"
        assert pad_len == num_samples, "pad_audio should pad to num_samples"


class TestModel(unittest.TestCase):
    def test_model(self):
        """ Test model.forward output is correct """
        batch_size = 8
        num_classes = 12
        model = TimmModel(num_classes,"tf_efficientnet_b4",True).to(DEVICE)
        model.eval()
        out = model.forward(torch.zeros((batch_size,3,224,224)).to(DEVICE))
        assert str(out.device).startswith(DEVICE), "Model output device is wrong"
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
        model = TimmModel(dataset.num_classes, "tf_efficientnet_b4", True).to(DEVICE)
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
            audio, label = utils.get_annotation(dataset.samples, i, dataset.class_to_idx, 
                                                cfg.sample_rate, num_samples, "cpu")
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
    os.system("pylint classification")
    print("Running pyright")
    os.system("pyright classification")
