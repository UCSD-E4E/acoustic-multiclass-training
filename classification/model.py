# pylint: disable=E1123:

""" Contains the model class

    Model: model with forward pass method. Generated automatically from a timm model

"""
from torchaudio import transforms as audtr
import torch
from torch import nn
import tensorflow as tf
import os
import requests
import numpy as np
# timm is a library of premade models
import timm
import requests

# birdnetlib is a library wrapper for birdnet

# pylint: disable=too-many-instance-attributes
class TimmModel(nn.Module):
    """ Efficient net neural network
    """
    # pylint: disable=too-many-arguments
    def __init__(self,
                 num_classes,
                 model_name="tf_efficientnet_b4",
                 pretrained=True,
                 CONFIG=None):
        """ Initializes the model
        """
        super().__init__()
        self.config = CONFIG
        self.num_classes = num_classes
        # See config.py for list of recommended models
        self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
        self.loss_fn = None
    
    def forward(self, images):
        """ Forward pass of the model
        """
        return self.model(images)

    def create_loss_fn(self,train_dataset):
        """ Returns the loss function and sets self.loss_fn
        """
        return cross_entropy_loss_fn(self, train_dataset)

def cross_entropy_loss_fn(self,train_dataset):
    """ Returns the loss function and sets self.loss_fn
    """
    if not self.config.imb: # normal loss
        self.loss_fn = nn.CrossEntropyLoss()
    else: # weighted loss
        self.loss_fn = nn.CrossEntropyLoss(
            weight=torch.tensor(
                [1 / p for p in train_dataset.class_id_to_num_samples.values()]
            ).to(self.device))
    return self.loss_fn


# pylint: disable=too-many-instance-attributes
class BirdnetYCNNModel(nn.Module):
    """ Inspired by the 6th place winner of birdclef2023

    This model combines a finetuned cnn(from timms) and the embeddings of birdnet
    into a fullyconnected layer

    https://www.kaggle.com/competitions/birdclef-2023/discussion/412708
    """
    # pylint: disable=too-many-arguments
    def __init__(self,
                 num_classes=130,
                 cnn_model_name="tf_efficientnet_b4",
                 pretrained=True,
                 CONFIG=None,
                 device="cuda:0"):
        """ Initializes the model
        """
        super().__init__()
        self.config = CONFIG
        self.device = device
        self.num_classes = num_classes
        self.birdnet = self.get_birdnet_model()
        self.input_index = self.birdnet.get_input_details()[0]["index"]
        self.output_index = self.birdnet.get_output_details()[0]["index"] - 1
        self.embedding_shape = 1024 #From birdnet docs
        self.loss_fn = None

        # See config.py for list of recommended models
        self.cnn_model = timm.create_model(
            cnn_model_name, pretrained=pretrained, num_classes=self.embedding_shape
        ).to(device)

        self.embedding = nn.Linear(self.embedding_shape * 2, self.embedding_shape).to(device)
        self.fc = nn.Linear(self.embedding_shape, num_classes).to(device)
               

    def get_birdnet_model(self):
        
        # https://github.com/kahst/BirdNET-Analyzer/blob/main/LICENSE
        birdnet_model_path = os.path.join(".", "model", "BirdNET_GLOBAL_6K_V2.4_Model_FP16.tflite")
        if not os.path.exists(birdnet_model_path):
            print("downloading birdnet")
            r = requests.get(
                'https://raw.githubusercontent.com/kahst/BirdNET-Analyzer/b32cdc54c9f2344b028e6378e9eae66e39110d27/checkpoints/V2.4/BirdNET_GLOBAL_6K_V2.4_Model_FP16.tflite'
                )
            open(birdnet_model_path, 'wb').write(r.content)
            
            print("download completed")
        return tf.lite.Interpreter(model_path=birdnet_model_path)
    
    def birdnet_forward(self, audio):
        # this section will be based on
        # https://github.com/kahst/BirdNET-Analyzer/blob/0d624d910f4f731f8f4ae1689ca67c398f34469f/model.py#L365
        
        #TODO This input is only 3 second files, we should ensure in the future
        # Its 3 second with the bn sample rate of 48_000
        # that the audio is this shape for training a model 
        audio = audio[:, :144000]    

        self.birdnet.resize_tensor_input(self.input_index, [len(audio), *audio[0].shape])
        self.birdnet.allocate_tensors()
        self.birdnet.set_tensor(self.input_index, np.array(audio.cpu(), dtype="float32"))
        self.birdnet.invoke()

        features = self.birdnet.get_tensor(self.output_index)
        birdnet_embeddings = torch.Tensor(features).to(self.device)
        
        return birdnet_embeddings
        
    def forward(self, images, audio):
        """ Forward pass of the model
        """
        
        bnt_embeddings = self.birdnet_forward(audio).to(self.device)
        cnn_embeddings = self.cnn_model(images).to(self.device)
        
        x = torch.cat((bnt_embeddings, cnn_embeddings), -1)
        x = self.embedding(x)
        x = self.fc(x)
        return x

    def create_loss_fn(self,train_dataset):
        """ Returns the loss function and sets self.loss_fn
        """
        return cross_entropy_loss_fn(self, train_dataset)

def test():
    #Startup
    torch.multiprocessing.set_start_method('spawn')
    CONFIG = get_config()
    set_seed(CONFIG.seed)
    train_ds, valid_ds = get_datasets(CONFIG=CONFIG)

    #Testing
    print("Starting Model Test")
    model = BirdnetYCNNModel(CONFIG=CONFIG) #, device=device

    #simulate a batch
    image, target, audio = train_ds[0]
    audio = audio.reshape(1,-1)
    print(image.shape)
    image = image.reshape(1,3, 194, 229)
    print("input shape for the audio ",audio.shape)
    print("input shape for the image ",image.shape)

    print("try passing data into model")
    
    out = model(image, audio)
    print(out.shape)

    print("i have no idea what happens next")
    print(model.state_dict())
    
    #https://discuss.pytorch.org/t/check-if-models-have-same-weights/4351/7
    old = model.state_dict().__str__()
    print(old == model.state_dict().__str__() )


if __name__ == "__main__":
    #Prevents circular dependecies, this is just for testing :)
    from dataset import get_datasets
    from config import get_config
    from utils import set_seed, print_verbose
    test()