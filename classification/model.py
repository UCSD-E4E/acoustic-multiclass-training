# pylint: disable=E1123:

""" Contains the model class

    Model: model with forward pass method. Generated automatically from a timm model

"""
import torch
from torch import nn
import tensorflow as tf

import requests
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
                 num_classes,
                 cnn_model_name="tf_efficientnet_b4",
                 pretrained=True,
                 CONFIG=None,
                 device="cuda"):
        """ Initializes the model
        """
        super().__init__()
        self.config = CONFIG
        self.num_classes = num_classes
        # See config.py for list of recommended models
        self.cnn_model = timm.create_model(cnn_model_name, pretrained=pretrained, num_classes=num_classes)
        self.birdnet = self.get_birdnet_model()
        self.input_index = self.birdnet.get_input_details()[0]["index"]
        self.output_index = self.birdnet.get_output_details()[0]["index"] - 1
        self.loss_fn = None

    def get_birdnet_model(self):
        # https://github.com/kahst/BirdNET-Analyzer/blob/main/LICENSE
        r = requests.get(
            'https://raw.githubusercontent.com/kahst/BirdNET-Analyzer/b32cdc54c9f2344b028e6378e9eae66e39110d27/checkpoints/V2.4/BirdNET_GLOBAL_6K_V2.4_Model_FP16.tflite'
            )
        birdnet_model_path = os.path.join(".", "model", "BirdNET_GLOBAL_6K_V2.4_Model_FP16.tflite")
        print(birdnet_model_path)
        if not os.path.exists(birdnet_model_path):
            open(birdnet_model_path, 'wb').write(request.content)
            
        return tf.lite.Interpreter(model_path=birdnet_model_path)
    
    def forward(self, images):
        """ Forward pass of the model
        """
        # this section will be based on
        # https://github.com/kahst/BirdNET-Analyzer/blob/0d624d910f4f731f8f4ae1689ca67c398f34469f/model.py#L365
        self.birdnet.resize_tensor_input(self.input_index, images.shape)
        self.birdnet.allocate_tensors()

        self.birdnet.set_tensor(self.input_index, np.array(images, dtype="float32"))
        self.birdnet.invoke()
        features = self.birdnet.get_tensor(output_index)

        birdnet_embeddings = torch.Tensor(features).to(self.device)
        cnn_embeddings = self.cnn_model(images)

        x = torch.cat((birdnet_embeddings, cnn_embeddings), 0)
        print(x)
        #birdnet_embeddings


        return self.cnn_model(images)

    def create_loss_fn(self,train_dataset):
        """ Returns the loss function and sets self.loss_fn
        """
        return cross_entropy_loss_fn(self, train_dataset)