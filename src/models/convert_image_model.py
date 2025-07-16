import torch
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from brainscore_vision.model_helpers.activations.temporal.model.pytorch import PytorchWrapper
from brainscore_vision.model_helpers.activations.temporal.model.base import ActivationWrapper
from brainscore_vision.model_helpers.activations.temporal.core.inferencer import CausalInferencer
from brainscore_vision.model_helpers.activations.temporal.inputs import Video, Image
from brainscore_vision.model_helpers.activations.temporal.inputs.image import get_image_size, PILImage
from brainscore_vision.model_helpers.activations.pytorch import load_preprocess_images, preprocess_images
from functools import partial
from collections import OrderedDict
import numpy as np
from types import MethodType


DEFAULT_FPS = 25

# submit an image model to be temporal model
def _image_to_temporal_model(brain_model, layers, batchsize):

    if "pixels" == brain_model.identifier:
        # submit pixel model
        layers = ['pixels']
        identifier = "pixels"
        wrapper = PixelsActivationWrapper()
    else:
        # submit pytorch model
        identifier = brain_model.identifier
        activations_model = brain_model.activations_model
        model = activations_model._model
        if hasattr(activations_model._extractor, "inferencer"):
            preprocessing = get_my_image_model_video_preprocessing(activations_model._extractor.inferencer._executor.preprocess)
        else:
            preprocessing = activations_model._extractor.preprocess
            if hasattr(preprocessing, "func") and preprocessing.func is load_preprocess_images:
                preprocess_kwargs = preprocessing.keywords if hasattr(preprocessing, "keywords") else {}
                preprocessing = partial(_image_model_video_preprocessing, **preprocess_kwargs) if preprocessing is not None else None
            else:
                print("Converting from customized preprocessing to temporary-file preprocessing...")
                print("This can be highly inefficient.")
                if preprocessing is not None:
                    preprocessing = partial(_tmpfile_image_model_video_preprocessing, custom_preprocess_images=preprocessing)
                else:
                    preprocessing = None

        # we have to guess
        layer_activation_format = {}
        for layer in layers:
            if "fc" in layer or "cls" in layer or "class" in layer or 'avgpool' in layer:
                layer_activation_format[layer] = "C"
            else:
                layer_activation_format[layer] = "CHW"
        print("We are guessing the layer activation format.")
        for k, v in layer_activation_format.items():
            # print(f"{k}: {v}")
            layer_activation_format[k] = "T" + v  # add the temporal dimension

        # don't forget to split the temporal dimension in the process output function
        def process_activation(layer, layer_name, inputs, output):
            # (torch.nn.Module, str, torch.Tensor, torch.Tensor) -> torch.Tensor
            BT = output.shape[0]
            T = BT // batchsize
            output = output.view(batchsize, T, *output.shape[1:])
            return output

        wrapper = PytorchWrapper(identifier, model, preprocessing, process_output=process_activation, 
                                inferencer_cls=CausalInferencer, fps=DEFAULT_FPS, batch_padding=True,
                                layer_activation_format=layer_activation_format)

        # switch to video forward
        wrapper.forward = MethodType(_video_forward, wrapper)

    return ModelCommitment(identifier, wrapper, layers=layers)

def _image_model_video_preprocessing(video, image_size, **kwargs):
    images = video.to_pil_imgs()
    ret = torch.Tensor(preprocess_images(images, image_size, **kwargs))
    return ret

class PixelsActivationWrapper(ActivationWrapper):
    @staticmethod
    def _preprocess(video):
        return video.set_size((256, 256)).to_numpy() / 255.0

    def __init__(self):
        super().__init__("pixels", preprocessing=self._preprocess, inferencer_cls=CausalInferencer, 
                            fps=DEFAULT_FPS, layer_activation_format={"pixels": "THWC"})

    def get_activations(self, inputs, layers):
        assert len(layers) == 1 and layers[0] == "pixels"
        return OrderedDict([("pixels", np.array(inputs))])

def _tmpfile_image_model_video_preprocessing(video, custom_preprocess_images):
    import torch
    images = video.to_pil_imgs()

    import tempfile
    with tempfile.TemporaryDirectory() as tmpdirname:
        for i, img in enumerate(images):
            img.save(f"{tmpdirname}/img_{i}.png")
        image_paths = [f"{tmpdirname}/img_{i}.png" for i in range(len(images))]
        return torch.Tensor(custom_preprocess_images(image_paths))

class ImageFromArray(Image):
    def __init__(self, array, size):
        self._array = array
        self._size = size

    def copy(self):
        return ImageFromArray(self._array, self._size)

    def get_frame(self):
        return self._array

    def from_path(path):
        array = np.array(PILImage.open(path).convert('RGB'))
        return ImageFromArray(path, get_image_size(path))

def get_my_image_model_video_preprocessing(transform):
    def _my_image_model_video_preprocessing(video):
        import torch
        from torchvision import transforms
        images = video.to_numpy()
        image = images[0]
        isize = image.shape[:2][::-1]
        ret = [transform(ImageFromArray(image, isize)) for image in images]
        return torch.stack(ret)
    return _my_image_model_video_preprocessing

def _video_forward(self, inputs):
    # this function gets a list of preprocessed inputs and does the forward pass
    import torch
    tensor = torch.stack(inputs)
    B, T = tensor.shape[:2]
    tensor = tensor.view(B*T, *tensor.shape[2:])
    tensor = tensor.to(self._device)
    ret = self._model(tensor)