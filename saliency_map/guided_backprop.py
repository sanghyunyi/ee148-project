"""
Created on Thu Oct 26 11:23:47 2017

@author: Utku Ozbulak - github.com/utkuozbulak
"""
import torch
from torch.nn import ReLU
from model import *
import glob, os, json
from pathlib import Path

from misc_functions import (get_example_params,
                            convert_to_grayscale,
                            save_gradient_images,
                            get_positive_negative_saliency)


class GuidedBackprop():
    """
       Produces gradients generated with guided back propagation from the given image
    """
    def __init__(self, model):
        self.model = model
        self.gradients = None
        self.forward_relu_outputs = []
        # Put model in evaluation mode
        self.model.eval()
        self.update_relus()
        self.hook_layers()

    def hook_layers(self):
        def hook_function(module, grad_in, grad_out):
            if self.gradients == None:
                self.gradients = grad_in[0]
            else:
                self.gradients += grad_in[0]

        # Register hook to the first layer
        first_layer = list(self.model.features._modules.items())[0][1]
        first_layer.register_backward_hook(hook_function)
        first_layer2 = list(self.model.features2._modules.items())[0][1]
        first_layer2.register_backward_hook(hook_function)

    def update_relus(self):
        """
            Updates relu activation functions so that
                1- stores output in forward pass
                2- imputes zero for gradient values that are less than zero
        """
        def relu_backward_hook_function(module, grad_in, grad_out):
            """
            If there is a negative gradient, change it to zero
            """
            # Get last forward output
            corresponding_forward_output = self.forward_relu_outputs[-1]
            corresponding_forward_output[corresponding_forward_output > 0] = 1
            modified_grad_out = corresponding_forward_output * torch.clamp(grad_in[0], min=0.0)
            del self.forward_relu_outputs[-1]  # Remove last forward output
            return (modified_grad_out,)

        def relu_forward_hook_function(module, ten_in, ten_out):
            """
            Store results of forward pass
            """
            self.forward_relu_outputs.append(ten_out)

        # Loop through layers, hook up ReLUs
        for pos, module in self.model.features._modules.items():
            if isinstance(module, ReLU):
                module.register_backward_hook(relu_backward_hook_function)
                module.register_forward_hook(relu_forward_hook_function)
        for pos, module in self.model.features2._modules.items():
            if isinstance(module, ReLU):
                module.register_backward_hook(relu_backward_hook_function)
                module.register_forward_hook(relu_forward_hook_function)
                
    def generate_gradients(self, input_image, target_class):
        # Forward pass
        model_output = self.model(input_image)
        scores = model_output
        scores = [score.cpu().data.numpy().tolist()[0][0] for score in scores][:4]
        # Zero gradients
        self.model.zero_grad()
        # Target for backprop
        target = torch.cuda.FloatTensor([[100]])
        model_output = model_output[target_class]
        model_output.backward(gradient=target)
        # Convert Pytorch variable to numpy array
        # [0] to get rid of the first channel (1,3,224,224)
        gradients_as_arr = self.gradients.cpu().data.numpy()[0]
        return scores, gradients_as_arr


if __name__ == '__main__':
    action_names = ["pinch", "clench", "poke", "palm"]
    #IMAGE_DIR = "../../part-affordance-dataset/tools/"
    IMAGE_DIR = "../../IIT_AFF_processed/rgb/"
    file_names = [str(path) for path in Path(IMAGE_DIR).rglob('*.jpg')]
    # Get params
    pred_json = {}
    for i, img_path in enumerate(file_names):
        print(i, img_path)
        for target_idx, action_name in enumerate(action_names):
            (original_image, prep_img, target_class, file_name_to_export, pretrained_model) =\
                get_example_params((img_path, target_idx))

            # Guided backprop
            GBP = GuidedBackprop(pretrained_model)
            # Get gradients
            scores, guided_grads = GBP.generate_gradients(prep_img, target_class)
            if target_idx == 0:
                pinch = scores[0]
                clench = scores[1]
                poke = scores[2]
                palm = scores[3]
                pred_json[os.path.basename(img_path)] = {
                    "pinch": pinch,
                    "clench": clench,
                    "poke": poke,
                    "palm": palm
                }

            # Save colored gradients
            save_gradient_images(guided_grads, file_name_to_export + '_Guided_BP_color_' + action_name)
            # Convert to grayscale
            grayscale_guided_grads = convert_to_grayscale(guided_grads)
            # Save grayscale gradients
            save_gradient_images(grayscale_guided_grads, file_name_to_export + '_Guided_BP_gray_' + action_name)
            # Positive and negative saliency maps
            pos_sal, neg_sal = get_positive_negative_saliency(guided_grads)
            save_gradient_images(pos_sal, file_name_to_export + '_pos_sal_' + action_name)
            save_gradient_images(neg_sal, file_name_to_export + '_neg_sal_' + action_name)

    json.dump(pred_json, open("../results/predictions.json", "w"))
    print('Guided backprop completed')
