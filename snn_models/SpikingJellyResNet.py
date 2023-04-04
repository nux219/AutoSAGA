# ref https://spikingjelly.readthedocs.io/zh_CN/0.0.0.0.12/clock_driven/16_train_large_scale_snn.html

import torch

from spikingjelly.clock_driven.model import spiking_resnet, sew_resnet
from spikingjelly.clock_driven import neuron, surrogate, functional


#Load the CIFAR-10 Spiking Jelly model 
def LoadCIFAR10SNNResNetBackProp(modelDir):
    #Model parameters
    sg = 'ATan'
    argsNeuron = 'MultiStepParametricLIFNode'
    arch = 'sew_resnet18'
    num_classes = 10
    timeStep = 4
    surrogate_gradient = {
                        'ATan' : surrogate.ATan(),
                        'Sigmoid' : surrogate.Sigmoid(),
                        'PiecewiseLeakyReLU': surrogate.PiecewiseLeakyReLU(),
                        'S2NN': surrogate.S2NN(),
                        'QPseudoSpike': surrogate.QPseudoSpike()
                    }
    sg_type = surrogate_gradient[sg]
    neuron_dict = {
        'MultiStepIFNode'               : neuron.MultiStepIFNode,
        'MultiStepParametricLIFNode'    : neuron.MultiStepParametricLIFNode,
        'MultiStepEIFNode'              : neuron.MultiStepEIFNode,
        'MultiStepLIFNode'              : neuron.MultiStepLIFNode,
    }
    neuron_type = neuron_dict[argsNeuron]
    model_arch_dict = {
                    'sew_resnet18'       : sew_resnet.multi_step_sew_resnet18, 
                    'sew_resnet34'       : sew_resnet.multi_step_sew_resnet34, 
                    'sew_resnet50'       : sew_resnet.multi_step_sew_resnet50,
                    'spiking_resnet18'   : spiking_resnet.multi_step_spiking_resnet18, 
                    'spiking_resnet34'   : spiking_resnet.multi_step_spiking_resnet34, 
                    'spiking_resnet50'   : spiking_resnet.multi_step_spiking_resnet50,
    }
    model_type = model_arch_dict[arch]
    model = model_type(T=timeStep, num_classes=num_classes, cnf='ADD', multi_step_neuron=neuron_type, surrogate_function=sg_type)
    #Load the model from the baseDir
    checkpoint = torch.load(modelDir)
    model.load_state_dict(checkpoint["snn_state_dict"], strict=True)
    return model


#Load the CIFAR-10 Spiking Jelly model
def LoadCIFAR100SNNResNetBackProp(modelDir):
    #Model parameters
    sg = 'ATan'
    argsNeuron = 'MultiStepParametricLIFNode'
    arch = 'sew_resnet18'
    num_classes = 100
    timeStep = 5
    surrogate_gradient = {
                        'ATan' : surrogate.ATan(),
                        'Sigmoid' : surrogate.Sigmoid(),
                        'PiecewiseLeakyReLU': surrogate.PiecewiseLeakyReLU(),
                        'S2NN': surrogate.S2NN(),
                        'QPseudoSpike': surrogate.QPseudoSpike()
                    }
    sg_type = surrogate_gradient[sg]
    neuron_dict = {
        'MultiStepIFNode'               : neuron.MultiStepIFNode,
        'MultiStepParametricLIFNode'    : neuron.MultiStepParametricLIFNode,
        'MultiStepEIFNode'              : neuron.MultiStepEIFNode,
        'MultiStepLIFNode'              : neuron.MultiStepLIFNode,
    }
    neuron_type = neuron_dict[argsNeuron]
    model_arch_dict = {
                    'sew_resnet18'       : sew_resnet.multi_step_sew_resnet18,
                    'sew_resnet34'       : sew_resnet.multi_step_sew_resnet34,
                    'sew_resnet50'       : sew_resnet.multi_step_sew_resnet50,
                    'spiking_resnet18'   : spiking_resnet.multi_step_spiking_resnet18,
                    'spiking_resnet34'   : spiking_resnet.multi_step_spiking_resnet34,
                    'spiking_resnet50'   : spiking_resnet.multi_step_spiking_resnet50,
    }
    model_type = model_arch_dict[arch]
    model = model_type(T=timeStep, num_classes=num_classes, cnf='ADD', multi_step_neuron=neuron_type, surrogate_function=sg_type)
    #Load the model from the baseDir
    checkpoint = torch.load(modelDir)
    model.load_state_dict(checkpoint["snn_state_dict"], strict=True)
    return model
