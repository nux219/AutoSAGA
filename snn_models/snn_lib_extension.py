import torch
import numpy as np
from snn_functions import *
from snn_lib import first_order_low_pass_layer, axon_layer


class neuron_layer_hardreset(torch.nn.Module):
    def __init__(self, input_size, neuron_number, step_num, batch_size, tau_m, **kargs):
        '''
        Same neuron model as STBP, adopted from https://github.com/yjwu17/BP-for-SpikingNN/blob/master/spiking_model.py
        :param input_size:
        :param neuron_number:
        :param step_num:
        :param batch_size:
        :param tau_m:
        :param train_bias:
        '''
        super().__init__()

        # default values
        self.options = {
            'train_decay_v': False,
            'train_bias': True,
            'return_state': True}

        self.options.update(kargs)

        self.input_size = input_size
        self.neuron_number = neuron_number
        self.step_num = step_num
        self.batch_size = batch_size
        self.return_state = self.options['return_state']

        self.weight = torch.nn.Linear(input_size, neuron_number, bias=self.options['train_bias'])
        self.tau_m = torch.full((neuron_number,), tau_m)

        self.decay_v = torch.exp(torch.tensor(-1 / tau_m))
        self.decay_v = torch.nn.Parameter(torch.full((self.neuron_number,), self.decay_v),
                                          requires_grad=self.options['train_decay_v'])

        self.enable_threshold = True

    def forward(self, input_spikes, states=None):
        """
        :param input_spikes: [batch, dim0 ,dim1..]
        :param  states: tuple (init_v, init_reset_v)
        :return:
        """

        if states is None:
            current_v, spike = self.create_init_states()

        # unbind along last dimension
        inputs = input_spikes.unbind(dim=-1)
        spikes = []
        for i in range(len(inputs)):

            weighted_psp = self.weight(inputs[i])

            current_v = (1 - spike) * current_v * self.decay_v + weighted_psp

            if self.enable_threshold:
                threshold_function = ActFun.apply
                spike = threshold_function(current_v)
            else:
                spike = current_v.clamp(0.0, 1.0)

            spikes += [spike]

        new_states = current_v

        if self.return_state:
            return torch.stack(spikes, dim=-1), new_states
        else:
            return torch.stack(spikes, dim=-1)

    def create_init_states(self):

        device = self.decay_v.device
        init_v = torch.zeros(self.neuron_number).to(device)
        init_spike = torch.zeros(self.neuron_number).to(device)

        return init_v, init_spike

    def named_parameters(self, prefix='', recurse=True):
        '''
        only return weight in neuron cell
        https://pytorch.org/docs/stable/_modules/torch/nn/modules/module.html#Module.named_parameters
        :return:
        '''
        for name, param in self.neuron_cell.weight.named_parameters(recurse=recurse):
            yield name, param


class conv2d_layer_hardreset(torch.nn.Module):
    def __init__(self, h_input, w_input, in_channels, out_channels, kernel_size, stride, padding, dilation, step_num,
                 batch_size,
                 tau_m, **kargs):
        '''
        Same neuron model as STBP, adopted from https://github.com/yjwu17/BP-for-SpikingNN/blob/master/spiking_model.py
        drop in replacement of conv2d_layer. A lot arguments actually are unused.
        :param input_size:
        :param neuron_number:
        :param step_num:
        :param batch_size:
        :param tau_m:
        :param train_decay_v:
        :param train_bias:
        '''
        super().__init__()

        self.options = {
            'train_decay_v': False,
            'train_bias': True,
            'return_state': True
        }

        self.options.update(kargs)

        self.step_num = step_num
        self.batch_size = batch_size
        self.train_bias = self.options['train_bias']
        self.return_state = self.options['return_state']

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.padding = padding
        self.kernel_size = kernel_size
        self.h_input = h_input
        self.w_input = w_input
        self.return_state = self.options['return_state']

        conv_out_h, conv_out_w = calculate_conv2d_outsize(h_input, w_input, padding, kernel_size, stride)
        self.output_shape = (out_channels, conv_out_h, conv_out_w)

        self.decay_v = torch.exp(torch.tensor(-1 / tau_m))
        self.decay_v = torch.nn.Parameter(torch.full(self.output_shape, self.decay_v))
        self.decay_v.requires_grad = self.options['train_decay_v']

        self.enable_threshold = True

        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                                    bias=self.train_bias)

    def forward(self, input_spikes, states=None):
        """
        :param input_spikes: [batch, dim0 ,dim1..]
        :param  states: tuple (init_v, init_reset_v)
        :return:
        """

        if states is None:
            current_v, spike = self.create_init_states()

        # unbind along last dimension
        inputs = input_spikes.unbind(dim=-1)
        spikes = []
        for i in range(len(inputs)):

            weighted_psp = self.conv(inputs[i])

            current_v = (1 - spike) * current_v * self.decay_v + weighted_psp

            if self.enable_threshold:
                threshold_function = ActFun.apply
                spike = threshold_function(current_v)
            else:
                spike = current_v.clamp(0.0, 1.0)

            spikes += [spike]

        new_states = current_v

        if self.return_state:
            return torch.stack(spikes, dim=-1), new_states
        else:
            return torch.stack(spikes, dim=-1)

    def create_init_states(self):

        device = self.conv.weight.device

        init_spike = torch.zeros(self.output_shape).to(device)
        init_v = torch.zeros(self.output_shape).to(device)

        return init_v, init_spike

    def named_parameters(self, prefix='', recurse=True):
        '''
        only return weight in neuron cell
        https://pytorch.org/docs/stable/_modules/torch/nn/modules/module.html#Module.named_parameters
        :return:
        '''
        for name, param in self.neuron_cell.weight.named_parameters(recurse=recurse):
            yield name, param


class neuron_layer_fixedpoint(torch.nn.Module):
    def __init__(self, input_size, neuron_number, step_num, batch_size, weight, bias, tau, **kargs):
        '''
        fixed point implementation, to mimic loihi

        v[t] = v[t-1](2^12 - compartmentVoltageDecay)/2^12 + compartmentCurrent[t] + biasMant*2^biasExp
        compartmentCurrent[t] = 2^(6+weightExp)\sum_i (spike_i * actWeight_i)
        actWeight_i = (weight_i >> numLsbBits) << numLsbBits

        :param input_size: int
        :param step_num:
        :param batch_size:
        :param weight: tensor [input size, neuron number]
        :param tau:
        :param compartmentVoltageDecay: compartmentVoltageDecay = 1/tau * 2^12
        :param reset_mode: 'soft'. 'hard', 'srm'
        :param biasMant: 0
        :param biasExp:  0
        :param vThMant: 0
        :param vTh: 0
        :param weightExp: 0
        :param numWeightBits: 0
        '''
        super().__init__()

        # default values
        self.options = {
            'reset_tau': None,
            'neuron_mode': 'soft',
            'biasMant': 0,
            'biasExp': 0,
            'vThMant': 100,
            'vTh': 1.0,
            'weightExp': 0,
            'numWeightBits': 8,
            'delay': 1,
            'refractory': 1
        }

        self.options.update(kargs)

        self.input_size = input_size
        self.neuron_number = neuron_number
        self.step_num = step_num
        self.batch_size = batch_size
        self.reset_tau = self.options['reset_tau']
        self.neuron_mode = self.options['neuron_mode']
        self.tau = tau
        vTh = self.options['vTh']
        self.delay = self.options['delay']
        self.vMaxExp = 23
        self.vMinExp = 23

        self.vMax = 2 ** self.vMaxExp - 1
        self.vMin = -2 ** self.vMinExp + 1

        self.scale_factor = 255.0 / weight.abs().max()

        scaled_weight = torch.tensor(weight * self.scale_factor, dtype=torch.int)
        numLsbBits = 8 - (self.options['numWeightBits'] - 1)
        scaled_weight = scaled_weight >> numLsbBits
        scaled_weight = scaled_weight << numLsbBits
        self.weight = torch.nn.Parameter(scaled_weight * 64, requires_grad=False)

        if bias is not None:
            scaled_bias = torch.tensor(bias * self.scale_factor, dtype=torch.int)
            self.bias = torch.nn.Parameter(scaled_bias * 64, requires_grad=False)
        else:
            self.bias = torch.nn.Parameter(torch.zeros((neuron_number,), dtype=torch.int), requires_grad=False)

        self.refractory = self.options['refractory']

        scaled_vTh = torch.tensor(vTh * self.scale_factor, dtype=torch.int, requires_grad=False)
        self.vTh = torch.nn.Parameter(scaled_vTh * 64, requires_grad=False)

        compartmentVoltageDecay = torch.tensor(4096 - np.exp(-1 / tau) * 4096, dtype=torch.int)
        self.compartmentVoltageDecay = torch.nn.Parameter(compartmentVoltageDecay, requires_grad=False)

        if self.neuron_mode == 'srm':
            resetDecay = torch.tensor(4096 - np.exp(-1 / self.reset_tau) * 4096, dtype=torch.int)
            self.resetDecay = torch.nn.Parameter(resetDecay, requires_grad=False)

    def forward(self, input_spikes):
        """
        :param input_spikes: [batch, input_size]
        :return: spike
        """

        prev_v = torch.zeros((self.batch_size, self.neuron_number), device=self.weight.device, dtype=torch.int)
        prev_reset_v = torch.zeros((self.batch_size, self.neuron_number), device=self.weight.device, dtype=torch.int)

        ref_t = torch.zeros((self.batch_size, self.neuron_number), device=self.weight.device, dtype=torch.int)

        inputs = input_spikes.unbind(dim=-1)
        spikes = []
        for i in range(len(inputs)):

            weighted_input = torch.matmul(inputs[i].int(), self.weight.T)

            if self.neuron_mode == 'hard':

                current_v = prev_v * (4096 - self.compartmentVoltageDecay) / 4096 + weighted_input + self.bias

                current_v[current_v > self.vMax] = self.vMax
                current_v[current_v < self.vMin] = self.vMin

                spike = current_v.clone()
                spike[spike < self.vTh] = 0
                spike[spike >= self.vTh] = 1
                current_v[current_v > self.vTh] = 0

            elif self.neuron_mode == 'soft':
                current_v = prev_v * (4096 - self.compartmentVoltageDecay) / 4096 + weighted_input + self.bias

                current_v[current_v > self.vMax] = self.vMax
                current_v[current_v < self.vMin] = self.vMin

                spike = current_v.clone()
                spike[spike < self.vTh] = 0
                spike[spike >= self.vTh] = 1
                current_v[current_v > self.vTh] = current_v[current_v > self.vTh] - self.vTh
            elif self.neuron_mode == 'srm':

                current_v = prev_v * (
                            4096 - self.compartmentVoltageDecay) / 4096 + weighted_input + self.bias - prev_reset_v

                current_v[current_v > self.vMax] = self.vMax
                current_v[current_v < self.vMin] = self.vMin

                spike = current_v.clone()
                spike[spike < self.vTh] = 0
                spike[spike >= self.vTh] = 1

                current_reset = prev_reset_v * (4096 - self.resetDecay) / 4096 + spike * self.vTh
                prev_reset_v = current_reset
            elif self.neuron_mode == 'nengo_loihi':

                # https://www.nengo.ai/nengo-loihi/_modules/nengo_loihi/neurons.html#LoihiLIF

                ref_t -= 1
                ref_t[ref_t < 0] = 0
                # delta_t = (1 - ref_t).clamp(0, 1)

                # 1 indicates in ref period, neuron should not accumulate input or decay
                # 0 means neuron should integrate input and decay
                ref_mask = ref_t.clamp(0, 1)

                current_v = (weighted_input + self.bias) * (1 - ref_mask) + (1 - ref_mask) * (
                            4096 - self.compartmentVoltageDecay) / 4096 * prev_v + ref_mask * prev_v

                spike_mask = current_v > 1

                spike = current_v.clone()
                spike[spike <= self.vTh] = 0
                spike[spike > self.vTh] = 1

                ref_t[spike_mask] = ref_t[spike_mask] + 1 + self.refractory

                current_v[spike_mask] = 0

            spikes.append(spike)

            prev_v = current_v

        if (self.delay == 1):
            spikes.insert(0, torch.zeros((self.batch_size, self.neuron_number), dtype=torch.int))
            spikes = spikes[0:-2]

        return torch.stack(spikes, dim=-1)


class readout_layer(torch.nn.Module):
    def __init__(self, input_size, neuron_number, step_num, batch_size, **kargs):
        '''
        ref: https://arxiv.org/pdf/2008.03658.pdf
        :param input_size:
        :param neuron_number:
        :param step_num:
        :param batch_size:
        :param train_bias:
        '''
        super().__init__()

        # default values
        self.options = {
            'train_bias': True,
            'return_state': True,
            'synapse_type': None,
            'bntt': True}

        self.options.update(kargs)

        self.input_size = input_size
        self.neuron_number = neuron_number
        self.step_num = step_num
        self.batch_size = batch_size

        self.weight = torch.nn.Linear(input_size, neuron_number, bias=self.options['train_bias'])

        if self.options['bntt']:
            self.bntt = torch.nn.ModuleList(
                [torch.nn.BatchNorm1d(neuron_number, eps=1e-4, momentum=0.1, affine=True) for i in
                 range(self.step_num)])

            for bntt_layer in self.bntt:
                bntt_layer.bias = None

    def forward(self, input_spikes, states=None):
        """
        :param input_spikes: [batch, dim0 ,dim1..]
        :param states: init_v
        :return:
        """

        if states is None:
            current_v = self.create_init_states()

        # unbind along last dimension
        inputs = input_spikes.unbind(dim=-1)
        outputs = []
        for i in range(len(inputs)):

            weighted_psp = self.weight(inputs[i])

            if self.options['bntt']:
                current_v = current_v + self.bntt[i](weighted_psp)
            else:
                current_v += weighted_psp

            outputs += [current_v]

        new_states = current_v

        if self.options['return_state']:
            return torch.stack(outputs, dim=-1), new_states
        else:
            return torch.stack(outputs, dim=-1)

    def create_init_states(self):

        device = self.weight.weight.device
        init_v = torch.zeros(self.neuron_number).to(device)

        return init_v


class neuron_layer_bntt_v2(torch.nn.Module):
    def __init__(self, input_size, neuron_number, step_num, batch_size, tau_m, **kargs):
        '''
        batch norm through time, ref； https://github.com/Intelligent-Computing-Lab-Yale/BNTT-Batch-Normalization-Through-Time/blob/main/model.py
        :param input_size:
        :param neuron_number:
        :param step_num:
        :param batch_size:
        :param tau_m:
        :param train_decay_v:
        :param train_reset_decay:
        :param train_reset_v:
        :param train_threshold:
        :param train_bias:
        :param membrane_filter:
        '''
        super().__init__()

        # default values
        self.options = {
            'train_decay_v': False,
            'train_reset_decay': False,
            'train_reset_v': False,
            'train_threshold': False,
            'train_bias': True,
            'membrane_filter': False,
            'reset_v': 1.0,
            # 'input_type'            : 'axon'
            'return_state': True,
            'synapse_type': None,
            'threshold_func': 'erfc'
        }

        self.options.update(kargs)

        self.input_size = input_size
        self.neuron_number = neuron_number
        self.step_num = step_num
        self.batch_size = batch_size
        self.return_state = self.options['return_state']
        self.enable_threshold = True

        self.synapse_type = self.options['synapse_type']
        self.synapse_filter = None

        if self.synapse_type == 'none':
            pass
        elif self.synapse_type == 'first_order_low_pass':
            self.synapse_filter = first_order_low_pass_layer((input_size,), step_num, batch_size,
                                                             kargs['synapse_tau_s'],
                                                             kargs['train_synapse_tau'])
        elif self.synapse_type == 'dual_exp':
            self.synapse_filter = axon_layer((input_size,), step_num, self.batch_size,
                                             kargs['synapse_tau_m'],
                                             kargs['synapse_tau_s'],
                                             kargs['train_synapse_tau'],
                                             kargs['train_synapse_tau'])
        else:
            raise Exception("unrecognized synapse filter type")

        self.weight = torch.nn.Linear(input_size, neuron_number, bias=self.options['train_bias'])
        self.tau_m = torch.full((neuron_number,), tau_m)

        self.sigma = torch.nn.Parameter(torch.tensor(0.4))
        self.sigma.requires_grad = False

        self.reset_decay = torch.exp(torch.tensor(-1.0 / tau_m))
        self.reset_decay = torch.nn.Parameter(torch.full((self.neuron_number,), self.reset_decay))
        self.reset_decay.requires_grad = self.options['train_reset_decay']

        self.reset_v = torch.nn.Parameter(torch.full((self.neuron_number,), self.options['reset_v']))
        self.reset_v.requires_grad = self.options['train_reset_v']

        self.decay_v = torch.exp(torch.tensor(-1 / tau_m))
        self.decay_v = torch.nn.Parameter(torch.full((self.neuron_number,), self.decay_v))
        self.decay_v.requires_grad = self.options['train_decay_v']

        self.threshold_offset = torch.nn.Parameter(
            torch.distributions.normal.Normal(torch.tensor([0.0]), torch.tensor([0.1])).sample([neuron_number]).reshape(
                -1))

        self.threshold_offset.requires_grad = self.options['train_threshold']
        self.train_threshold = self.options['train_threshold']

        self.enable_threshold = True
        self.membrane_filter = self.options['membrane_filter']

        self.bntt = torch.nn.ModuleList(
            [torch.nn.BatchNorm1d(neuron_number, eps=1e-4, momentum=0.1, affine=True) for i in range(self.step_num)])

        for bntt_layer in self.bntt:
            bntt_layer.bias = None

        self.threshold_func = erfc.apply

    def forward(self, input_spikes, states=None):
        """
        :param input_spikes: [batch, dim0 ,dim1..]
        :param  states: tuple (init_v, init_reset_v)
        :return:
        """

        if self.synapse_filter is not None:
            x, _ = self.synapse_filter(input_spikes)
        else:
            x = input_spikes

        if states is None:
            prev_v, prev_reset = self.create_init_states()

        # unbind along last dimension
        inputs = x.unbind(dim=-1)
        spikes = []

        for i in range(len(inputs)):

            weighted_psp = self.weight(inputs[i])

            if self.membrane_filter:
                current_v = prev_v * self.decay_v + self.bntt[i](weighted_psp) - prev_reset
            else:
                current_v = weighted_psp - prev_reset

            if self.train_threshold:
                current_v = current_v + self.threshold_offset

            if self.enable_threshold:
                spike = self.threshold_func(current_v)
            else:
                spike = current_v.clamp(0.0, 1.0)

            current_reset = prev_reset * self.reset_decay + spike * self.reset_v

            if self.train_threshold:
                current_v = current_v - self.threshold_offset

            spikes += [spike]

            prev_v = current_v
            prev_reset = current_reset

        if self.return_state:
            return torch.stack(spikes, dim=-1), (current_v, current_reset)
        else:
            return torch.stack(spikes, dim=-1)

    def create_init_states(self):

        device = self.reset_decay.device
        init_v = torch.zeros(self.neuron_number).to(device)
        init_reset_v = torch.zeros(self.neuron_number).to(device)

        init_states = (init_v, init_reset_v)

        return init_states

    def change_threshold_func(self, threshold_func):
        self.threshold_func = threshold_func.apply


class conv2d_layer_bntt_v2(torch.nn.Module):
    def __init__(self, h_input, w_input, in_channels, out_channels, kernel_size, stride, padding, dilation, step_num,
                 batch_size,
                 tau_m, **kwargs):
        '''
        :param input_size:
        :param neuron_number:
        :param step_num:
        :param batch_size:
        :param tau_m:
        :param train_decay_v:
        :param train_reset_decay:
        :param train_reset_v:
        :param train_threshold:
        :param train_bias:
        :param membrane_filter:
        '''
        super().__init__()

        # default values
        self.options = {
            'train_decay_v': False,
            'train_reset_decay': False,
            'train_reset_v': False,
            'train_threshold': False,
            'train_bias': True,
            'membrane_filter': False,
            'reset_v': 1.0,
            'input_type': 'axon',
            'return_state': True,
            'synapse_type': None}

        self.options.update(kwargs)

        self.step_num = step_num
        self.batch_size = batch_size

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.padding = padding
        self.kernel_size = kernel_size
        self.h_input = h_input
        self.w_input = w_input
        self.return_state = self.options['return_state']

        self.enable_threshold = True
        self.membrane_filter = self.options['membrane_filter']

        self.synapse_type = self.options['synapse_type']
        self.synapse_filter = None

        conv_out_h, conv_out_w = calculate_conv2d_outsize(h_input, w_input, padding, kernel_size, stride)
        self.output_shape = (out_channels, conv_out_h, conv_out_w)

        if self.synapse_type == 'none':
            pass
        elif self.synapse_type == 'first_order_low_pass':
            self.synapse_filter = first_order_low_pass_layer((in_channels, h_input, w_input), step_num, batch_size,
                                                             kwargs['synapse_tau_s'],
                                                             kwargs['train_synapse_tau'])
        elif self.synapse_type == 'dual_exp':
            self.synapse_filter = axon_layer((in_channels, h_input, w_input), step_num, self.batch_size,
                                             kwargs['synapse_tau_m'],
                                             kwargs['synapse_tau_s'],
                                             kwargs['train_synapse_tau'],
                                             kwargs['train_synapse_tau'])
        else:
            raise Exception("unrecognized synapse filter type")

        self.train_threshold = self.options['train_threshold']

        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                                    bias=self.options['train_bias'])

        conv_out_h, conv_out_w = calculate_conv2d_outsize(h_input, w_input, padding, kernel_size, stride)

        # output shape will be (batch, out_channels, conv_out_h, conv_out_w)
        # this is also the shape of neurons and time constants and reset_v
        self.output_shape = (out_channels, conv_out_h, conv_out_w)

        self.sigma = torch.nn.Parameter(torch.tensor(0.4))
        self.sigma.requires_grad = False

        self.reset_decay = torch.exp(torch.tensor(-1.0 / tau_m))
        self.reset_decay = torch.nn.Parameter(torch.full(self.output_shape, self.reset_decay))
        self.reset_decay.requires_grad = self.options['train_reset_decay']

        self.reset_v = torch.nn.Parameter(torch.full(self.output_shape, 1.0))
        self.reset_v.requires_grad = self.options['train_reset_v']

        self.decay_v = torch.exp(torch.tensor(-1 / tau_m))
        self.decay_v = torch.nn.Parameter(torch.full(self.output_shape, self.decay_v))
        self.decay_v.requires_grad = self.options['train_decay_v']

        self.threshold_offset = torch.nn.Parameter(
            torch.distributions.normal.Normal(torch.tensor([0.0]), torch.tensor([0.1])).sample(
                self.output_shape).squeeze())

        self.threshold_offset.requires_grad = self.options['train_threshold']

        self.bntt = torch.nn.ModuleList(
            [torch.nn.BatchNorm2d(self.out_channels, eps=1e-4, momentum=0.1, affine=True) for i in
             range(self.step_num)])

        for bntt_layer in self.bntt:
            bntt_layer.bias = None

        self.threshold_func = erfc.apply

    def forward(self, input_spikes, states=None):
        """
        :param input_spikes: [batch, dim0 ,dim1..,t]
        :param  prev_states: tuple (prev_psp_m, prev_psp_s)
        :return:
        """

        if self.synapse_filter is not None:
            x, _ = self.synapse_filter(input_spikes)
        else:
            x = input_spikes

        if states is None:
            prev_v, prev_reset = self.create_init_states()

        # unbind along last dimension
        inputs = x.unbind(dim=-1)
        spikes = []
        for i in range(len(inputs)):

            conv2d_out = self.conv(inputs[i])

            if self.membrane_filter:
                current_v = prev_v * self.decay_v + self.bntt[i](conv2d_out) - prev_reset
            else:
                current_v = self.bntt[i](conv2d_out) - prev_reset

            if self.train_threshold:
                current_v = current_v + self.threshold_offset

            if self.enable_threshold:
                spike = self.threshold_func(current_v)
            else:
                spike = current_v.clamp(0.0, 1.0)
                # print('spike', spike)

            current_reset = prev_reset * self.reset_decay + spike * self.reset_v

            if self.train_threshold:
                current_v = current_v - self.threshold_offset

            prev_v = current_v
            prev_reset = current_reset

            spikes += [spike]

        if self.return_state:
            return torch.stack(spikes, dim=-1), (current_v, current_reset)
        else:
            return torch.stack(spikes, dim=-1)

    def create_init_states(self):

        device = self.reset_decay.device

        init_v = torch.zeros(self.output_shape).to(device)
        init_reset_v = torch.zeros(self.output_shape).to(device)

        init_states = (init_v, init_reset_v)

        return init_states

    def change_threshold_func(self, threshold_func):
        self.threshold_func = threshold_func.apply
