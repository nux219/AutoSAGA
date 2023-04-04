

from h5py._hl.filters import _normalize_external
from numpy.core.fromnumeric import repeat
from torch.utils.data import Dataset, DataLoader

import numpy as np
from scipy.io import loadmat
from torchvision import transforms, utils
import torch
import os
# from quantities import ms
from ebdataset.vision import IBMGesture, H5IBMGesture
from ebdataset.vision.transforms import ToDense
from ebdataset.vision import NMnist


def get_rand_transform(transform_config):
    t1_size = transform_config['RandomResizedCrop']['size']
    t1_scale = transform_config['RandomResizedCrop']['scale']
    t1_ratio = transform_config['RandomResizedCrop']['ratio']
    t1 = transforms.RandomResizedCrop(t1_size, scale=t1_scale, ratio=t1_ratio, interpolation=2)
    
    t2_angle = transform_config['RandomRotation']['angle']
    t2 = transforms.RandomRotation(t2_angle, resample=False, expand=False, center=None)
    t3 = transforms.Compose([t1, t2])

    rand_transform = transforms.RandomApply([t1, t2, t3], p=transform_config['RandomApply']['probability'])

    return rand_transform


class MNISTDataset_Poisson_Spike(Dataset):
    """mnist dataset

    torchvision_mnist: dataset object
    length: number of steps of snn
    max_rate: a scale factor. MNIST pixel value is normalized to [0,1], and them multiply with this value
    flatten: return 28x28 image or a flattened 1d vector
    transform: transform

    How to use:

    from torch.utils.data import Dataset, DataLoader

    #load mnist training dataset
    mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=rand_transform)

    #load mnist test dataset
    mnist_testset = datasets.MNIST(root='./data', train=False, download=True, transform=None)

    train_data = MNISTDataset(mnist_trainset, max_rate=1, length=100, flatten=True)
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)


    """

    def __init__(self, torchvision_mnist, length, max_rate=1, flatten=False, transform=None):
        self.dataset = torchvision_mnist
        self.transform = transform
        self.flatten = flatten
        self.length = length
        self.max_rate = max_rate

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img = self.dataset[idx][0]
        if self.transform:
            img = self.transform(img)

        #shape of image [h,w]
        img = np.array(self.dataset[idx][0], dtype=np.float32) / 255.0 * self.max_rate
        shape = img.shape

        #flatten image
        img = img.reshape(-1)

        # shape of spike_trains [h*w, length]
        spike_trains = np.zeros((len(img), self.length), dtype=np.float32)

        #extend last dimension for time, repeat image along the last dimension
        img_tile = np.expand_dims(img,1)
        img_tile = np.tile(img_tile, (1,self.length))
        rand = np.random.uniform(0,1,(len(img), self.length))
        spike_trains[np.where(img_tile > rand)] = 1

        if self.flatten == False:
            spike_trains = spike_trains.reshape([shape[0], shape[1], self.length])

        return spike_trains, self.dataset[idx][1]


class MNISTDataset_Even_Spaced_Spike(Dataset):
    """mnist dataset

    """

    def __init__(self, torchvision_mnist, length, max_rate=1, flatten=False):
        self.dataset = torchvision_mnist
        self.flatten = flatten
        self.length = length
        self.max_rate = max_rate

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if torch.is_tensor(idx):
            idx = idx.tolist()

        img = self.dataset[idx][0]

        #shape of image [h,w]
        img = np.array(self.dataset[idx][0], dtype=np.float32) / 255.0 * self.max_rate
        
        #flatten image
        if self.flatten == True:
            img = img.reshape(-1)
            shape = img.shape[0]
            spike_trains = np.zeros([shape, self.length])

            nums = img * self.length

            for idx, row in enumerate(spike_trains):
                non_zero_idx = np.linspace(0, self.length, int(nums[idx]), dtype=np.int, endpoint=False)
                spike_trains[idx, non_zero_idx] = 1
        else:
            raise Exception('error')

        return spike_trains, self.dataset[idx][1]

class MNISTDataset(Dataset):
    """mnist dataset

    torchvision_mnist: dataset object
    length: number of steps of snn
    max_rate: a scale factor. MNIST pixel value is normalized to [0,1], and them multiply with this value
    flatten: return 28x28 image or a flattened 1d vector
    transform: transform
    
    How to use:

    from torch.utils.data import Dataset, DataLoader

    #load mnist training dataset
    mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=rand_transform)

    #load mnist test dataset
    mnist_testset = datasets.MNIST(root='./data', train=False, download=True, transform=None)

    train_data = MNISTDataset(mnist_trainset, max_rate=1, length=100, flatten=True)
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)    

    
    """

    def __init__(self, torchvision_mnist, length, max_rate = 1, flatten = False, transform=None, padding=0):
        self.dataset = torchvision_mnist
        self.transform = transform
        self.flatten = flatten
        self.length = length
        self.max_rate = max_rate
        self.padding = padding

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        img = self.dataset[idx][0]
        if self.transform:
            img = self.transform(img)
        
        if self.padding != 0:
            ## pad 0 by default
            img = np.pad(img, ((self.padding, self.padding),(self.padding, self.padding)), mode='constant')

        img = np.array(img,dtype=np.float32)/255.0 * self.max_rate
        shape = img.shape
        img_spike = None
        if self.flatten == True:
            img = img.reshape(-1)

        return img, self.dataset[idx][1]

class N_MNISTDataset(Dataset):
    """
    data loader of neuromorphic mnist
    
    How to use:

    train_dataset_folder = home_folder + '/dataset/nmnist/frame11/train'
    train_label_path = home_folder + '/dataset/nmnist/frame11/train_label.mat'
    test_dataset_folder = home_folder + '/dataset/nmnist/frame11/test'
    test_label_path = home_folder + '/dataset/nmnist/frame11/test_label.mat'


    train_data = MNISTDataset(train_dataset_folder, train_label_path, max_rate=max_rate, length=length,
                          rand_padding_probability=rand_padding_probability,
                          rand_time_padding_probability=rand_time_padding_probability, flatten=False, transform=False)

    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)


    """

    def __init__(self, dataset_path, label_path, length, rand_padding_probability, rand_time_padding_probability, max_rate=1, flatten=False,
                 transform=None):
        '''
        dataset_path: folder contains .mat file of nmnist dataset
        label_path: .mat label file. stores the file name and corresponding class id
        length: number of steps of the snn
        rand_padding_probability: the max size is 34x34. some samples are smaller than 34x34. rand pad means extend the size to 34x34.
            but the original data maynot be placed in the center. e.g. a sample size is 28x28 and extended to 34x34. left margin may be 2 
            and right margin is 4.
        rand_time_padding_probability: samples length are also not the same. e.g. the snn will run 20 steps, but a sample has 15 frames. 5 frames will be 
            inserted. the inserted frame comes from the original sample. they are selected in two ways. 1: evenly select 5 frames from the original sample,
            and insert. 2: randomly select 5 frames and insert.

            see steps_to_pad and padded_step to know how frames are selected and inserted.
        
        max_rate: pixel value multiply with this argument

        flatten: not used yet

        transform: not used yet

        '''
        self.dataset_path = dataset_path
        self.label_path = label_path

        self.transform = transform
        self.flatten = flatten
        self.length = length
        self.extended_steps = np.array(np.arange(length))

        self.max_rate = max_rate

        self.file_list = []

        self.transform = transform

        self.rand_padding_probability = rand_padding_probability
        self.rand_time_padding_probability = rand_time_padding_probability

        len_list = []

        # get a list of all file's path
        for file in os.listdir(dataset_path):
            if file.endswith(".mat"):
                mat_file_path = os.path.join(dataset_path, file)

                self.file_list.append(mat_file_path)

                _, _, _, l = loadmat(mat_file_path)['result'].shape

                len_list.append(l)

        # each file may have different number of frames, find the min and frame number
        self.min_len = min(len_list)
        self.max_len = max(len_list)

        # steps in extended image that are same as original
        self.selected_step = np.linspace(0, self.length, self.min_len, False, dtype=np.int)

        # steps in extended image to be padded
        self.padded_step = np.setdiff1d(self.extended_steps, self.selected_step)

        self.len_diff = self.length - self.min_len

        # step in original image that will be used to pad
        self.steps_to_pad = np.linspace(0, self.min_len, self.len_diff, False, dtype=np.int)

        label_list = loadmat(label_path)['label_list']

        self.label = {}
        for item in label_list:
            idx = item[0]
            label = item[1]
            self.label[idx] = label

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        mat_file_path = self.file_list[idx]

        # img shape [h,w,c,len]
        img = loadmat(mat_file_path)['result'].astype(np.float32)

        # select 0-1 channel from 0 to min_len
        # some file may be greater than 34x34, to prevent this, slice it
        img = img[0:34, 0:34, 0:2, :self.min_len]
        img = img * self.max_rate

        # every mat file has different size, so pad them to 34*34
        # random padding
        full_img = np.zeros([34, 34, 2, self.min_len], dtype=np.float32)
        rand_padding = True
        if self.rand_padding_probability > np.random.uniform():
            h_diff = 34 - img.shape[0]
            w_diff = 34 - img.shape[1]

            top_margin = np.random.randint(0, h_diff + 1)
            left_margin = np.random.randint(0, w_diff + 1)

            full_img[top_margin:img.shape[0] + top_margin, left_margin:img.shape[1] + left_margin, :, :, ] = img
        else:
            h_diff = 34 - img.shape[0]
            w_diff = 34 - img.shape[1]
            top_margin = h_diff // 2
            left_margin = w_diff // 2
            # center pad
            full_img[top_margin:img.shape[0] + top_margin, left_margin:img.shape[1] + left_margin, :, :, ] = img

        # if snn length is larger than frame number
        # in this case, we need to pad in time to extend data
        if self.length > self.min_len:
            extended_image = np.zeros([34, 34, 2, self.length], dtype=np.float32)

            extended_image[:, :, :, self.selected_step] = full_img

            if self.rand_time_padding_probability > np.random.uniform():
                rand_step = np.random.randint(0, self.min_len, size=self.len_diff)
                extended_image[:, :, :, self.padded_step] = full_img[:, :, :, rand_step]
            else:
                extended_image[:, :, :, self.padded_step] = full_img[:, :, :, self.steps_to_pad]

            full_img = extended_image

        mat_file_idx = int(mat_file_path.split('/')[-1].split('.')[0])

        label = self.label[mat_file_idx].astype(np.int)

        return full_img, label


class NMNIST_Dataset_Raw(Dataset):
    """
    data loader of neuromorphic mnist
    
    """

    def __init__(self, dataset_path, length, bin_size, is_train, rand_time_padding_probability, rand_shift_probability, scale=1):
        '''
        dataset_path: folder contains Train and Test folder
        length: number of frames
        rand_time_padding_probability: samples length are also not the same. e.g. the snn will run 20 steps, but a sample has 15 frames. 5 frames will be 
            inserted. the inserted frame comes from the original sample. they are selected in two ways. 1: evenly select 5 frames from the original sample,
            and insert. 2: randomly select 5 frames and insert.

            see steps_to_pad and padded_step to know how frames are selected and inserted.

        '''

        self.length = length
        self.extended_steps = np.array(np.arange(length))

        self.scale = scale
        self.bin_size = bin_size

        self.ToDense = ToDense(bin_size*ms)

        self.data = NMnist(dataset_path, is_train=is_train, transforms=self.ToDense)

        self.rand_time_padding_probability = rand_time_padding_probability

        self.rand_shift_probability = rand_shift_probability

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        img = self.data[idx][0].numpy()
        h,w,c,l = img.shape

        img = img * self.scale

        # if length is larger than frame number
        # in this case, we need to pad in time to extend data
        if self.length > l:

            # steps in extended image that are same as original
            selected_step = np.linspace(0, self.length, l, False, dtype=np.int)

            # steps in extended image to be padded
            padded_step = np.setdiff1d(self.extended_steps, selected_step)

            # step in original image that will be used to pad
            steps_to_pad = np.linspace(0, l, self.length - l, False, dtype=np.int)

            extended_image = np.zeros([34, 34, 2, self.length], dtype=np.float32)

            extended_image[:, :, :, selected_step] = img

            if self.rand_time_padding_probability > np.random.uniform():
                rand_step = np.random.randint(0, l, size= self.length - l)
                extended_image[:, :, :, padded_step] = img[:, :, :, rand_step]
            else:
                extended_image[:, :, :, padded_step] = img[:, :, :, steps_to_pad]

            img = extended_image
        
        elif self.length < l:

            len_diff = l - self.length

            start = np.random.randint(0, len_diff)

            img = img[:,:,:,start:start+self.length]
        
        if self.rand_shift_probability > np.random.uniform():
            horizontal_shift = np.random.randint(-4,4)
            vertical_shift = np.random.randint(-4,4)

            # data shape: [34,34,2,len], [h, w, channel, len]
            data = np.roll(img,horizontal_shift,axis=1)
            data = np.roll(img,vertical_shift,axis=0)

        label = self.data[idx][1]

        return img, label


class Dvs128Dataset(Dataset):
    """
    dvs 128 data
    
    """

    def __init__(self, file_list, length, rand_length_clip_probability = 0, rand_length_pad_probability = 0,
                 rand_space_transform_probability = 0, flatten=False, transform=None):
        self.file_list = file_list

        self.transform = transform
        self.flatten = flatten
        self.length = length

        self.rand_length_clip_probability = rand_length_clip_probability
        self.rand_length_pad_probability = rand_length_pad_probability
        self.rand_space_transform_probability = rand_space_transform_probability

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        file_info_dict = self.file_list[idx]

        # shape [length, 128, 128, 32]
        # shape [length, 128, 128, 32]
        data = np.load(file_info_dict['path'])
        data = data.astype(np.float32)

        dvs_length,_,_,_ = data.shape

        #dvs length is larger than specified length
        start_offset = 0

        length_diff = dvs_length - self.length
        if length_diff > 0:
            if self.rand_length_clip_probability > np.random.uniform():
                start_offset = np.random.randint(0, length_diff + 1)

        data = data[start_offset:start_offset+self.length]

        if length_diff < 0:
            choice = np.random.randint(0,4)

            if self.rand_length_pad_probability > np.random.uniform():
                choice = 0

            extended_data = np.zeros(self.length,128,128,2)
            # 4 ways to handle this
            # 1. pad 0 in rest part
            if choice == 0:
                extended_data[0:dvs_length] = data
            # 2. repeat to fill the length
            elif choice == 1:
                repeat_time = int(np.ceil(self.length/dvs_length))
                tiled_data = np.tile(data, (repeat_time, 1,1,1))
                extended_data = tiled_data[:self.length]
            # 3. insert adjacent frames evenly
            # elif choice == 2:
            #
            # # 4. insert adjacent frame in random way


        # data *= 2.0
        label = file_info_dict['label']

        if self.rand_space_transform_probability > np.random.uniform():
            #shift up/down, left, right

            horizontal_shift = np.random.randint(-10,10)
            vertical_shift = np.random.randint(-10,10)

            # data shape: [len, 128, 128, 2], [batch, length, h, w, channel]
            data = np.roll(data,horizontal_shift,axis=2)
            data = np.roll(data,vertical_shift,axis=1)
        return data, label

class Dvs128DatasetH5(Dataset):
    """
    dvs 128 data
    
    """

    def __init__(self, h5_path, length, is_train: bool, dt = 30, rand_length_clip_probability = 0, rand_length_pad_probability = 0,
                 rand_space_transform_probability = 0, flatten=False, skip_random_gesture=True):
        
        self.data = H5IBMGesture(h5_path, is_train)

        self.ToDense = ToDense(dt*ms)

        self.flatten = flatten
        self.length = length

        self.is_train = is_train
        self.skip_random_gesture = skip_random_gesture

        self.rand_length_clip_probability = rand_length_clip_probability
        self.rand_length_pad_probability = rand_length_pad_probability
        self.rand_space_transform_probability = rand_space_transform_probability

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        spike_train, label = self.data[idx]

        # class 11 is random gesture
        if label == 11 and self.skip_random_gesture == True:
            # if get a sample of random gesture, randomly choose another sample
            idx = np.random.randint(0, 10)

            spike_train, label = self.data[idx]

        # in h5 file, class starts from 1, so minus 1
        label -= 1

        # shape [128, 128, 2, length] [hight, width, channel, length]
        data = self.ToDense(spike_train)
        data = data.numpy()

        # swap axis to [length, hight, width, channel]
        data = np.transpose(data, (3, 0, 1, 2))

        dvs_length,_,_,_ = data.shape

        #dvs length is larger than specified length
        start_offset = 0

        length_diff = dvs_length - self.length
        if length_diff > 0:
            if self.rand_length_clip_probability > np.random.uniform():
                start_offset = np.random.randint(0, length_diff + 1)

        data = data[start_offset:start_offset+self.length]

        if length_diff < 0:
            choice = np.random.randint(0,2)

            if self.rand_length_pad_probability > np.random.uniform():
                choice = 0

            extended_data = np.zeros((self.length,128,128,2),dtype=np.float32)
            # 4 ways to handle this
            # 1. pad 0 in rest part
            if choice == 0:
                extended_data[0:dvs_length] = data
            # 2. repeat to fill the length
            elif choice == 1:
                repeat_time = int(np.ceil(self.length/dvs_length))
                tiled_data = np.tile(data, (repeat_time, 1,1,1))
                extended_data = tiled_data[:self.length]
            # 3. insert adjacent frames evenly
            # elif choice == 2:
            #
            # # 4. insert adjacent frame in random way

            data = extended_data

        if self.rand_space_transform_probability > np.random.uniform():
            #shift up/down, left, right

            horizontal_shift = np.random.randint(-10,10)
            vertical_shift = np.random.randint(-10,10)

            # data shape: [len, 128, 128, 2], [batch, length, h, w, channel]
            data = np.roll(data,horizontal_shift,axis=2)
            data = np.roll(data,vertical_shift,axis=1)

        # shape of data: [length, h, w, channel]
        # swap axis to fit conv layer [channel,h,w,length]

        data = np.transpose(data, (3, 1, 2, 0))

        return data, label

class Dvs128_Dataset_Enhanced(Dataset):
    """
    dvs 128 data

    root: the path which contains .h5 file or tar.gz file
    length: length of output (the last dimension, in other words, number of frames). Samples that have less frames
            will be padded with 0. Samples that have more frames will be clipped to 30.
    is_train: bool, true for train set, false for test set
    dt: none or a number. If dt is given, inferred dt will be overwritten
    reshape: output shape. if none, output shape is [c,h,w,l]. if want to flatten it, reshape is [128*128*2,l]
    skip_random_gesture: class 11 is a randon gesture, true to skip this class.

    Usage:
    put .h5 or tar.gz file to root, folder structure should be like this：
    if root = './dataset', tar.gz o5 .h5 file should be located at './dataset/dvsgesture.h5' or './dataset/dvsgesture.tar.gz'
    This class will first search if there exists a .h5 file. If there is, it will load it. Otherwise it will search
    .tar.gz file, and extract it to 'root'/DvsGesture/. The extracted files are raw data in aedat format. Then this class
    will automatically parse them and create a .h5 file which contains all train and test samples. 
    A sample in .h5 file is shown below:

    print(dvsdataset.data[0][0])
    rec.array([( 43,  58, False,       0), ( 53, 107, False,      15),
           ( 53, 107, False,      15), ..., ( 44, 109,  True, 4568789),
           ( 55, 108,  True, 4568813), (125,   8, False, 4568973)],
          dtype=[('x', '<u2'), ('y', '<u2'), ('p', '?'), ('ts', '<u8')])

    Essentially each item is a list of tuples, each tuple is a spike event (x, y, polarity, time stamp)
    Since the parsing and conversion are slow, you can keep the ,h5 file for future use.

    Notes:
    The length of each sample in this dataset varies significantly, the longest sample is 10 time longer than
    shortest sample. So it is important and also tricky to choose a proper dt/length. Consider following cases:
    1. Suppose we want length 30, and the longes sample can just fit 30 frames, because we need unified length.
        shorter samples have to be padded with 0 to get 30 frames. thus dt = 15591329 / (30*1000) = 519 ms
        The shortest sample only has  1798364/(519*1000) = 3.4 = 4 frames. The rest 26 frames are padded with 0.
    2. Suppose we want length 30, and the shortest sample can just fit 30 frames. Samples that longer than this will
        be clipped. thus dt = 1798364 / (30*1000) = 59.9 ms. The longest sample will have 15591329/(59.9*1000) = 260
        frames, 260 - 30 = 230 frames will be clipped.
    
    Obviously, above two choices are not proper because we have to either clip or pad too much. So I calculate dt based
    on average length. dt is calclated as:
        dt = (average length in us) / (length * 1000)
    
    Or you can also specify a dt, it will overwrite inferred dt.
    
    below are statistics:
    choose a appropriate dt or length.
    max length in test set: 15591329 us
    min length in test set: 1798364 us
    average length in test set: 6805963 us

    max length in train set: 18456873 us
    min length in train set: 1749843 us
    average length in train set: 6464258 us
    
    """

    def __init__(self, root, length, is_train: bool, dt = None, reshape=None, skip_random_gesture=True):
        

        self.dataset_h5_file_default_name = 'dvs_gesture.h5'
        self.url = 'http://ibm.biz/EventCameraData'

        self.data = None
        self.reshape = reshape
        self.length = length
        self.is_train = is_train
        self.skip_random_gesture = skip_random_gesture

        # shape of output spike trains, e.g. [c,h,w,l] or [c*h*w,l]
        if reshape is not None:
            self.out_shape = (*reshape, length)
        else:
            self.out_shape = (2,128,128,length)

        # dataset statistics, don't change it !
        # length in us
        if is_train == True:
            self.max_time = 18456873
            self.average_time = 6464258
            self.min_time = 1749843
            pass
        else:
            self.max_time = 1798364
            self.average_time = 6805963
            self.min_time = 15591329
        
        # calculate desired dt which can satisfy length requirement
        inferred_dt = self.average_time/(self.length*1000)
        self.dt = inferred_dt

        # if dt is provided, overwrite inferred dt
        if dt is None:
            print('To guarantee length: {}, inferred dt is: {} ms'.format(length, self.dt))
        else:
            print('dt: {} ms is given, override inferred dt: {} ms'.format(dt, inferred_dt))
            self.dt = dt * 1000
        
        self.ToDense = ToDense(self.dt*ms)

        # find '.h5' file
        if os.path.isdir(root) == False:

            _, file_extension = os.path.splitext(root)
            if file_extension != '.h5':
                raise Exception('Need a .h5 file')
            
            self.data = H5IBMGesture(root, is_train)
            print('Successfully load file: {}'.format(root))
        else:
            # if given a path. try to find h5 file
            for file in os.listdir(root):
                if file.endswith(".h5"):
                    print('Find file: {} in {}, load it.'.format(file, root))
                    h5_file_path = os.path.join(root, file)
                    self.data = H5IBMGesture(h5_file_path, is_train)
                    print('Successfully load file: {}'.format(h5_file_path))
            
            # if still cannot find .h5 file, search .gz file
            if self.data is None:
                print('.h5 file is not found, search .gz file')
                for file in os.listdir(root):
                    if file.endswith(".gz"):

                        tar_file_path = os.path.join(root, file)
                        print('Find .tar.gz file: {}, extract it'.format(tar_file_path))
                        import tarfile

                        try:
                            file = tarfile.open(tar_file_path)
                            file.extractall(root)
                            file.close()
                            print('Successfully extracted file: {} to folder {}'.format(tar_file_path, os.path.join(root, 'DvsGesture')))
                        except:
                            print("Extract failed, or file already exists") 
    

                        # # file.extractal creates a subfolder 'DvsGesture'. Move all files inside it
                        # # to root
                        # import shutil
                            
                        # source_dir = os.path.join(root, 'DvsGesture/')
                        # file_names = os.listdir(source_dir)
                            
                        # for file_name in file_names:
                        #     shutil.move(os.path.join(source_dir, file_name), root)
                        
                        # os.rmdir(source_dir)
                        
                        print('Convert extracted dataset to h5 file')
                        
                        h5_file_path = os.path.join(root, self.dataset_h5_file_default_name)

                        # file.extractal creates a subfolder 'DvsGesture'.
                        H5IBMGesture.convert(dvs_folder_path=os.path.join(root, 'DvsGesture'), h5_output_path=h5_file_path)

                        print('Successfully converted dataset to file: {}'.format(h5_file_path))
                        print('Load .h5 file: {}'.format(h5_file_path))

                        self.data = H5IBMGesture(h5_file_path, is_train)
                        print('Successfully load file: {}'.format(h5_file_path))
                
            if self.data is None:
                raise Exception(('Not able to find .h5 or .gz file!\n Download it from {}'.format(self.url)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        spike_train, label = self.data[idx]

        # class 11 is random gesture
        while label == 11 and self.skip_random_gesture == True:

            # if get a sample of random gesture, randomly choose another sample
            idx = np.random.randint(0, self.__len__())
            spike_train, label = self.data[idx]

        # in h5 file, class starts from 1, so minus 1
        label -= 1

        # shape [128, 128, 2, length] [hight, width, channel, length]
        data = self.ToDense(spike_train)
        # data = data.numpy()

        # # swap axis to [length, hight, width, channel]
        data = data.permute(3, 0, 1, 2)

        dvs_length,_,_,_ = data.shape

        # #dvs length is larger than specified length
        length_diff = dvs_length - self.length
        if length_diff > 0:
            data_new = data[:self.length]
        elif length_diff < 0:
            data_new = torch.zeros(self.length, 128, 128, 2)
            data_new[:dvs_length] = data
        else:
            data_new = data

        # shape of data_new: [length, h, w, channel]
        # swap axis to fit conv layer [channel,h,w,length]
        data_new = data_new.permute(3, 1, 2, 0)

        if self.reshape is not None:
            data_new = data_new.reshape(self.reshape)

        return data_new, label


class Image_Dataset_Adapter(Dataset):
    """mnist dataset

    pytorch_dataset: torchvision dataset object
    length: the size of a window
    max_rate: max spike rate
    transform: torchvision transform object
    reshape: specify this if you want to flatten it
            for example: given cifar 10 dataset, image shape is [3,32,32], if you want to flatten it to input to mlp,
                        reshape should be (3*32*32,) so that output shape is [3*32*32, length]
    
    return: torch tensor representing an image by rate coding, shape [c, w, h, l] or [c*h*w, l]

    # attention: 1. torchvision.transforms.ToTensor() automatically scales image pixel value range to [0,1],
                2. If input is grayscale, in other words, shape is [h,w], torchvision.transforms.ToTensor() automatically
                    insert a dimension, such that returned tensor is [1,h,w].
                3. If you want to flatten image, must specify argument 'reshape'

    How to use:

    This class convert torchvision dataset to spike format. output shape is [c,h,w,l] or [c*h*w,l]

    from torch.utils.data import Dataset, DataLoader

    # example 1, output shape is [1, 28, 28, 100]:
    #load mnist training dataset
    mnist_trainset = datasets.MNIST(root='./data', train=True, download=True)
    train_data = Image_To_Poisson_Spike(mnist_trainset, max_rate=1, length=100)
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)

    # example 2, output shape is [3*32*32, 100]:
    cifar10_trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True)
    cifar10_spike_reshape = Image_Dataset_Adapter(cifar10_trainset, 100, mode='poisson', reshape=(32*32*3,))

    """

    def __init__(self, pytorch_dataset, length, mode = 'poisson', reshape=None, max_rate=1, transform=None, return_original=False):
        self.dataset = pytorch_dataset
        self.transform = transform

        # used to reshape image.
        self.reshape = reshape
        self.length = length
        self.max_rate = max_rate
        self.mode = mode # can be 'poisson', 'even interval', 'repeat'

        self.return_original = return_original

        # pytorch_dataset may be PIL or numpy array, check type
        # if it is PIL or numpy array, need to transform to torch tensor
        self.to_tensor = transforms.ToTensor()

        self.isTensorInput = False

        import PIL
        if isinstance(pytorch_dataset[0][0], np.ndarray):
            pass
        elif isinstance(pytorch_dataset[0][0], PIL.Image.Image):
            pass
        elif isinstance(pytorch_dataset[0][0], torch.Tensor):
            self.isTensorInput = True
        else:
            raise Exception('Dataset type: {}'.format(type(pytorch_dataset[0][0])))
        
        # shape of output spike trains, e.g. [c,h,w,l] or [c*h*w,l]
        if reshape is not None:
            self.out_shape = (*reshape, length)
        else:
            if self.isTensorInput != True:
                self.out_shape = (*self.to_tensor(pytorch_dataset[0][0]).shape, length)
        
        if self.mode == 'poisson':
            self.converter = self.to_poisson_spike
        elif self.mode == 'even interval':
            self.converter = self.to_even_interval_spike
        elif self.mode == 'repeat':
            self.converter = self.repeat
        else:
            raise Exception(('wrong mode!'))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img = self.dataset[idx][0]
        if self.transform:
            img = self.transform(img)

        # img can either be a numpy array or pil image
        # to_tensor uses torchvision.transforms.ToTensor(), it automatically does normalization, so img is scaled to [0,1]
        # it also does dimension expension if image shape is [h,w], so that output shape of to_tensor() is [1,h,w]
        if self.isTensorInput != True:
            img = self.to_tensor(img)

        img = img * self.max_rate

        spike_trains = self.converter(img, self.length)

        if self.reshape is not None:
            spike_trains = spike_trains.reshape(self.out_shape)

        if self.return_original:
            return spike_trains, self.dataset[idx][1], img

        return spike_trains, self.dataset[idx][1]

    @staticmethod
    def to_poisson_spike(img, length):

        # shape of spike trains
        spike_trains = torch.zeros(*img.shape, length)

        #extend last dimension for time, repeat image along the last dimension
        img_tile = torch.unsqueeze(img, -1) # shape [c, h, w, 1]

        # copy along time dimension
        img_tile = img_tile.repeat((1,1,1,length)) # shape [c,h,w,length]

        # random number tensor, shape is same as img_tile
        rand = torch.rand(img_tile.shape)

        spike_trains[torch.where(img_tile > rand)] = 1

        return spike_trains

    @staticmethod
    def to_even_interval_spike(img, length):

        '''
        convert pixel values to spike train, spikes are evenly distributed
        very slow. Not recommended.
        '''

        shape = img.shape

        flattened = img.reshape(-1)

        # because image is scaled to [0,1], to calculate spike number,
        # we need to scale to to [0, length]
        flattened = flattened * length

        spike_trains = torch.zeros(*flattened.shape, length)

        for idx, row in enumerate(spike_trains):

            # length-1: do not include end point
            # pytorch does not have argument endpoiont as numpy
            non_zero_idx = torch.linspace(0, length-1, int(flattened[idx]))
            non_zero_idx = non_zero_idx.type(torch.long)
            spike_trains[idx, non_zero_idx] = 1
        
        spike_trains = spike_trains.reshape(*shape, length)
        
        return spike_trains
    
    @staticmethod
    def repeat(img, length):
        '''
        Repeat image along time dimension.

        This can be used with if_encoder as an alternative way to generate even interval spike trains. 
        Hopefully it is faster.
        See below example:

        cifar10_trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True)
        # mode should be repeat
        cifar10_spike = Image_Dataset_Adapter(cifar10_trainset, 100, mode='repeat')

        class mysnn(torch.nn.Module):
            def __init__(self):
                super().__init__()

                self.encoder = if_encoder((3,34,34)), 500, self.length, self.batch_size)
                self.dense_1 = neuron_layer(500, 500, self.length, self.batch_size)
            def forward(self, inputs):

                encoder_out,_ = self.encoder(inputs)
                dense1_out,_ = self.dense_1(encoder_out)
        
        encoder_out is a batch of even interval spike trains representing images

        '''

        #extend last dimension for time, repeat image along the last dimension
        img_tile = torch.unsqueeze(img, -1) # shape [c, h, w, 1] 

        # copy along time dimension
        img_tile = img_tile.repeat((1,1,1,length)) # shape [c,h,w,self.length]

        return img_tile

    @staticmethod
    def repeat_batch(img, length):
        '''
        img: shape [b,c,h,w]
        Repeat a batch of images along time dimension.

        '''

        #extend last dimension for time, repeat image along the last dimension
        img_tile = torch.unsqueeze(img, -1) # shape [b, c, h, w, 1] 

        # copy along time dimension
        img_tile = img_tile.repeat((1,1,1,1,length)) # shape [b,c,h,w,self.length]

        return img_tile
class SHD_Dataset(Dataset):
    """
    SHD dataset

    root: the folder which stores dataset
    length: number of steps
    train: if true, use train dataset, otherwise test dataset

    return:
    pytorch tensor, shape [700, length]

    """

    def __init__(self, root, length, train=True, transform=None):

        self.root = root
        self.transform = transform

        self.length = length
        self.train = train

        self.base_url = "https://compneuro.net/datasets"

        self.origin_file_names = ['shd_train.h5.gz', 'shd_test.h5.gz']
        self.file_names = ['shd_train.h5', 'shd_test.h5']

        make_dir(self.root)

        if self.train == True:
            self.file_path = os.path.join(root, self.file_names[0])
            self.origin_name = self.origin_file_names[0]
        else:
            self.file_path = os.path.join(root, self.file_names[1])
            self.origin_name = self.origin_file_names[1]
            
        if os.path.isfile(self.file_path):
            print('File: {} exists. Load file: {}'.format(self.file_path, self.file_path))
        else:
            print('Dataset not found, download it')

            origin_url = "%s/%s"%(self.base_url, self.origin_name)
            self.get_and_gunzip(origin_url, self.origin_name, self.root)

        # conversion

        import tables
        fileh = tables.open_file(self.file_path, mode='r')
        units = fileh.root.spikes.units
        times = fileh.root.spikes.times
        labels = fileh.root.labels
        extra = fileh.root.extra

        print('Converting')
        self.data, self.label = self.shd2mat_fast(times, units, labels, None, length, 1, 10000, np.float32)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):

        item = self.data[idx]

        label = self.label[idx]

        return item, label

    
    @staticmethod
    def download(dir):
        '''
        download and unzip
        '''
        
        make_dir(dir)

        # %% Download the Spiking Heidelberg Digits (SHD) dataset
        files = [ "shd_train.h5.gz", 
                "shd_test.h5.gz",
            ]

        extracted_files = []

        for fn in files:
            origin = "%s/%s"%(SHD_Dataset.base_url,fn)
            hdf5_file_path = SHD_Dataset.get_and_gunzip(origin, fn, dir)
            print(hdf5_file_path)
            extracted_files.append(hdf5_file_path)
        
        return extracted_files

    @staticmethod
    def get_and_gunzip(origin, filename, path):

        import wget, gzip, shutil

        filepath = os.path.join(path, filename)
        if os.path.exists(filepath) == False:
            print('Downloading: {}'.format(origin))
            wget.download(origin, filepath)
        else:
            print('{} already exists, skip downloading'.format(filename))

        gz_file_path = filepath
        hdf5_file_path=gz_file_path[:-3]
        if not os.path.isfile(hdf5_file_path) or os.path.getctime(gz_file_path) > os.path.getctime(hdf5_file_path):
            print("Decompressing %s"%gz_file_path)
            with gzip.open(gz_file_path, 'r') as f_in, open(hdf5_file_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        return hdf5_file_path

    @staticmethod
    def shd2mat_fast(times, units, labels, classes = None, step = 100, max_time = 1, numberPerClass=10000, dtype=np.int8):
        '''
        A fast function convert time events to matrix
        Adopted from https://github.com/fzenke/spytorch/blob/master/notebooks/SpyTorchTutorial4.ipynb
        '''
        from tqdm import tqdm

        new_label = []
        dataSet = []

        time_bins = np.linspace(0, max_time, num=step)

        if classes is None:
            classCount = np.zeros(20)
        else:
            classCount = np.zeros(len(classes))
        totalSamples = numberPerClass * 20
        count = 0

        for i in tqdm(range(len(times))):
        # for i in range(len(times)):

            mat = np.zeros((700, step),dtype=dtype)
            time_idx = np.digitize(times[i], time_bins) # col
            neuron_idx = units[i]                       # row

            # label 0 - 9 are english digits
            if(classes is None or labels[i] in classes):
                
                if count > totalSamples:
                    break

                if numberPerClass <= classCount[labels[i]]:
                    continue
                
                if (time_idx.max() >= step):
                    idx_within_window = np.where(time_idx<step)
                    time_idx = time_idx[idx_within_window]
                    neuron_idx = neuron_idx[idx_within_window]
                
                mat[neuron_idx, time_idx] = 1

                classCount[labels[i]] += 1
                count += 1

                dataSet.append(mat)
                new_label.append(labels[i])
        
        dataSet = np.stack(dataSet)
        new_label = np.array(new_label)

        return dataSet.astype(dtype), new_label.astype(dtype)

def make_dir(dir):

    if os.path.isdir(dir) == True:
        print('Folder {} exists.'.format(dir))
    else:
        print('Folder {} does not exist, create it'.format(dir))
        try:
            os.mkdir(dir)
        except OSError:
            print ("Creation of the directory %s failed" % dir)
        else:
            print ("Successfully created the directory %s " % dir)


# %%
class NMNIST_Dataset_Enhanced(Dataset):
    """
    NMNIST dataset

    root: the path to dataset
    is_train: set True for train set, otherwise test set
    dt: if not provide, infer dt from length. if provided, overwrite inferred dt
    download: if cannot find dataset, download it
    dt: bin size, unit is ms

    Notes:
    NMNIST stores in AER(address event representation) formar. Essentially it consists of events flow.
    An event consists of x y coordinate, polarity and time stamp. In addition, the time resolution is very high (us).
    However, pytorch requires dense matrix format as input data. Therefore, we need to bin the dataset along time.
    For example, assume a bin size 10 ms, at particular position (X,Y), the event flow ranges from 0 ms to 1000 ms.
    Hence after bin it, we have an array, length is ceil(1000/10) = 100. Each element in this array corresponds to a 10 ms interval.
    in each 10 ms interval, as long as there is a spike, corresponding position in the array is set to 1.
    This conversion is done by ebdataset.

    The lengths of each samples in NMNIST are not identical. Therefore, after we bin them, resulting array does not have
    identical. The work around is set a fixed length, if binned array is longer, we randomly crop it. If binned array is shorter,
    we pad 0.

    
    """

    def __init__(self, root, length, is_train: bool, dt = None,  reshape=None, download=True):
        
        from quantities import ms

        self.root = root
        self.length = length
        self.reshape = reshape

        # shape of output spike trains, e.g. [c,h,w,l] or [c*h*w,l]
        if reshape is not None:
            self.out_shape = (*reshape, length)
        else:
            self.out_shape = (2,34,34,length)

        # dataset statistics, don't change it !
        # length in us
        if is_train == True:
            self.max_time = 315474
            self.average_time = 307663
            self.min_time = 295483
            pass
        else:
            self.max_time = 336040
            self.average_time = 307660
            self.min_time = 296072
        
        # calculate desired dt which can satisfy length requirement
        inferred_dt = self.max_time/(self.length*1000)
        self.dt = inferred_dt

        # if dt is provided, overwrite inferred dt
        if dt is None:
            print('To guarantee length: {}, inferred dt is: {} ms'.format(length, self.dt))
        else:
            print('dt: {} ms is given, override inferred dt: {} ms'.format(dt, inferred_dt))
            self.dt = dt * 1000

        self.bin_transform = ToDense(self.dt*ms)

        self.data = NMnist('./mnist', is_train=True, transforms=self.bin_transform, download_if_missing=download)

        print('dataset time resolution: us')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        
        # spike_train shape [h,w,c,l] [34,34,2,length]
        spike_train, label = self.data[idx]

        # shape [l,h,w,c]
        spike_train = spike_train.permute(3,0,1,2)

        if spike_train.shape[0] == self.length:
            pass
        elif spike_train.shape[0] > self.length:

            diff_len = spike_train.shape[0] - self.length
            offset = np.random.randint(0, diff_len+1)
            # random clip
            spike_train = spike_train[offset:offset+self.length]

        else:
            diff_len = self.length - spike_train.shape[0]
            
            #offset = np.random.randint(0, diff_len+1)

            padded = torch.zeros(self.length,34,34,2)

            padded[:spike_train.shape[0]] = spike_train

            spike_train = padded   
        
        # change shape from [l,h,w,c] to [c,h,w,l]
        spike_train = spike_train.permute(3,1,2,0)

        if self.reshape is not None:
            spike_train = spike_train.reshape(self.out_shape)
        
        return spike_train,label
    
    def get_statistic(self):

        length_list = [x[0].ts[-1] for x in self.data]
        length_list = np.array(length_list)
        max_time = length_list.max()
        average_time = length_list.mean()
        min_time = length_list.min()

        return max_time, min_time, average_time
