import healpy as hp
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import camb
import os
import pysm3
import pysm3.units as u
from tqdm import tqdm

T0 = 2.726 # K

def conv3x3(in_channels, out_channels, stride=1, 
            padding=1, bias=True, groups=1):    
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=padding,
        bias=bias,
        groups=groups)

def upconv2x2(in_channels, out_channels, mode='transpose'):
    if mode == 'transpose':
        return nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2)
    else:
        # out_channels is always going to be the same
        # as in_channels
        return nn.Sequential(
            nn.Upsample(mode='bilinear', scale_factor=2),
            conv1x1(in_channels, out_channels))

def conv1x1(in_channels, out_channels, groups=1):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=1,
        groups=groups,
        stride=1)


class DownConv(nn.Module):
    """
    A helper Module that performs 2 convolutions and 1 MaxPool.
    A ReLU activation follows each convolution.
    """
    def __init__(self, in_channels, out_channels, pooling=True):
        super(DownConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pooling = pooling

        self.conv1 = conv3x3(self.in_channels, self.out_channels)
        self.conv2 = conv3x3(self.out_channels, self.out_channels)

        if self.pooling:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        before_pool = x
        if self.pooling:
            x = self.pool(x)
        return x, before_pool


class UpConv(nn.Module):
    """
    A helper Module that performs 2 convolutions and 1 UpConvolution.
    A ReLU activation follows each convolution.
    """
    def __init__(self, in_channels, out_channels, 
                 merge_mode='concat', up_mode='transpose'):
        super(UpConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.merge_mode = merge_mode
        self.up_mode = up_mode

        self.upconv = upconv2x2(self.in_channels, self.out_channels, 
            mode=self.up_mode)

        if self.merge_mode == 'concat':
            self.conv1 = conv3x3(
                2*self.out_channels, self.out_channels)
        else:
            # num of input channels to conv2 is same
            self.conv1 = conv3x3(self.out_channels, self.out_channels)
        self.conv2 = conv3x3(self.out_channels, self.out_channels)


    def forward(self, from_down, from_up):
        """ Forward pass
        Arguments:
            from_down: tensor from the encoder pathway
            from_up: upconv'd tensor from the decoder pathway
        """
        from_up = self.upconv(from_up)
        if self.merge_mode == 'concat':
            x = torch.cat((from_up, from_down), 1)
        else:
            x = from_up + from_down
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return x


class UNet(nn.Module):
    """ `UNet` class is based on https://arxiv.org/abs/1505.04597

    The U-Net is a convolutional encoder-decoder neural network.
    Contextual spatial information (from the decoding,
    expansive pathway) about an input tensor is merged with
    information representing the localization of details
    (from the encoding, compressive pathway).

    Modifications to the original paper:
    (1) padding is used in 3x3 convolutions to prevent loss
        of border pixels
    (2) merging outputs does not require cropping due to (1)
    (3) residual connections can be used by specifying
        UNet(merge_mode='add')
    (4) if non-parametric upsampling is used in the decoder
        pathway (specified by upmode='upsample'), then an
        additional 1x1 2d convolution occurs after upsampling
        to reduce channel dimensionality by a factor of 2.
        This channel halving happens with the convolution in
        the tranpose convolution (specified by upmode='transpose')
    """

    def __init__(self, num_classes, in_channels=3, depth=5, 
                 start_filts=64, up_mode='transpose', 
                 merge_mode='concat'):
        """
        Arguments:
            in_channels: int, number of channels in the input tensor.
                Default is 3 for RGB images.
            depth: int, number of MaxPools in the U-Net.
            start_filts: int, number of convolutional filters for the 
                first conv.
            up_mode: string, type of upconvolution. Choices: 'transpose'
                for transpose convolution or 'upsample' for nearest neighbour
                upsampling.
        """
        super(UNet, self).__init__()

        if up_mode in ('transpose', 'upsample'):
            self.up_mode = up_mode
        else:
            raise ValueError("\"{}\" is not a valid mode for "
                             "upsampling. Only \"transpose\" and "
                             "\"upsample\" are allowed.".format(up_mode))
    
        if merge_mode in ('concat', 'add'):
            self.merge_mode = merge_mode
        else:
            raise ValueError("\"{}\" is not a valid mode for"
                             "merging up and down paths. "
                             "Only \"concat\" and "
                             "\"add\" are allowed.".format(up_mode))

        # NOTE: up_mode 'upsample' is incompatible with merge_mode 'add'
        if self.up_mode == 'upsample' and self.merge_mode == 'add':
            raise ValueError("up_mode \"upsample\" is incompatible "
                             "with merge_mode \"add\" at the moment "
                             "because it doesn't make sense to use "
                             "nearest neighbour to reduce "
                             "depth channels (by half).")

        self.num_classes = num_classes
        self.in_channels = in_channels
        self.start_filts = start_filts
        self.depth = depth

        self.down_convs = []
        self.up_convs = []

        # create the encoder pathway and add to a list
        for i in range(depth):
            ins = self.in_channels if i == 0 else outs
            outs = self.start_filts*(2**i)
            pooling = True if i < depth-1 else False

            down_conv = DownConv(ins, outs, pooling=pooling)
            self.down_convs.append(down_conv)

        # create the decoder pathway and add to a list
        # - careful! decoding only requires depth-1 blocks
        for i in range(depth-1):
            ins = outs
            outs = ins // 2
            up_conv = UpConv(ins, outs, up_mode=up_mode,
                merge_mode=merge_mode)
            self.up_convs.append(up_conv)

        self.conv_final = conv1x1(outs, self.num_classes)

        # add the list of modules to current module
        self.down_convs = nn.ModuleList(self.down_convs)
        self.up_convs = nn.ModuleList(self.up_convs)

        self.reset_params()

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv2d):
            init.xavier_normal(m.weight)
            init.constant(m.bias, 0)


    def reset_params(self):
        for i, m in enumerate(self.modules()):
            self.weight_init(m)


    def forward(self, x):
        encoder_outs = []
         
        # encoder pathway, save outputs for merging
        for i, module in enumerate(self.down_convs):
            x, before_pool = module(x)
            encoder_outs.append(before_pool)

        for i, module in enumerate(self.up_convs):
            before_pool = encoder_outs[-(i+2)]
            x = module(before_pool, x)
        
        # No softmax is used. This means you need to use
        # nn.CrossEntropyLoss is your training script,
        # as this module includes a softmax already.
        x = self.conv_final(x)
        return x

def _expand_array( original_array):
    '''
    to be used in nestedArray2nestedMap, expand the given small array into a large array
    
    :param original_array: with the shape of (2**n, 2**n), where n=1, 2, 4, 6, 8, 10, ...
    '''
    add_value = original_array.shape[0]**2
    array_0 = original_array
    array_1 = array_0 + add_value
    array_2 = array_0 + add_value*2
    array_3 = array_0 + add_value*3
    array_3_1 = np.c_[array_3, array_1]
    array_2_0 = np.c_[array_2, array_0]
    array = np.r_[array_3_1, array_2_0]
    return array

def _ordinal_array(nside):
    '''
    obtain an array containing the ordinal number, the shape is (nside, nside)
    '''
    circle_num = (int(np.log2((nside/2)**2)) - 2)//2 #use //
    ordinal_array = np.array([[3.,1.],[2.,0.]])
    for i in range(circle_num):
        ordinal_array = _expand_array(ordinal_array)
    return ordinal_array, circle_num

def nestedArray2nestedMap(map_cut, nside):
    '''
    reorder the cut map into NESTED ordering to show the same style using 
    plt.imshow() as that using Healpix
    
    :param map_cut: the cut map, the shape of map_cut is (nside**2,)
    
    return the reorded data, the shape is (nside, nside)
    '''
    array_fill, circle_num = _ordinal_array(nside)
    for i in range(2**(circle_num+1)):
        for j in range(2**(circle_num+1)):
            array_fill[i][j] = map_cut[int(array_fill[i][j])]
    #array_fill should be transposed to keep the figure looking like that in HEALPix
    array_fill = array_fill.T
    return array_fill

def nestedMap2nestedArray(map_block, nside):
        '''
        Restore the cut map(1/12 of full sky map) into an array which is in NESTED ordering
        
        need transpose if the map is transposed in nestedArray2nestedMap function
        '''
        map_block = map_block.T #!!!
        map_cut = np.ones((nside//2)**2)
        array_fill, circle_num = _ordinal_array(nside)
        for i in range(2**(circle_num+1)):
            for j in range(2**(circle_num+1)):
                map_cut[int(array_fill[i][j])] = map_block[i][j]
        return map_cut
    
def _full(map_block, nside, indices_interes, base_map = None):
        '''
        return a full sphere map
        '''
        map_block_array = nestedMap2nestedArray(map_block, nside)
        map_NEST = np.zeros(12*nside**2)
    
        map_NEST[indices_interes] = map_block_array
        map_RING = hp.reorder(map_NEST, n2r=True)
        return map_RING

def cmb_temperature_map(nside, pars, indices_interes):
    results = camb.get_results(pars)
    powers = results.get_cmb_power_spectra(pars, raw_cl=True)#, CMB_unit='muK')
    totCL=powers['total']
    ClTT=totCL[:,0]#/T0**2 #Dimensionless
    alm = hp.sphtfunc.synalm(ClTT)
    map_cmb_ring = T0*hp.sphtfunc.alm2map(alm, nside=nside)*1e6 # muK
    map_cmb_nest = hp.reorder(map_cmb_ring, r2n=True)
    map_cmb_array = map_cmb_nest[indices_interes]
    box_cmb = nestedArray2nestedMap(map_cmb_array, nside) 
    return box_cmb

def dust_temperature_map(nside, freq, indices_interes):
    sky_dust = pysm3.Sky(nside=nside, preset_strings=['d1'])
    map_dust = sky_dust.get_emission(freq * u.GHz)
    map_dust = map_dust.to(u.uK_CMB, equivalencies=u.cmb_equivalencies(freq*u.GHz))
    map_dust_values = map_dust.value
    map_dust_nest = hp.reorder(map_dust_values[0], r2n=True)
    map_dust_array = map_dust_nest[indices_interes]
    box_dust = nestedArray2nestedMap(map_dust_array, nside) 
    return box_dust

def frequency_cont_maps(nside,pars,freqs, indices_interes):
    box_cont_freq = []
    box_cmb = cmb_temperature_map(nside, pars, indices_interes)
    for i in range(len(freqs)):
        box_cont = box_cmb + dust_temperature_map(nside,freqs[i],indices_interes)
        box_cont_freq.append(box_cont)
    return np.array(box_cont_freq), box_cmb

def train_set_generator(N, nside, pars, freqs, indices_interes):

    folder_train = "Train_set"
    if not os.path.exists(folder_train):
        os.makedirs(folder_train)
    os.chdir(folder_train)
    
    folder_clean = "Clean_maps"
    folder_cont = "Cont_maps"
    
    if not os.path.exists(folder_clean):
        os.makedirs(folder_clean)
    if not os.path.exists(folder_cont):
        os.makedirs(folder_cont)
        
    for i in tqdm(range(N)):
        gen_box_cont = frequency_cont_maps(nside,pars,freqs, indices_interes)
        box_cont = gen_box_cont[0]
        box_clean = gen_box_cont[1]
        os.chdir(folder_cont)
        np.save(str(i)+'cont.npy', box_cont)
        os.chdir('../'+str(folder_clean))
        np.save(str(i)+'clean.npy', box_clean)
        os.chdir('..')
    
    os.chdir('..')
    
    
def valid_set_generator(N, nside, pars, freqs, indices_interes):
    
    folder_valid = "Valid_set"
    if not os.path.exists(folder_valid):
        os.makedirs(folder_valid)
    os.chdir(folder_valid)
    
    folder_clean = "Clean_maps"
    folder_cont = "Cont_maps"
    
    if not os.path.exists(folder_clean):
        os.makedirs(folder_clean)
    if not os.path.exists(folder_cont):
        os.makedirs(folder_cont)
        
    for i in tqdm(range(N)):
        gen_box_cont = frequency_cont_maps(nside,pars,freqs, indices_interes)
        box_cont = gen_box_cont[0]
        box_clean = gen_box_cont[1]
        os.chdir(folder_cont)
        np.save(str(i)+'cont.npy', box_cont)
        os.chdir('../'+str(folder_clean))
        np.save(str(i)+'clean.npy', box_clean)
        os.chdir('..')
    
    os.chdir('..')
    
def test_set_generator(N, nside, pars, freqs, indices_interes):
    
    folder_test = "Test_set"
    if not os.path.exists(folder_test):
        os.makedirs(folder_test)
    os.chdir(folder_test)
    
    folder_clean = "Clean_maps"
    folder_cont = "Cont_maps"
    
    if not os.path.exists(folder_clean):
        os.makedirs(folder_clean)
    if not os.path.exists(folder_cont):
        os.makedirs(folder_cont)
        
    for i in tqdm(range(N)):
        gen_box_cont = frequency_cont_maps(nside,pars,freqs, indices_interes)
        box_cont = gen_box_cont[0]
        box_clean = gen_box_cont[1]
        os.chdir(folder_cont)
        np.save(str(i)+'cont.npy', box_cont)
        os.chdir('../'+str(folder_clean))
        np.save(str(i)+'clean.npy', box_clean)
        os.chdir('..')
    
    os.chdir('..')
    
    
def arrange_input(map_nums):
    os.chdir('Cont_maps')
    lista_archivos = os.listdir()
    maps = []
    for i in range(len(map_nums)):
        map_name = lista_archivos[map_nums[i]]
        map_loaded = np.load(map_name)
        map_tensor = torch.tensor(map_loaded)
        maps.append(map_tensor)
    inputs = torch.stack(maps, dim=0)
    inputs = inputs.to(torch.float32)
    os.chdir('..')
    return inputs


def arrange_target(map_nums):
    os.chdir('Clean_maps')
    lista_archivos = os.listdir()
    maps = []
    for i in range(len(map_nums)):
        map_name = lista_archivos[map_nums[i]]
        map_loaded = np.load(map_name)
        map_loaded = map_loaded[:, :]
        map_tensor = torch.tensor(map_loaded).unsqueeze(0)
        maps.append(map_tensor)
    targets = torch.stack(maps, dim=0)
    os.chdir('..')
    return targets
