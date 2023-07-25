"""code from https://github.com/tstandley/taskgrouping/blob/master/model_definitions/resnet_taskonomy.py"""
import torch.nn as nn
import torch
import math
import torch.nn.functional as F
from torch.hub import load_state_dict_from_url

__all__ = ['resnet18_taskonomy', 'resnet34_taskonomy',
		   'resnet50_taskonomy', 'resnet101_taskonomy',
		   'resnet152_taskonomy']


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
	"""3x3 convolution with padding"""
	return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
					 padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
	"""1x1 convolution"""
	return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
	expansion = 1

	def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
				 base_width=64, dilation=1, norm_layer=None):
		super(BasicBlock, self).__init__()
		if norm_layer is None:
			norm_layer = nn.BatchNorm2d
		if groups != 1 or base_width != 64:
			raise ValueError('BasicBlock only supports groups=1 and base_width=64')
		if dilation > 1:
			raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
		# Both self.conv1 and self.downsample layers downsample the input when stride != 1
		self.conv1 = conv3x3(inplanes, planes, stride)
		self.bn1 = norm_layer(planes)
		self.relu = nn.ReLU(inplace=True)
		self.conv2 = conv3x3(planes, planes)
		self.bn2 = norm_layer(planes)
		self.downsample = downsample
		self.stride = stride

	def forward(self, x):
		identity = x

		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)

		out = self.conv2(out)
		out = self.bn2(out)

		if self.downsample is not None:
			identity = self.downsample(x)

		out += identity
		out = self.relu(out)

		return out


class Bottleneck(nn.Module):
	expansion = 4

	def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
				 base_width=64, dilation=1, norm_layer=None):
		super(Bottleneck, self).__init__()
		if norm_layer is None:
			norm_layer = nn.BatchNorm2d
		width = int(planes * (base_width / 64.)) * groups
		# Both self.conv2 and self.downsample layers downsample the input when stride != 1
		self.conv1 = conv1x1(inplanes, width)
		self.bn1 = norm_layer(width)
		self.conv2 = conv3x3(width, width, stride, groups, dilation)
		self.bn2 = norm_layer(width)
		self.conv3 = conv1x1(width, planes * self.expansion)
		self.bn3 = norm_layer(planes * self.expansion)
		self.relu = nn.ReLU(inplace=True)
		self.downsample = downsample
		self.stride = stride

	def forward(self, x):
		identity = x

		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)

		out = self.conv2(out)
		out = self.bn2(out)
		out = self.relu(out)

		out = self.conv3(out)
		out = self.bn3(out)

		if self.downsample is not None:
			identity = self.downsample(x)

		out += identity
		out = self.relu(out)

		return out

class BasicWideBlock(nn.Module):
    """
    Implements a basic block module for WideResNets.
    Arguments:
        in_planes (int): number of input planes.
        out_planes (int): number of output filters.
        stride (int): stride of convolution.
        dropRate (float): dropout rate.
    """
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
        super(BasicWideBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                                                                padding=0, bias=False) or None

    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)

class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate)
    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate))
        return nn.Sequential(*layers)
    def forward(self, x):
        return self.layer(x)

class WideResNetEncoder(nn.Module):
    def __init__(self, depth, widen_factor=1, dropRate=0.0,
				 img_size=256):

        super(WideResNetEncoder, self).__init__()
        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        assert((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = BasicWideBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        #self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x, _eval=False):
        if _eval:
            # switch to eval mode
            self.eval()
        else:
            self.train()

        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))

        return out

        #out = F.avg_pool2d(out, 8)
        #out = out.view(-1, self.nChannels)

        #self.train()

        #return self.fc(out)

class ResNetEncoder(nn.Module):

	def __init__(self, block, layers, widths=[64, 128, 256, 512], num_classes=1000, zero_init_residual=False,
				 groups=1, width_per_group=64, replace_stride_with_dilation=None,
				 norm_layer=None, img_size=256):
		super(ResNetEncoder, self).__init__()
		if norm_layer is None:
			norm_layer = nn.BatchNorm2d
		self._norm_layer = norm_layer

		self.inplanes = 64
		self.dilation = 1
		if replace_stride_with_dilation is None:
			# each element in the tuple indicates if we should replace
			# the 2x2 stride with a dilated convolution instead
			replace_stride_with_dilation = [False, False, False]
		if len(replace_stride_with_dilation) != 3:
			raise ValueError("replace_stride_with_dilation should be None "
							 "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
		self.groups = groups
		self.base_width = width_per_group

		if img_size==32:
			self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=3,
								   bias=False)
			self.bn1 = norm_layer(self.inplanes)
			self.relu = nn.ReLU(inplace=True)
			self.maxpool = nn.Identity()
		else:
			self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
								   bias=False)
			self.bn1 = norm_layer(self.inplanes)
			self.relu = nn.ReLU(inplace=True)
			self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
		self.layer1 = self._make_layer(block, widths[0], layers[0])
		self.layer2 = self._make_layer(block, widths[1], layers[1], stride=2,
									   dilate=replace_stride_with_dilation[0])
		self.layer3 = self._make_layer(block, widths[2], layers[2], stride=2,
									   dilate=replace_stride_with_dilation[1])
		self.layer4 = self._make_layer(block, widths[3], layers[3], stride=2,
									   dilate=replace_stride_with_dilation[2])

		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
			elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)

		# Zero-initialize the last BN in each residual branch,
		# so that the residual branch starts with zeros, and each residual block behaves like an identity.
		# This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
		if zero_init_residual:
			for m in self.modules():
				if isinstance(m, Bottleneck):
					nn.init.constant_(m.bn3.weight, 0)
				elif isinstance(m, BasicBlock):
					nn.init.constant_(m.bn2.weight, 0)

	def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
		norm_layer = self._norm_layer
		downsample = None
		previous_dilation = self.dilation
		if dilate:
			self.dilation *= stride
			stride = 1
		if stride != 1 or self.inplanes != planes * block.expansion:
			downsample = nn.Sequential(
				conv1x1(self.inplanes, planes * block.expansion, stride),
				norm_layer(planes * block.expansion),
			)

		layers = []
		layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
							self.base_width, previous_dilation, norm_layer))
		self.inplanes = planes * block.expansion
		for _ in range(1, blocks):
			layers.append(block(self.inplanes, planes, groups=self.groups,
								base_width=self.base_width, dilation=self.dilation,
								norm_layer=norm_layer))

		return nn.Sequential(*layers)

	def forward(self, x):
		x = self.conv1(x)
		x = self.bn1(x)
		x = self.relu(x)
		x = self.maxpool(x)

		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)
		x = self.layer4(x)

		return x


class Decoder(nn.Module):
	def __init__(self, output_channels=32, num_classes=None, base_match=512, downscale=1):
		super(self.__class__, self).__init__()

		self.output_channels = output_channels
		self.num_classes = num_classes
		self.downscale = downscale

		self.relu = nn.ReLU(inplace=True)
		if num_classes is not None:
			self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
			# self.fc = nn.Linear(512, num_classes)
			self.fc = nn.Linear(base_match, num_classes)
		else:
			self.upconv0 = nn.ConvTranspose2d(base_match, 256, 2, 2)
			self.bn_upconv0 = nn.BatchNorm2d(256)
			self.conv_decode0 = nn.Conv2d(256, 256, 3, padding=1)
			self.bn_decode0 = nn.BatchNorm2d(256)
			self.upconv1 = nn.ConvTranspose2d(256, 128, 2, 2)
			self.bn_upconv1 = nn.BatchNorm2d(128)
			self.conv_decode1 = nn.Conv2d(128, 128, 3, padding=1)
			self.bn_decode1 = nn.BatchNorm2d(128)
			self.upconv2 = nn.ConvTranspose2d(128, 64, 2, 2)
			self.bn_upconv2 = nn.BatchNorm2d(64)
			self.conv_decode2 = nn.Conv2d(64, 64, 3, padding=1)
			self.bn_decode2 = nn.BatchNorm2d(64)

			if self.downscale==1:
				self.upconv3 = nn.ConvTranspose2d(64, 48, 2, 2)
				self.bn_upconv3 = nn.BatchNorm2d(48)
				self.conv_decode3 = nn.Conv2d(48, 48, 3, padding=1)
				self.bn_decode3 = nn.BatchNorm2d(48)
				self.upconv4 = nn.ConvTranspose2d(48, 32, 2, 2)
			elif self.downscale==2:
				self.upconv4 = nn.ConvTranspose2d(64, 32, 2, 2)


			self.bn_upconv4 = nn.BatchNorm2d(32)
			self.conv_decode4 = nn.Conv2d(32, output_channels, 3, padding=1)

	def forward(self, representation):
		# batch_size=representation.shape[0]
		if self.num_classes is None:
			# x2 = self.conv_decode_res(representation)
			# x2 = self.bn_conv_decode_res(x2)
			# x2 = interpolate(x2,size=(256,256))

			x = self.upconv0(representation)
			x = self.bn_upconv0(x)
			x = self.relu(x)
			x = self.conv_decode0(x)
			x = self.bn_decode0(x)
			x = self.relu(x)

			x = self.upconv1(x)
			x = self.bn_upconv1(x)
			x = self.relu(x)
			x = self.conv_decode1(x)
			x = self.bn_decode1(x)
			x = self.relu(x)
			x = self.upconv2(x)
			x = self.bn_upconv2(x)
			x = self.relu(x)
			x = self.conv_decode2(x)

			x = self.bn_decode2(x)
			x = self.relu(x)

			if self.downscale == 1:
				x = self.upconv3(x)
				x = self.bn_upconv3(x)
				x = self.relu(x)
				x = self.conv_decode3(x)
				x = self.bn_decode3(x)
				x = self.relu(x)
			x = self.upconv4(x)

			x = self.bn_upconv4(x)
			# x = torch.cat([x,x2],1)
			# print(x.shape,self.static.shape)
			# x = torch.cat([x,x2,input,self.static.expand(batch_size,-1,-1,-1)],1)
			x = self.relu(x)
			x = self.conv_decode4(x)

			# z = x[:,19:22,:,:].clone()
			# y = (z).norm(2,1,True).clamp(min=1e-12)
			# print(y.shape,x[:,21:24,:,:].shape)
			# x[:,19:22,:,:]=z/y

		else:
			x = representation
			#TODO: I add this

			x = F.adaptive_avg_pool2d(x, (1, 1))
			x = x.view(x.size(0), -1)
			# print("---------------------- x size ---------------------- : ", x.size(0))
			x = self.fc(x)
		return x


def get_tasks_to_params(num_classes=None):
	num_classes = (1000, 63) if num_classes is None else num_classes
	tsks =  {
		'autoencoder'       : {'output_channels' : 3,  'downscale' : 1},
		'gabor'       : {'output_channels' : 0,  'num_classes' : 96},
		'hog'       : {'output_channels' : 1,  'downscale' : 1},
		'sift': {'output_channels': 1, 'downscale': 1},
		'class_object'      	: {'output_channels' : 0, 'num_classes':num_classes[0]},
		'class_places'       	: {'output_channels' : 0, 'num_classes':num_classes[-1]},
		'depth_eucan'   	: {'output_channels' : 1,  'downscale' : 1},
		'depth_z'     	: {'output_channels' : 1,  'downscale' : 1},
		'depth'     : {'output_channels' : 1,  'downscale' : 1},
		'edge_occn'    	: {'output_channels' : 1,  'downscale' : 1},
		'edge_texture'      	: {'output_channels' : 1,  'downscale' : 1},
		'keypoints2d'       	: {'output_channels' : 1,  'downscale' : 1},
		'keypoints3d'       	: {'output_channels' : 1,  'downscale' : 1},
		'normal'            	: {'output_channels' : 3,  'downscale' : 1},
		'principal_curture' 	: {'output_channels' : 2,  'downscale' : 1},
		'reshading'         	: {'output_channels' : 1,  'downscale' : 1},
		'room_layout'       	: {'output_channels' : 0, 'num_classes':9},
		'segment_un25d'  	: {'output_channels' : 64, 'downscale' : 1},
		'segment_u2d'   	: {'output_channels' : 64, 'downscale' : 1},
		'segmentseic'   	: {'output_channels' : 18, 'downscale' : 1},
		'vanishingnt'   	: {'output_channels' : 0, 'num_classes':9},
		'sex'   : {'output_channels' : 0, 'num_classes':2},
		'age'   : {'output_channels' : 0, 'num_classes':1},
		'autoencoder1c': {'output_channels': 1, 'downscale': 1},
		'class_object#shape'      : {'output_channels' : 0, 'num_classes':2},
		'class_object#pose': {'output_channels': 0, 'num_classes': 3},
		'class_object#texture': {'output_channels': 0, 'num_classes': 2},
		'class_object#context': {'output_channels': 0, 'num_classes': 2},
		'class_object#weather': {'output_channels': 0, 'num_classes': 2},
		'class_object#imagenetr': {'output_channels': 0, 'num_classes': 15},

		'class_object#multilabel'      	: {'output_channels' : 0, 'num_classes':num_classes[0]},


	}

	if len(num_classes)>1:
		if isinstance(num_classes[1],dict):
			tsks = { **tsks,
					 'jigsaw': {'output_channels': 0, 'num_classes': num_classes[1].get("permutations_jigsaw")},
					 'rotation': {'output_channels': 0, 'num_classes': num_classes[1].get("nb_rotations")},
			'class_object#macro'      : {'output_channels': 0, 'num_classes': num_classes[1].get("nb_secondary_labels")},
			'class_object#detector'      : {'output_channels': 0, 'num_classes': num_classes[1].get("nb_secondary_labels")}
					 }
		else:
			tsks = {**tsks,
					'jigsaw': {'output_channels': 0, 'num_classes': num_classes[1]},
					'rotation': {'output_channels': 0, 'num_classes': num_classes[1]},
					'class_object#macro': {'output_channels': 0, 'num_classes': num_classes[1]},
					'class_object#detector': {'output_channels': 0, 'num_classes': num_classes[1]}
					}

	import torchxrayvision as xrv
	pathologies = xrv.datasets.default_pathologies
	for i, p in enumerate(pathologies):
		tsks[f"class_object#{p}"] = tsks["class_object"]
	return tsks

class MTResnet(nn.Module):

	def __init__(self, num_classes=None, tasks=None, base_match=512, size=1, **kwargs):
		super(MTResnet, self).__init__()

		self.task_to_decoder = {}
		self.tasks = tasks
		num_classes = (1000, 63) if num_classes is None else num_classes
		base = {1: 512, 2: 720, 3: 880, 2.5: 640, 0.5: 360}
		rescaled = {k: b * (base_match // 512) for k, b in base.items()}

		if tasks is not None:
			# self.final_conv = nn.Conv2d(728,512,3,1,1)
			# self.final_conv_bn = nn.BatchNorm2d(512)

			self.task_to_params = get_tasks_to_params(num_classes)
			for task in self.tasks: self.task_to_params[task]['base_match'] = rescaled[size]
			self.task_to_decoder = nn.ModuleDict({task: Decoder(**self.task_to_params[task]) for task in self.tasks})
		else:
			self.task_to_decoder =  nn.ModuleDict({"classification": Decoder(output_channels=0, base_match=base_match,num_classes=num_classes)})

		# print('task decoders name', self.task_to_decoder.values(), self.task_to_decoder.keys())

		# print("debug decoder sequence? ", self.decoders)

		# ------- init weights --------
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
				m.weight.data.normal_(0, math.sqrt(2. / n))
			elif isinstance(m, nn.BatchNorm2d):
				m.weight.data.fill_(1)
				m.bias.data.zero_()

	# -----------------------------

	def forward(self, input):
		rep = self.encoder(input)

		if self.tasks is None:
			return self.task_to_decoder['classification'](rep)

		# rep = self.final_conv(rep)
		# rep = self.final_conv_bn(rep)

		outputs = {'rep': rep}
		# if self.ozan:
		#     # OzanRepFunction.n = len(self.decoders)
		#     # rep = ozan_rep_function(rep)
		#     for i, (task, decoder) in enumerate(zip(self.task_to_decoder.keys(), self.decoders)):
		#         outputs[task] = decoder(rep[i])
		# else:
		#     TrevorRepFunction.n = len(self.decoders)
		#     rep = trevor_rep_function(rep)
		# print("the tasks are loaded in this order")
		for i, (task, decoder) in enumerate(self.task_to_decoder.items()):
			# print("task:", task)
			outputs[task] = decoder(rep)
		# print('task', task, 'decoder', outputs[task].size())

		# print('forward model outputs', outputs)
		return outputs

class ResNet(MTResnet):
	def __init__(self, block, layers, ozan=False, size=1, tasks=None, base_match=512, **kwargs):
		super(ResNet, self).__init__( size=size, base_match=base_match, tasks=tasks, **kwargs)
		if size == 1:
			self.encoder = ResNetEncoder(block, layers, **kwargs)
		elif size == 2:
			self.encoder = ResNetEncoder(block, layers, [96, 192, 384, 720], **kwargs)
		elif size == 3:
			self.encoder = ResNetEncoder(block, layers, [112, 224, 448, 880], **kwargs)
		elif size == 0.5:
			self.encoder = ResNetEncoder(block, layers, [48, 96, 192, 360], **kwargs)
		self.ozan = ozan

class WideResNet(MTResnet):
	def __init__(self, depth, widen_factor=1, drop_rate=0, size=1, **kwargs):
		super(WideResNet, self).__init__( size=size, **kwargs)

		self.encoder = WideResNetEncoder(depth, widen_factor, drop_rate, img_size=kwargs.get("img_size"))

def _resnet(arch, block, layers, pretrained, **kwargs):
	if "wideresnet" in arch:
		model = WideResNet(depth=layers,**kwargs)
	else:
		model = ResNet(block=block, layers=layers, **kwargs)
	if pretrained:
		from torchvision.models import resnet
		if pretrained=="imagenet":
			print("Loading imagenet pretrained weights for architecture {}".format(arch))
			state_dict = load_state_dict_from_url(resnet.model_urls[arch],progress=True)
		else:
			state_dict = torch.load(pretrained).state_dict()
		original_state_dict = model.state_dict()
		state_dict_e = {"encoder.{}".format(k):v for (k,v) in state_dict.items() if "fc" not in k}

		model.load_state_dict({**original_state_dict, **state_dict_e})
	return model


def wideresnet_28_10(pretrained=False, **kwargs):
	"""Constructs a ResNet-18 model.

	Args:
		pretrained (bool): If True, returns a model pre-trained on ImageNet
		progress (bool): If True, displays a progress bar of the download to stderr
	"""
	return _resnet('wideresnet_28_10', None, 28, pretrained, widen_factor=10, base_match=512,
				size=2.5,   **kwargs)


def wideresnet_70_16(pretrained=False, **kwargs):
	"""Constructs a ResNet-18 model.

	Args:
		pretrained (bool): If True, returns a model pre-trained on ImageNet
		progress (bool): If True, displays a progress bar of the download to stderr
	"""
	return _resnet('wideresnet_70_16', None, 70, pretrained, widen_factor=16, base_match=1024,
				   **kwargs)

def resnet18_taskonomy(pretrained=False, **kwargs):
	"""Constructs a ResNet-18 model.

	Args:
		pretrained (bool): If True, returns a model pre-trained on ImageNet
		progress (bool): If True, displays a progress bar of the download to stderr
	"""
	return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, base_match=512,
				   **kwargs)

def resnet34_taskonomy(pretrained=False, **kwargs):
	"""Constructs a ResNet-34 model.

	Args:
		pretrained (bool): If True, returns a model pre-trained on ImageNet
		progress (bool): If True, displays a progress bar of the download to stderr
	"""
	return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained,
				   **kwargs)


def resnet50_taskonomy(pretrained=False, **kwargs):
	"""Constructs a ResNet-50 model.

	Args:
		pretrained (bool): If True, returns a model pre-trained on ImageNet
		progress (bool): If True, displays a progress bar of the download to stderr
	"""
	return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, base_match=2048,
				   **kwargs)


def resnet101_taskonomy(pretrained=False, **kwargs):
	"""Constructs a ResNet-101 model.

	Args:
		pretrained (bool): If True, returns a model pre-trained on ImageNet
		progress (bool): If True, displays a progress bar of the download to stderr
	"""
	return _resnet('resnet101', Bottleneck, [3, 4, 23, 3], pretrained,
				   **kwargs)


def resnet152_taskonomy(pretrained=False, **kwargs):
	"""Constructs a ResNet-152 model.

	Args:
		pretrained (bool): If True, returns a model pre-trained on ImageNet
		progress (bool): If True, displays a progress bar of the download to stderr
	"""
	return _resnet('resnet152', Bottleneck, [3, 8, 36, 3], pretrained, base_match=2048,
				   **kwargs)


def wide_resnet50_2(pretrained=False, progress=True, **kwargs):
	"""Wide ResNet-50-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_
    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
	kwargs['width_per_group'] = 64 * 2
	return _resnet('wide_resnet50_2', Bottleneck, [3, 4, 6, 3],
                   pretrained, base_match=2048, **kwargs)


def wide_resnet101_2(pretrained=False, progress=True, **kwargs):
	"""Wide ResNet-101-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_
    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
	kwargs['width_per_group'] = 64 * 2
	return _resnet('wide_resnet101_2', Bottleneck, [3, 4, 23, 3],
                   pretrained, base_match=2048, **kwargs)
