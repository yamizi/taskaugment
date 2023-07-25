"""
Adapted from https://gist.github.com/DerThorsten/7117b9b7a41da4e0a13d6500f9a1b657
"""
import math
import torch
import numpy
from torch import nn


class GaborFilters(nn.Module):
    def __init__(self, 
        in_channels, 
        n_sigmas = 3,
        n_lambdas = 4,
        n_gammas = 1,
        n_thetas = 7,
        kernel_radius=15,
        rotation_invariant=True
    ):
        super().__init__()
        self.in_channels = in_channels
        kernel_size = kernel_radius*2 + 1
        self.kernel_size = kernel_size
        self.kernel_radius = kernel_radius
        self.n_thetas = n_thetas
        self.rotation_invariant = rotation_invariant
        def make_param(in_channels, values, requires_grad=True, dtype=None):
            if dtype is None:
                dtype = 'float32'
            values = numpy.require(values, dtype=dtype)
            n = in_channels * len(values)
            data=torch.from_numpy(values).view(1,-1)
            data = data.repeat(in_channels, 1)
            return torch.nn.Parameter(data=data, requires_grad=requires_grad)


        # build all learnable parameters
        self.sigmas = make_param(in_channels, 2**numpy.arange(n_sigmas)*2)
        self.lambdas = make_param(in_channels, 2**numpy.arange(n_lambdas)*4.0)
        self.gammas = make_param(in_channels, numpy.ones(n_gammas)*0.5)
        self.psis = make_param(in_channels, numpy.array([0, math.pi/2.0]))

        print(len(self.sigmas))


        thetas = numpy.linspace(0.0, 2.0*math.pi, num=n_thetas, endpoint=False)
        thetas = torch.from_numpy(thetas).float()
        self.register_buffer('thetas', thetas)

        indices = torch.arange(kernel_size, dtype=torch.float32) -  (kernel_size - 1)/2
        self.register_buffer('indices', indices)


        # number of channels after the conv
        self._n_channels_post_conv = self.in_channels * self.sigmas.shape[1] * \
                                     self.lambdas.shape[1] * self.gammas.shape[1] * \
                                     self.psis.shape[1] * self.thetas.shape[0] 


    def make_gabor_filters(self):

        sigmas=self.sigmas
        lambdas=self.lambdas
        gammas=self.gammas
        psis=self.psis
        thetas=self.thetas
        y=self.indices
        x=self.indices

        in_channels = sigmas.shape[0]
        assert in_channels == lambdas.shape[0]
        assert in_channels == gammas.shape[0]

        kernel_size = y.shape[0], x.shape[0]



        sigmas  = sigmas.view (in_channels, sigmas.shape[1],1, 1, 1, 1, 1, 1)
        lambdas = lambdas.view(in_channels, 1, lambdas.shape[1],1, 1, 1, 1, 1)
        gammas  = gammas.view (in_channels, 1, 1, gammas.shape[1], 1, 1, 1, 1)
        psis    = psis.view (in_channels, 1, 1, 1, psis.shape[1], 1, 1, 1)

        thetas  = thetas.view(1,1, 1, 1, 1, thetas.shape[0], 1, 1)
        y       = y.view(1,1, 1, 1, 1, 1, y.shape[0], 1)
        x       = x.view(1,1, 1, 1, 1, 1, 1, x.shape[0])

        sigma_x = sigmas
        sigma_y = sigmas / gammas

        sin_t = torch.sin(thetas)
        cos_t = torch.cos(thetas)
        y_theta = -x * sin_t + y * cos_t
        x_theta =  x * cos_t + y * sin_t
        


        gb = torch.exp(-.5 * (x_theta ** 2 / sigma_x ** 2 + y_theta ** 2 / sigma_y ** 2)) \
             * torch.cos(2.0 * math.pi  * x_theta / lambdas + psis)

        gb = gb.view(-1,kernel_size[0], kernel_size[1])

        return gb


    def forward(self, x):
        batch_size = x.size(0)
        sy = x.size(2)
        sx = x.size(3)  
        gb = self.make_gabor_filters()

        assert gb.shape[0] == self._n_channels_post_conv
        assert gb.shape[1] == self.kernel_size
        assert gb.shape[2] == self.kernel_size
        gb = gb.view(self._n_channels_post_conv,1,self.kernel_size,self.kernel_size)

        res = nn.functional.conv2d(input=x, weight=gb,
            padding=self.kernel_radius, groups=self.in_channels)
       
        
        if self.rotation_invariant:
            res = res.view(batch_size, self.in_channels, -1, self.n_thetas,sy, sx)
            res,_ = res.max(dim=3)

        res = res.view(batch_size, -1,sy, sx)


        return res

if __name__ == "__main__":
    import pylab
    import skimage.data
    astronaut = skimage.data.astronaut()
    #astronaut[...,0] = astronaut[...,0].T
    astronaut = numpy.moveaxis(astronaut,-1,0)[None,...]
    astronaut = torch.from_numpy(astronaut).float()


    gb = GaborFilters(in_channels=3, n_sigmas=1).to("cuda:0")
    res = gb(astronaut.to("cuda:0")).cpu()
    print(res.shape)



    for c in range(res.size(1)):
        img = res[0,c,...]
        img = img.detach().numpy()
        fig = pylab.imshow(img)
        pylab.show()

    print("end")