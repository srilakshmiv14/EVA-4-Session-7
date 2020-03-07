# EVA-4-Session-7

Building a CNN Model with CIFAR10 dataset

Image Classifier

Architecture is of 4 Blocks with 3 max pooling as Transition layers.

 def forward(self, x):
      x=self.convblock1(x)
      x=self.transblock1(x)
      x=self.convblock2(x)
      x=self.transblock2(x)
      x=self.convblock3(x)
      x=self.transblock3(x)
      x=self.convblock4(x)
      x=x.view(-1,10)
      return x
model = Net()

In one of the layer, Conv2 block using depthwise seperable convolution and Conv3 block using dilation.

Dilated Convolution :
We “observe” a large receptive filed without adding additional costs. Because of that, dilated convolution is used to cheaply increase the receptive field of output units without increasing the kernel size,

The dilated convolution follows:
 

When l = 1, the dilated convolution becomes as the standard convolution.
Intuitively, dilated convolutions “inflate” the kernel by inserting spaces between the kernel elements. This additional parameter l (dilation rate) indicates how much we want to widen the kernel. Implementations may vary, but there are usually l-1 spaces inserted between kernel elements. 

Depthwise Seperable convolution:

The depth wise separable convolutions consist of two steps: depthwise convolutions and 1x1 convolutions.
First, we apply depthwise convolution to the input layer. Instead of using a single filter of size 3 x 3 x 3 in 2D convolution, we used 3 kernels, separately. Each filter has size 3 x 3 x 1. Each kernel convolves with 1 channel of the input layer (1 channel only, not all channels!). Each of such convolution provides a map of size 5 x 5 x 1. We then stack these maps together to create a 5 x 5 x 3 image. After this, we have the output with size 5 x 5 x 3. We now shrink the spatial dimensions, but the depth is still the same as before.
As the second step of depthwise separable convolution, to extend the depth, we apply the 1x1 convolution with kernel size 1x1x3. Convolving the 5 x 5 x 3 input image with each 1 x 1 x 3 kernel provides a map of size 5 x 5 x 1.Thus, after applying 128 1x1 convolutions, we can have a layer with size 5 x 5 x 128.

With these two steps, depthwise separable convolution also transform the input layer (7 x 7 x 3) into the output layer (5 x 5 x 128).

here we using L1 with Cross entropy criterion and L2 regularization with lr = 0.001 and momentum =0.9. 

Achieved accuracy of 80.07% in 40 epochs.

