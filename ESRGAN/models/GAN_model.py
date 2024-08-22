import torch
from torch import nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_act, **kwargs):
        super().__init__()
        self.cnn = nn.Conv2d(in_channels, out_channels, **kwargs, bias=True, )
        self.act = nn.LeakyReLU(0.2, inplace=True) if use_act else nn.Identity() # 判断是否需要activation，如果为否则用Identity原封不动地输出
        
    def forward(self, x):  # 定义forward()方法
        return self.act(self.cnn(x))
    
    
class conv_block(nn.Module):
    """
    Convolutional block with one convolutional layer
    and ReLU activation function. 
    """
    def __init__(self, in_channels, out_channels, kernel_size, padding=1, bias = False):
        super(conv_block,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding, bias=bias),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        return self.conv(x)


class Conv_Trans_Block(nn.Module):
    """
    Transposed Convolutional block with one transposed convolutional 
    layer and ReLU activation function. 
    """
    def __init__(self, in_channels, out_channels, kernel_size):
        super(Conv_Trans_Block, self).__init__()
        self.conv_trans = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=1, bias=False),
            nn.ReLU()
        )

    def forward(self, x):
        return  self.conv_trans(x)

class Multi_Scale_Conv(nn.Module):
  """
  Multi scale convolutional block with 3 convolutional blocks
  with kernel size of 3x3, 5x5 and 7x7. Which is then concatenated
  and fed into a 1x1 convolutional block. 
  """
  def __init__(self, in_channels, out_channels):
        super(Multi_Scale_Conv, self).__init__()
        self.conv3x3 = conv_block(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv5x5 = conv_block(in_channels, out_channels, kernel_size=5, padding=2)
        self.conv7x7 = conv_block(in_channels, out_channels, kernel_size=7, padding=3)
        self.conv1x1 = conv_block(out_channels*3, out_channels, kernel_size=1, padding=0)

  def forward(self, x):
      x1 = self.conv3x3(x)
      x2 = self.conv5x5(x)
      x3 = self.conv7x7(x)
      comb = torch.cat((x1, x2, x3), 1)
      return self.conv1x1(comb)


class Encoder(nn.Module):
    """
    Encoder class of the generator with multiple multi-scale
    convolutional blocks.  
    """
    def __init__(self):
        super(Encoder, self).__init__()

        self.multi_conv1 = Multi_Scale_Conv(1, 32)
        self.conv1 = conv_block(in_channels=32, out_channels=32, kernel_size=3, padding=1)

        self.multi_conv2 = Multi_Scale_Conv(32,64)
        self.conv2 = conv_block(in_channels=64, out_channels=64, kernel_size=3, padding=1)

        self.multi_conv3 = Multi_Scale_Conv(64,128)
        self.conv3 = conv_block(in_channels=128, out_channels=128, kernel_size=3, padding=1)

        self.multi_conv4 = Multi_Scale_Conv(128,256)

    def  forward(self, x): 

        out1 = self.multi_conv1(x)  # 1 - 32 
        out1 = self.conv1(out1)  # 32 - 32

        out2 = self.multi_conv2(out1)  # 32 - 64
        out2 = self.conv2(out2)  # 64 -64 

        out3 = self.multi_conv3(out2)  # 64 - 128
        out3 = self.conv3(out3)  # 128 - 128
         
        out4 = self.multi_conv4(out3)  # 128 - 256
        return out1, out2, out3, out4  # 32，64，128，256
 
class Decoder(nn.Module):
    """
    Decoder class of the generator with multiple transposed
    convolutional blocks.  
    """
    def __init__(self):
        super(Decoder,self).__init__()

        self.conv_trans1 = Conv_Trans_Block(in_channels=256, out_channels=128, kernel_size=3)

        self.conv_trans2 = Conv_Trans_Block(in_channels=128*2, out_channels=128, kernel_size=3)
        self.conv_trans3 = Conv_Trans_Block(in_channels=128, out_channels=64, kernel_size=3)

        self.conv_trans4 = Conv_Trans_Block(in_channels=64*2, out_channels=64, kernel_size=3)
        self.conv_trans5 = Conv_Trans_Block(in_channels=64, out_channels=32, kernel_size=3)

        self.conv_trans6 = Conv_Trans_Block(in_channels=32*2, out_channels=32, kernel_size=3)
        self.conv_trans7 = Conv_Trans_Block(in_channels=32, out_channels=1, kernel_size=3)

    def forward(self, in1, in2, in3, in4):
        out1 = self.conv_trans1(in4)  # 256 - 128

        out2 = torch.cat((out1, in3), 1)  # 128+ 128 -> 256
        out2 = self.conv_trans2(out2)  # 256 - 128
        out2 = self.conv_trans3(out2)  # 128 - 64
        
        out3 = torch.cat((out2, in2), 1)  # 64 + 64 -> 128
        out3 = self.conv_trans4(out3)  # 128 - 64
        out3 = self.conv_trans5(out3)  # 64 - 32

        out4 = torch.cat((out3, in1), 1)  # 32 + 32 -> 64
        out4 = self.conv_trans6(out4)  # 64 - 32
        out4 = self.conv_trans7(out4)  # 

        return out4

class Generator(nn.Module):
    """
    Generator model proposed by the model, which takes in frame as a input,
    along the features learned by forward and backward ConvLSTMs and generates 
    a motion artefact free image. Here, skip connections are employed between 
    the encoder and decoder.   
    """
    def __init__(self):
        super(Generator,self).__init__()

        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        out1, out2, out3, out4 = self.encoder(x)
        return self.decoder(out1, out2, out3, out4)


class Discriminator(nn.Module):
    def __init__(self, in_channels=1, features=None):
        if features is None:
            features = [64,64,128,128,256,256,512,512]
        super().__init__()
        blocks = []
        for idx, feature in enumerate(features):
            blocks.append(
                ConvBlock(
                    in_channels,
                    feature,
                    kernel_size=3,
                    stride=1+idx%2,
                    padding=1,
                    use_act=True,
                ),
            )
            in_channels = feature # why stride change??
        self.blocks = nn.Sequential(*blocks)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((6,6)),  # 指定希望的输出尺寸
            nn.Flatten(),
            nn.Linear(512*6*6, 512),
            nn.LeakyReLU(0.2, True),
            nn.Linear(512, 1),
        )
    def forward(self, x):
        x = self.blocks(x)
        return self.classifier(x)

def initialize_weights(model, scale=1):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight.data)
            m.weight.data *= scale # 较小的初始化 适用于训练较深的网络


# use demo
if __name__ == "__main__":
    x = input = torch.randn(1, 1, 128, 128)
    inv_input = torch.flip(input, dims=[1])
    gen = Generator()
    v = gen(x)
    print(v)
    disc = Discriminator()
    d = disc(x)
    print(d)
    
    




