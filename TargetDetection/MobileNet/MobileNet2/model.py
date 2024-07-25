import torch
from torch import nn




#ch卷积核个数，即输出特征的通道数
#将卷积核个数调整为距离divisor最近的整数倍的数值，便于多gpu运算
def _make_divisible(ch,divisor=8,min_ch=None):
    if min_ch is None:
        min_ch=divisor
    #加上一个divisor/2相当于做四舍五入
    new_ch=max(min_ch,int(ch+divisor/2)//divisor*divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_ch<0.9*ch:
        new_ch+=divisor

    return new_ch



class ConvBNRelu(nn.Sequential):
    def __init__(self,inChannel,outChannel,keynelSize=3,stride=1,group=1):
        padding=(keynelSize-1)//2
        super(ConvBNRelu,self).__init__(
            nn.Conv2d(inChannel,outChannel,keynelSize,stride,padding,groups=group,bias=False),
            nn.BatchNorm2d(outChannel),
            #inplace=True 参数用于指示是否原地（inplace）地修改输入张量，而不是返回一个新的张量副本。
            nn.ReLU6(inplace=True)
        )


class InvertedResidual(nn.Module):
    def __init__(self,inChannel,outChannel,stride,expandRatio):
        super(InvertedResidual,self).__init__()
        hiddenChannel=inChannel*expandRatio
        self.useShortcut=stride==1 and inChannel==outChannel

        layers=[]
        if expandRatio !=1:
            layers.append(ConvBNRelu(inChannel,hiddenChannel,keynelSize=1))
        #extend和append效果一样，支持批量插入
        layers.extend([
            #3*3  depthwise conv，group如果等于1就是普通卷积，如果等于输入通道数的话，就是dw卷积
            ConvBNRelu(hiddenChannel,hiddenChannel,stride=stride,group=hiddenChannel),
            #1*1 pointwise conv,使用linear激活函数，即y=x激活函数
            nn.Conv2d(hiddenChannel,outChannel,kernel_size=1,bias=False),
            nn.BatchNorm2d(outChannel),])

        self.conv=nn.Sequential(*layers)

    def forward(self,x):
        if self.useShortcut:
            return x+self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    #alpha用于控制卷积层使用卷积核个数的倍率
    def __init__(self,numClasses=1000,alpha=1.0,roundNearest=8):
        super(MobileNetV2,self).__init__()
        block=InvertedResidual
        #将卷积核个数调整到8的整数倍
        inputChannel=_make_divisible(32*alpha,roundNearest)
        lastChannel=_make_divisible(1280*alpha,roundNearest)


        invertedResidualSetting=[
            # t(扩展因子), c（输出特征通道）, n（bottleneck重复次数）, s（步距）
            [1, 16, 1, 1],#第一个bottleneck对应对的t，c，n，s
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]
        features=[]
        features.append(ConvBNRelu(3,inputChannel,stride=2))

        for t,c,n,s in invertedResidualSetting:
            outputChannel=_make_divisible(c*alpha,roundNearest)
            for i in range(n):
                stride=s if i==0 else 1
                features.append(block(inputChannel,outputChannel,stride,expandRatio=t))
                inputChannel=outputChannel

        #构建最后几层
        features.append(ConvBNRelu(inputChannel,lastChannel,1))

        self.features=nn.Sequential(*features)

        #构建classifier
        self.avgpool=nn.AdaptiveAvgPool2d((1,1))
        self.classifier=nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(lastChannel,numClasses)
        )

        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,mode='fan_out')
                if m.bias is not  None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m,nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m,nn.Linear):
                nn.init.normal_(m.weight,0,0.01)
                nn.init.zeros_(m.bias)

    def forward(self,x):
        x=self.features(x)
        x=self.avgpool(x)
        x=torch.flatten(x,1)
        x=self.classifier(x)
        return x

