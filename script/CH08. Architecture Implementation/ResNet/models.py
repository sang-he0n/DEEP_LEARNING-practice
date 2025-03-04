# 00. 
import torch

# 01. 
#(1) Define `ResidualBlock` class
class ResidualBlock(torch.nn.Module) :
    '''

    '''
    def __init__(self, in_planes:int, planes:int, stride:int=1) :
        super().__init__()
        self.conv_block = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=in_planes, out_channels=planes, kernel_size=3, stride=stride, padding=1, bias=False),
            torch.nn.BatchNorm2d(num_features=planes),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=planes, out_channels=planes, kernel_size=3, stride=1, padding=1, bias=False),
            torch.nn.BatchNorm2d(num_features=planes)
        )
        self.downsample = torch.nn.Sequential()
        CON = (
            (stride != 1) or 
            (in_planes != planes)
        )
        if CON :
            self.downsample = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=in_planes, out_channels=planes, kernel_size=1, stride=stride, bias=False),
                torch.nn.BatchNorm2d(num_features=planes)
            )
    def forward(self, x:torch.Tensor) -> torch.Tensor :
        identity = x
        out = self.conv_block(x)   
        identity = self.downsample(identity)
        out += identity
        out = torch.nn.functional.relu(out)
        return out

#(2) Define `ResNet` class
class ResNet(torch.nn.Module) :
    '''
    
    '''
    def __init__(self, block:torch.nn.Module, blocks_num_list:list, num_classes:int=10) :
        super().__init__()
        self.in_planes = 64
        self.base = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, kernel_size=3,stride=1, padding=1, bias=False),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU()
        )
        self.resblock_01 = self._make_layer(block=block, planes=64, blocks_num=blocks_num_list[0], stride=1)
        self.resblock_02 = self._make_layer(block=block, planes=128, blocks_num=blocks_num_list[1], stride=2)
        self.resblock_03 = self._make_layer(block=block, planes=256, blocks_num=blocks_num_list[2], stride=2)
        self.resblock_04 = self._make_layer(block=block, planes=512, blocks_num=blocks_num_list[3], stride=2)
        self.global_pool = torch.nn.AvgPool2d(kernel_size=4)
        self.fc_block = torch.nn.Linear(in_features=512, out_features=num_classes)
    def _make_layer(self, block:torch.nn.Module, planes:int, blocks_num:int, stride:int) -> torch.nn.Sequential :
        strides = [stride]+[1]*(blocks_num-1)
        layers = []
        for stride in strides :
            layers.append(block(in_planes=self.in_planes, planes=planes, stride=stride))
            self.in_planes = planes
        layers = torch.nn.Sequential(*layers)
        return layers
    def forward(self, x:torch.Tensor) -> torch.Tensor :
        x = self.base(x)
        x = self.resblock_01(x)
        x = self.resblock_02(x)
        x = self.resblock_03(x)
        x = self.resblock_04(x)
        x = self.global_pool(x)
        x = x.reshape(shape=(x.size(dim=0), -1))
        x = self.fc_block(x)
        return x

#(3) Define `check_model()` function
def check_model(model_nm:str) -> torch.nn.Module :
    '''
    
    '''
    if model_nm == 'resnet18' :
        model = ResNet(block=ResidualBlock, blocks_num_list=[2, 2, 2, 2])
    elif model_nm == 'resnet34' :
        model = ResNet(block=ResidualBlock, blocks_num_list=[3, 4, 6, 3])
    return model

