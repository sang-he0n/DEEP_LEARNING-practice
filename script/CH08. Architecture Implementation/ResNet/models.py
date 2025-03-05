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
        CON = (stride != 1) or (in_planes != planes)
        if CON :
            self.ds_block = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=in_planes, out_channels=planes, kernel_size=1, stride=stride, bias=False),
                torch.nn.BatchNorm2d(num_features=planes)
            )
        else : 
            self.ds_block = torch.nn.Sequential()
    def forward(self, x:torch.Tensor) -> torch.Tensor :
        identity = x 
        x = self.conv_block(x)   
        identity = self.ds_block(identity)
        x += identity
        x = torch.nn.functional.relu(x)
        return x

#(2) Define `ResNet` class
class ResNet(torch.nn.Module) :
    '''
    
    '''
    def __init__(self, blocks_list:list, class_num:int=10) :
        super().__init__()
        self.in_planes = 64
        self.conv_block = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, kernel_size=3,stride=1, padding=1, bias=False),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU()
        )
        self.res_block_01 = self._make_res_block(planes=64, blocks_num=blocks_list[0], stride=1)
        self.res_block_02 = self._make_res_block(planes=128, blocks_num=blocks_list[1], stride=2)
        self.res_block_03 = self._make_res_block(planes=256, blocks_num=blocks_list[2], stride=2)
        self.res_block_04 = self._make_res_block(planes=512, blocks_num=blocks_list[3], stride=2)
        self.global_pool = torch.nn.AvgPool2d(kernel_size=4)
        self.fc_block = torch.nn.Linear(in_features=512, out_features=class_num)
    def _make_res_block(self, planes:int, blocks_num:int, stride:int) -> torch.nn.Sequential :
        strides = [stride]+[1]*(blocks_num-1)
        blocks = []
        for stride in strides :
            blocks.append(ResidualBlock(in_planes=self.in_planes, planes=planes, stride=stride))
            self.in_planes = planes
        blocks = torch.nn.Sequential(*blocks)
        return blocks
    def forward(self, x:torch.Tensor) -> torch.Tensor :
        x = self.conv_block(x)
        x = self.res_block_01(x)
        x = self.res_block_02(x)
        x = self.res_block_03(x)
        x = self.res_block_04(x)
        x = self.global_pool(x)
        x = x.reshape(shape=(x.shape[0], -1))
        x = self.fc_block(x)
        return x

#(3) Define `check_model()` function
def check_model(model_nm:str) -> torch.nn.Module :
    '''
    
    '''
    if model_nm == 'resnet18' :
        model = ResNet(blocks_list=[2, 2, 2, 2])
    elif model_nm == 'resnet34' :
        model = ResNet(blocks_list=[3, 4, 6, 3])
    return model

