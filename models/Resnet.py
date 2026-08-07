import torch
import torch.nn as nn


# -----------------------------
# Residual Block for 1D signals
# -----------------------------

class ResidualBlock1D(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1):

        super().__init__()

        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1
        )

        self.bn1 = nn.BatchNorm1d(out_channels)

        self.relu = nn.ReLU()

        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1
        )

        self.bn2 = nn.BatchNorm1d(out_channels)


        # Skip connection
        # Needed when dimensions change

        if in_channels != out_channels or stride != 1:

            self.shortcut = nn.Sequential(

                nn.Conv1d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride
                ),

                nn.BatchNorm1d(out_channels)

            )

        else:

            self.shortcut = nn.Identity()



    def forward(self,x):

        identity = self.shortcut(x)


        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)


        out = self.conv2(out)
        out = self.bn2(out)


        out += identity

        out = self.relu(out)


        return out



# -----------------------------
# ResNet1D for BP estimation
# -----------------------------

class ResNet1D(nn.Module):

    def __init__(self):

        super().__init__()


        self.input_layer = nn.Sequential(

            nn.Conv1d(
                in_channels=1,
                out_channels=32,
                kernel_size=7,
                stride=2,
                padding=3
            ),

            nn.BatchNorm1d(32),

            nn.ReLU(),

            nn.MaxPool1d(3, stride=2, padding=1)

        )



        self.res_blocks = nn.Sequential(


            # 32 channels
            ResidualBlock1D(
                32,
                32
            ),


            ResidualBlock1D(
                32,
                64,
                stride=2
            ),


            ResidualBlock1D(
                64,
                64
            ),


            ResidualBlock1D(
                64,
                128,
                stride=2
            ),


            ResidualBlock1D(
                128,
                128
            )

        )



        self.pool = nn.AdaptiveAvgPool1d(1)



        self.regressor = nn.Sequential(

            nn.Flatten(),

            nn.Linear(
                128,
                64
            ),

            nn.ReLU(),

            nn.Dropout(0.3),


            nn.Linear(
                64,
                2
            )

        )



    def forward(self,x):

        x = self.input_layer(x)

        x = self.res_blocks(x)

        x = self.pool(x)

        x = self.regressor(x)

        return x



# -----------------------------------------
# Basic Residual Block
# -----------------------------------------

class BasicBlock(nn.Module):

    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):

        super().__init__()

        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False
        )

        self.bn1 = nn.BatchNorm1d(out_channels)

        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )

        self.bn2 = nn.BatchNorm1d(out_channels)

        self.downsample = None

        if stride != 1 or in_channels != out_channels:

            self.downsample = nn.Sequential(

                nn.Conv1d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),

                nn.BatchNorm1d(out_channels)

            )

    def forward(self, x):

        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(identity)

        out += identity
        out = self.relu(out)

        return out


# -----------------------------------------
# ResNet18
# -----------------------------------------

class ResNet18_1D(nn.Module):

    def __init__(self, num_outputs=2):

        super().__init__()

        self.in_channels = 64

        # Initial convolution

        self.conv1 = nn.Conv1d(
            1,
            64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )

        self.bn1 = nn.BatchNorm1d(64)

        self.relu = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool1d(
            kernel_size=3,
            stride=2,
            padding=1
        )

        # ResNet18 layers

        self.layer1 = self._make_layer(64, 2, stride=1)

        self.layer2 = self._make_layer(128, 2, stride=2)

        self.layer3 = self._make_layer(256, 2, stride=2)

        self.layer4 = self._make_layer(512, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool1d(1)

        self.dropout = nn.Dropout(0.3)

        self.fc = nn.Linear(512, num_outputs)

    def _make_layer(self, out_channels, blocks, stride):

        layers = []

        layers.append(
            BasicBlock(
                self.in_channels,
                out_channels,
                stride
            )
        )

        self.in_channels = out_channels

        for _ in range(1, blocks):

            layers.append(
                BasicBlock(
                    self.in_channels,
                    out_channels
                )
            )

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

        x = self.avgpool(x)

        x = x.squeeze(-1)

        x = self.dropout(x)

        x = self.fc(x)

        return x



# Adding Squeeze-and-Excitation (SE) block for channel-wise attention
class SEBlock(nn.Module):

    def __init__(self, channels, reduction=16):

        super().__init__()

        self.pool = nn.AdaptiveAvgPool1d(1)

        self.fc = nn.Sequential(

            nn.Linear(channels, channels // reduction),

            nn.ReLU(),

            nn.Linear(channels // reduction, channels),

            nn.Sigmoid()

        )

    def forward(self, x):

        b, c, _ = x.size()

        y = self.pool(x).view(b, c)

        y = self.fc(y).view(b, c, 1)

        return x * y


# -------------------------------------------------
# Residual Block with SE Attention
# -------------------------------------------------

class SEResidualBlock1D(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1):

        super().__init__()

        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False
        )

        self.bn1 = nn.BatchNorm1d(out_channels)

        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=False
        )

        self.bn2 = nn.BatchNorm1d(out_channels)

        # -------- SE Attention --------

        self.se = SEBlock(out_channels)

        # -------- Skip Connection --------

        if in_channels != out_channels or stride != 1:

            self.shortcut = nn.Sequential(

                nn.Conv1d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),

                nn.BatchNorm1d(out_channels)

            )

        else:

            self.shortcut = nn.Identity()

    def forward(self, x):

        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # -------- Apply SE Attention --------

        out = self.se(out)

        # -------- Residual Connection --------

        out += identity

        out = self.relu(out)

        return out


# =====================================================
# SE-ResNet1D
# =====================================================

class SEResNet1D(nn.Module):

    def __init__(self):

        super().__init__()

        # Input Stem
        self.input_layer = nn.Sequential(

            nn.Conv1d(
                in_channels=1,
                out_channels=32,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False
            ),

            nn.BatchNorm1d(32),

            nn.ReLU(inplace=True),

            nn.MaxPool1d(
                kernel_size=3,
                stride=2,
                padding=1
            )

        )

        # Residual Stages
        self.res_blocks = nn.Sequential(

            # Stage 1
            SEResidualBlock1D(
                32,
                32
            ),

            # Stage 2
            SEResidualBlock1D(
                32,
                64,
                stride=2
            ),

            SEResidualBlock1D(
                64,
                64
            ),

            # Stage 3
            SEResidualBlock1D(
                64,
                128,
                stride=2
            ),

            SEResidualBlock1D(
                128,
                128
            )

        )

        # Global Average Pooling
        self.pool = nn.AdaptiveAvgPool1d(1)

        # Regression Head
        self.regressor = nn.Sequential(

            nn.Flatten(),

            nn.Linear(
                128,
                64
            ),

            nn.ReLU(inplace=True),

            nn.Dropout(0.3),

            nn.Linear(
                64,
                2
            )

        )

    def forward(self, x):

        x = self.input_layer(x)

        x = self.res_blocks(x)

        x = self.pool(x)

        x = self.regressor(x)

        return x



# =====================================================
# SE-ResNet18 1D
# =====================================================

class SEResNet18_1D(nn.Module):

    def __init__(self):

        super().__init__()


        self.in_channels = 64


        # Initial convolution

        self.conv1 = nn.Conv1d(
            1,
            64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )


        self.bn1 = nn.BatchNorm1d(64)

        self.relu = nn.ReLU(inplace=True)


        self.maxpool = nn.MaxPool1d(
            kernel_size=3,
            stride=2,
            padding=1
        )


        # ResNet18 stages

        self.layer1 = self.make_layer(
            64,
            2,
            stride=1
        )


        self.layer2 = self.make_layer(
            128,
            2,
            stride=2
        )


        self.layer3 = self.make_layer(
            256,
            2,
            stride=2
        )


        self.layer4 = self.make_layer(
            512,
            2,
            stride=2
        )


        self.avgpool = nn.AdaptiveAvgPool1d(1)


        self.fc = nn.Sequential(

            nn.Dropout(0.3),

            nn.Linear(
                512,
                2
            )

        )



    def make_layer(self, channels, blocks, stride):

        layers = []


        layers.append(

            SEResidualBlock1D(
                self.in_channels,
                channels,
                stride
            )

        )


        self.in_channels = channels


        for _ in range(1, blocks):

            layers.append(

                SEResidualBlock1D(
                    channels,
                    channels
                )

            )


        return nn.Sequential(*layers)



    def forward(self,x):

        x = self.conv1(x)

        x = self.bn1(x)

        x = self.relu(x)

        x = self.maxpool(x)


        x = self.layer1(x)

        x = self.layer2(x)

        x = self.layer3(x)

        x = self.layer4(x)


        x = self.avgpool(x)

        x = torch.flatten(x,1)


        x = self.fc(x)


        return x