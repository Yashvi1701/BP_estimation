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