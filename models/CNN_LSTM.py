import torch
import torch.nn as nn


class CNN_LSTM(nn.Module):

    def __init__(self):

        super().__init__()


        # -------------------------
        # CNN Feature Extractor
        # -------------------------

        self.features = nn.Sequential(

            nn.Conv1d(
                in_channels=1,
                out_channels=32,
                kernel_size=5,
                padding=2
            ),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.2),


            nn.Conv1d(
                in_channels=32,
                out_channels=64,
                kernel_size=5,
                padding=2
            ),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.2),


            nn.Conv1d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)

        )


        # -------------------------
        # LSTM
        # -------------------------

        self.lstm = nn.LSTM(

            input_size=128,

            hidden_size=64,

            num_layers=2,

            batch_first=True,

            dropout=0.3

        )


        # -------------------------
        # Regression Head
        # -------------------------

        self.regressor = nn.Sequential(

            nn.Linear(64,64),

            nn.ReLU(),

            nn.Dropout(0.3),

            nn.Linear(64,32),

            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(32,2)
            

        )


        # Final BP prediction layer

        self.output_layer = nn.Linear(32,2)


    def forward(self,x):

        # -------------------------
        # CNN
        # -------------------------

        x = self.features(x)

        # (batch,128,250)


        # -------------------------
        # Prepare for LSTM
        # -------------------------

        x = x.permute(0,2,1)

        # (batch,250,128)


        # -------------------------
        # LSTM
        # -------------------------

        output, (hidden, cell) = self.lstm(x)


        # Take last hidden state

        x = output[:, -1, :]

        # (batch,64)


        # -------------------------
        # Regression
        # -------------------------

        x = self.regressor(x)

        x = self.output_layer(x)


        return x