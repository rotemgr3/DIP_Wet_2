import torch
import torch.nn as nn


def get_model(config):
    if config["architecture"] == "FC":
        return FullyConnectedDenoiser(config["crop_size"], config["hidden_dims"])
    elif config["architecture"] == "UNet":
        return UNetDenoiser(config["crop_size"], config["hidden_dims"])
    else:
        raise ValueError(f"Invalid architecture: {config['architecture']}")


class FullyConnectedDenoiser(nn.Module):
    def __init__(self, crop_size, hidden_dims):
        super(FullyConnectedDenoiser, self).__init__()
        self.input_shape = (3, crop_size, crop_size)
        input_dim = self.input_shape[0] * \
            self.input_shape[1] * self.input_shape[2]
        layers = []
        dims = [input_dim] + hidden_dims
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(dims[-1], input_dim))
        self.fc = nn.Sequential(*layers)

    def forward(self, x):
        # Flatten input
        x = x.view(x.size(0), -1)
        # Forward pass
        x = self.fc(x)
        # Reshape to original shape
        x = x.view(x.size(0), *self.input_shape)
        return x


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(ResBlock, self).__init__()
        padding = kernel_size // 2  # Ensure same padding
        self.conv1 = nn.Conv2d(in_channels, out_channels,
                               kernel_size, stride=1, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               kernel_size, stride=1, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # For residual connection
        self.residual = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, stride=1, bias=False)

    def forward(self, x):
        identity = self.residual(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        return out


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(EncoderBlock, self).__init__()
        self.res_block = ResBlock(in_channels, out_channels, kernel_size)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.res_block(x)
        skip = x  # Save for skip connection
        x = self.pool(x)
        return x, skip


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(DecoderBlock, self).__init__()
        self.upconv = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=2, stride=2)
        self.res_block = ResBlock(in_channels, out_channels, kernel_size)

    def forward(self, x, skip):
        x = self.upconv(x)
        x = torch.cat((x, skip), dim=1)  # Concatenate along channel dimension
        x = self.res_block(x)
        return x


class UNetDenoiser(nn.Module):
    def __init__(self, crop_size, hidden_dims, kernel_size=3):
        super(UNetDenoiser, self).__init__()

        self.input_shape = (3, crop_size, crop_size)
        self.in_channels = self.input_shape[0]
        self.hidden_dims = hidden_dims
        self.kernel_size = kernel_size

        # Encoder
        self.encoders = nn.ModuleList()
        prev_channels = self.in_channels
        for h_dim in hidden_dims:
            self.encoders.append(EncoderBlock(
                prev_channels, h_dim, kernel_size))
            prev_channels = h_dim

        # Bottleneck
        self.bottleneck = ResBlock(
            hidden_dims[-1], hidden_dims[-1] * 2, kernel_size)

        # Decoder
        self.decoders = nn.ModuleList()
        reversed_hidden_dims = hidden_dims[::-1]
        prev_channels = hidden_dims[-1] * 2
        for h_dim in reversed_hidden_dims:
            self.decoders.append(DecoderBlock(
                prev_channels, h_dim, kernel_size))
            prev_channels = h_dim

        # Final convolution
        self.final_conv = nn.Conv2d(
            hidden_dims[0], self.in_channels, kernel_size=1)

    def forward(self, x):
        # Encoder path
        skips = []
        for encoder in self.encoders:
            x, skip = encoder(x)
            skips.append(skip)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder path
        for i, decoder in enumerate(self.decoders):
            x = decoder(x, skips[-(i + 1)])

        # Final output
        x = self.final_conv(x)

        # Reshape to original shape
        x = x.view(x.size(0), *self.input_shape)
        return x
