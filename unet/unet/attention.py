import torch.nn as nn


class AttentionGate(nn.Module):
    """
    Attention Gate module.

    This gate learns attention coefficients to suppress irrelevant background features
    and highlight crack-related regions in skip connections.

    Args:
        F_g: Number of channels for upsampling features.
        F_l: Number of channels for skip-connection features.
        F_int: Number of channels for intermediate features (dimensionality reduction).
    """

    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        # Convolution layer for processing upsampling features
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1),  # 1x1 conv for channel reduction
            nn.BatchNorm2d(F_int),
        )
        # Convolution layer for processing skip-connection features
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1),  # 1x1 conv for channel reduction
            nn.BatchNorm2d(F_int),
        )
        # Convolution layer for generating the attention map
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1),  # Output a single-channel attention map
            nn.BatchNorm2d(1),
            nn.Sigmoid(),  # Limit values to [0, 1]
        )
        # Activation function
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        """
        Compute attention coefficients and apply them to the skip features.

        Args:
            g: Upsampled features from the decoder.
            x: Skip-connection features from the encoder.

        Returns:
            The reweighted skip-connection features.
        """
        # Dimensionality reduction
        g1 = self.W_g(g)
        x1 = self.W_x(x)

        # Feature fusion and activation
        psi = self.relu(g1 + x1)

        # Generate attention coefficients in [0, 1]
        psi = self.psi(psi)

        # Apply attention to the original skip features
        return x * psi
