import torch
import torch.nn as nn

from spd_learn.modules import BiMap, CovLayer, LogEig, ReEig, SPDBatchNormMeanVar

from activation_test.activation.elementwise import coshP, expT


class TSMNetCustom(nn.Module):
    """TSMNet with a configurable SPD activation (ReEig, coshP or expT).

    Architecture:
        CNN (temporal + spatiotemporal conv)
        -> CovPool
        -> BiMap
        -> Activation  [ReEig | coshP | expT]
        -> SPDBatchNormMeanVar
        -> LogEig
        -> Linear

    Parameters
    ----------
    activation : str
        One of "reeig", "coshP", "expT".
    n_chans : int
        Number of EEG channels (input).
    n_temp_filters : int
        Temporal conv filters.
    temp_kernel_length : int
        Temporal conv kernel size.
    n_spatiotemp_filters : int
        Spatiotemporal conv filters.
    n_bimap_filters : int
        BiMap output dimension.
    threshold : float
        Threshold.
    n_outputs : int
        Number of ???.
    """

    def __init__(
        self,
        activation="reeig",
        n_chans=None,
        n_temp_filters=4,
        temp_kernel_length=25,
        n_spatiotemp_filters=40,
        n_bimap_filters=20,
        threshold=1e-4,
        n_outputs=None,
    ):
        super().__init__()

        if n_chans is None:
            raise ValueError("n_chans must be provided")
        if n_outputs is None:
            raise ValueError("n_outputs must be provided")

        self.n_chans = n_chans
        self.n_outputs = n_outputs
        self.n_temp_filters = n_temp_filters
        self.n_temp_kernel = temp_kernel_length
        self.n_spatiotemp_filters = n_spatiotemp_filters
        self.n_bimap_filters = n_bimap_filters
        self.activation_type = activation
        self.threshold = threshold

        n_tangent_dim = int(n_bimap_filters * (n_bimap_filters + 1) / 2)

        self.cnn = nn.Sequential(
            nn.Conv2d(
                1,
                n_temp_filters,
                kernel_size=(1, temp_kernel_length),
                padding="same",
                padding_mode="reflect",
            ),
            nn.Conv2d(n_temp_filters, n_spatiotemp_filters, (n_chans, 1)),
            nn.Flatten(start_dim=2),
        )
        self.covpool = CovLayer()
        self.bimap = BiMap(in_features=n_spatiotemp_filters, out_features=n_bimap_filters)
        self.activation = self._make_activation(threshold)
        self.spdbnorm = SPDBatchNormMeanVar(
            n_bimap_filters,
            affine=True,
            bias_requires_grad=False,
            weight_requires_grad=True,
        )
        self.logeig = nn.Sequential(LogEig(), nn.Flatten(start_dim=1))
        self.head = nn.Linear(n_tangent_dim, n_outputs)

    def _make_activation(self, threshold):
        if self.activation_type == "reeig":
            return ReEig(threshold=threshold)
        elif self.activation_type == "coshP":
            return coshP()
        elif self.activation_type == "expT":
            return expT()
        else:
            raise ValueError(f"Unknown activation '{self.activation_type}'. Choose from: reeig, coshP, expT")

    def forward(self, x: torch.Tensor, domain=None) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : (batch_size, n_chans, n_times)  raw EEG epochs
            ??? pas sur de ce que je fais domain : ignored, kept for compatibility with test_stat.py

        Returns
        ----------
        torch.Tensor
            Output tensor of shape (batch_size, n_outputs)
        """
        x_filtered = self.cnn(x[:, None, ...])
        x_cov = self.covpool(x_filtered)
        # === Change from spdnet to BiMap + Activation Functions ===
        x_bimap = self.bimap(x_cov)
        x_activated = self.activation(x_bimap)
        # ==========================================================
        # Juste avant self.spdbnorm(x_activated)
        eigs = torch.linalg.eigvalsh(x_activated)
        if torch.any(eigs <= 0):
            print(f"ALERTE : Matrice non-SPD détectée ! Min Eig: {eigs.min().item()}")
        x_tangent = self.logeig(self.spdbnorm(x_activated))
        return self.head(x_tangent)
