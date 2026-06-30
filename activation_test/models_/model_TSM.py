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
    n_outputs : int
        Number of classes.
    n_temp_filters : int
        Temporal conv filters.
    temp_kernel_length : int
        Temporal conv kernel size.
    n_spatiotemp_filters : int
        Spatiotemporal conv filters.
    n_bimap_filters : int
        BiMap output dimension.
    threshold : float
        Threshold for ReEig (ignored for other activations).
    """

    def __init__(
        self,
        activation="reeig",
        n_chans=None,
        n_outputs=None,
        n_temp_filters=4,
        temp_kernel_length=25,
        n_spatiotemp_filters=40,
        n_bimap_filters=20,
        threshold=1e-4,
    ):
        super().__init__()

        if n_chans is None:
            raise ValueError("n_chans must be provided")
        if n_outputs is None:
            raise ValueError("n_outputs must be provided")

        self.activation_type = activation
        self.threshold = threshold
        n_tangent_dim = n_bimap_filters * (n_bimap_filters + 1) // 2

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
        x : (batch, n_chans, n_times)  raw EEG epochs
        domain : ignored, kept for compatibility with test_stat.py

        Note on SPDBatchNorm
        --------------------
        SPDBatchNormMeanVar computes the Riemannian Frechet mean via iterative
        eigh calls. It is designed for ReEig outputs (bounded condition number).
        Element-wise activations (coshP, expT) apply cosh/Taylor element-wise
        to matrix entries: the result is symmetric but NOT SPD in general, and
        has condition numbers >> 1e6 that make frechet_mean diverge.
        For these activations we skip SPDBatchNorm, matching how modelSPDNet
        uses coshP/expT (BiMap -> activation -> LogEig, no batch norm).
        """
        x = self.cnn(x[:, None, ...])
        x = self.covpool(x)
        x = self.bimap(x)
        if self.activation_type in ("coshP", "expT"):
            # Trace-normalize before element-wise activation: bring diagonal
            # entries to scale ~1 so cosh/exp don't overflow (BiMap entries
            # can reach ~50, giving cosh(25) ~ 1e10 which breaks eigh in LogEig).
            n = x.shape[-1]
            trace = x.diagonal(dim1=-2, dim2=-1).sum(dim=-1, keepdim=True).unsqueeze(-1)
            x = x * n / trace.clamp(min=1e-8)
        x = self.activation(x)
        if self.activation_type == "reeig":
            x = self.spdbnorm(x)
        x = self.logeig(x)
        return self.head(x)
