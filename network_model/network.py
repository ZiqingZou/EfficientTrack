import torch.nn as nn


"""
    Network architecture of ErrorTrack.
    Adapted from https://github.com/nicklashansen/tdmpc2/blob/main/tdmpc2/common/layers.py
"""


class NormedLinear(nn.Linear):
    """
    Linear layer with optionally dropout, LayerNorm, and activation.
    """

    def __init__(self, *args, dropout=0., act, **kwargs):
        super().__init__(*args, **kwargs)
        self.ln = nn.LayerNorm(self.out_features)
        self.act = act
        self.dropout = nn.Dropout(dropout) if dropout else None

    def forward(self, x):
        x = super().forward(x)
        if self.dropout:
            x = self.dropout(x)
        return self.act(self.ln(x)) if self.act is not None else self.ln(x)
        # return self.act(x) if self.act is not None else self.ln(x)

    def __repr__(self):
        repr_dropout = f", dropout={self.dropout.p}" if self.dropout else ""
        return f"NormedLinear(in_features={self.in_features}, " \
               f"out_features={self.out_features}, " \
               f"bias={self.bias is not None}{repr_dropout}, " \
               f"act={self.act.__class__.__name__})"


def mlp_norm(in_dim, mlp_dims, out_dim, dropout=0., tanh_out=False):
    """
    input -> NormedLinear(Linear -> Dropout -> LayerNorm -> act)
          -> NormedLinear(Linear -> LayerNorm -> act) -> ...
          -> NormedLinear(Linear -> LayerNorm -> act)
          -> NormedLinear(Linear -> LayerNorm)/Linear -> output
    """
    if isinstance(mlp_dims, int):
        mlp_dims = [mlp_dims]
    dims = [in_dim] + mlp_dims + [out_dim]
    net = nn.ModuleList()
    for i in range(len(dims) - 2):
        net.append(NormedLinear(dims[i], dims[i + 1], dropout=dropout * (i == 0), act=nn.ELU()))
    # net.append(NormedLinear(dims[-2], dims[-1], act=None))
    net.append(nn.Linear(dims[-2], dims[-1]))  # no layer norm
    if tanh_out:
        net.append(nn.Tanh())
    return nn.Sequential(*net)


def mlp(in_dim, mlp_dims, out_dim, tanh_out=False):
    """
    input -> (Linear -> ELU) -> ... -> (Linear -> ELU) -> output
    """
    if isinstance(mlp_dims, int):
        mlp_dims = [mlp_dims]
    dims = [in_dim] + mlp_dims + [out_dim]

    net = nn.ModuleList()
    for i in range(len(dims) - 2):
        net.append(nn.Linear(dims[i], dims[i + 1]))
        net.append(nn.ELU())  # net.append(nn.Mish())  #
    net.append(nn.Linear(dims[-2], dims[-1]))

    if tanh_out:
        net.append(nn.Tanh())

    return nn.Sequential(*net)
