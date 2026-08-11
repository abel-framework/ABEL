#!/usr/bin/env python3
#
# Helper definitions adapted from the ImpactX PyTorch surrogate example:
# https://impactx.readthedocs.io/en/latest/usage/examples/pytorch_surrogate_model/README.html

from enum import Enum

try:
    import torch
    from torch import nn
except ImportError as exc:
    raise SystemExit("PyTorch is required to run this surrogate example.") from exc


class Activation(Enum):
    ReLU = 1
    Tanh = 2
    PReLU = 3
    Sigmoid = 4


def get_enum_type(type_to_test, enum_class):
    if isinstance(type_to_test, enum_class):
        return type_to_test
    if isinstance(type_to_test, int):
        return enum_class(type_to_test)
    if isinstance(type_to_test, str):
        return getattr(enum_class, type_to_test)
    raise TypeError(f"Cannot convert {type_to_test!r} to {enum_class.__name__}")


class ConnectedNN(nn.Module):
    def __init__(self, layers, device=None):
        super().__init__()
        self.stack = nn.Sequential(*layers)
        if device is not None:
            self.to(device)

    def forward(self, x):
        return self.stack(x)


class OneActNN(ConnectedNN):
    def __init__(self, n_in, n_out, n_hidden_nodes, n_hidden_layers, act, device=None):
        self.n_in = n_in
        self.n_out = n_out
        self.n_hidden_layers = n_hidden_layers
        self.n_hidden_nodes = n_hidden_nodes
        self.act = act

        layers = [nn.Linear(self.n_in, self.n_hidden_nodes)]
        for ii in range(self.n_hidden_layers):
            if self.act is Activation.ReLU:
                layers += [nn.ReLU()]
            if self.act is Activation.Tanh:
                layers += [nn.Tanh()]
            if self.act is Activation.PReLU:
                layers += [nn.PReLU()]
            if self.act is Activation.Sigmoid:
                layers += [nn.Sigmoid()]

            if ii < self.n_hidden_layers - 1:
                layers += [nn.Linear(self.n_hidden_nodes, self.n_hidden_nodes)]

        layers += [nn.Linear(self.n_hidden_nodes, self.n_out)]
        super().__init__(layers, device)


class surrogate_model:
    """
    Wrapper for the pretrained ImpactX surrogate models.

    The stored model expects dimensional inputs, normalizes them internally, runs
    the fully connected network, and returns dimensional outputs.
    """

    def __init__(self, model_file, device=None):
        self.device = device
        if device is None:
            model_dict = torch.load(model_file, map_location="cpu", weights_only=False)
        else:
            model_dict = torch.load(model_file, map_location=device, weights_only=False)

        self.source_means = torch.tensor(
            model_dict["source_means"], device=self.device, dtype=torch.float64
        )
        self.target_means = torch.tensor(
            model_dict["target_means"], device=self.device, dtype=torch.float64
        )
        self.source_stds = torch.tensor(
            model_dict["source_stds"], device=self.device, dtype=torch.float64
        )
        self.target_stds = torch.tensor(
            model_dict["target_stds"], device=self.device, dtype=torch.float64
        )

        n_in = len(self.source_means)
        n_out = len(self.target_means)
        n_hidden_nodes = model_dict["n_hidden_nodes"]
        n_hidden_layers = model_dict["n_hidden_layers"]
        activation = get_enum_type(model_dict["activation"], Activation)

        self.model = OneActNN(
            n_in=n_in,
            n_out=n_out,
            n_hidden_nodes=n_hidden_nodes,
            n_hidden_layers=n_hidden_layers,
            act=activation,
            device=self.device,
        )
        self.model.load_state_dict(model_dict["model_state_dict"])
        self.model.eval()

    def __call__(self, x):
        x = (x - self.source_means) / self.source_stds
        y = self.model(x.float()).double()
        return y * self.target_stds + self.target_means
