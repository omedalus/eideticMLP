import torch
import torch.nn as nn

from eidetic_hidden_layer_lookup import EideticHiddenLayerLookup

# TEMPORARILY EXCLUDED WHILE TESTING IS OCCURRING


# MLP with skip connections and a memory injector.
class MLP_2HLSkipWithEideticMem(nn.Module):
    def __init__(self):
        super().__init__()
        self.eidetic_mem = EideticHiddenLayerLookup()

        self.fullconn_sensory_to_indexer = nn.Linear(784, 64)
        self.fullconn_indexer_to_integrator = nn.Linear(64, 32)

        self.fullconn_sensory_skip_to_integrator = nn.Linear(784, 32)
        self.fullconn_recaller_to_integrator = nn.Linear(784, 32)

        self.fullconn_integrator_to_output = nn.Linear(32, 10)

        self.relu = nn.ReLU()

    def forward(self, x_sensory: torch.Tensor):
        # Remember: x_sensory is a *batch* of sensory inputs.
        activations_indexer = self.relu(self.fullconn_sensory_to_indexer(x_sensory))

        # activations_indexer is now a *batch* of activation vectors of the
        # indexer layer. For each one, we need to find the corresponding
        # past sensory vector in the eidetic memory, which will populate
        # the recall layer.
        x_recaller = torch.zeros_like(x_sensory)
        if len(self.eidetic_mem) > 0:
            x_recaller = self.eidetic_mem.lookup_batch(activations_indexer)

        if self.training:
            # Update the eidetic memory, associating the current
            # indexer activations with the sensory input
            # Only do this on the training pass.
            self.eidetic_mem.insert_batch(activations_indexer, x_sensory)

        activations_integrator = self.relu(
            self.fullconn_indexer_to_integrator(activations_indexer)
            + self.fullconn_recaller_to_integrator(x_recaller)
        )

        activations_output = self.fullconn_integrator_to_output(activations_integrator)
        return activations_output
