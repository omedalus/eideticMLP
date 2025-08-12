import torch
import torch.nn as nn

from eidetic_hidden_layer_lookup import EideticHiddenLayerLookup


# MLP with skip connections and a memory injector.
class MLP_2HLSkipWithEideticMem(nn.Module):
    def __init__(self):
        super().__init__()
        self.eidetic_mem = EideticHiddenLayerLookup()

        # Sensory (784)  -> HL1 (64)                           -> HL2 (16) -> Output (10)
        #                                   -> Indexer (16)    ->
        #                                      Recaller (784)  ->

        self.fc_sensory_to_hl1 = nn.Linear(784, 64)

        self.fc_hl1_to_indexer = nn.Linear(64, 16)

        self.fc_hl1_to_hl2 = nn.Linear(64, 16)
        self.fc_indexer_to_hl2 = nn.Linear(16, 16)
        self.fc_recaller_to_hl2 = nn.Linear(784, 16)

        self.fc_hl2_to_output = nn.Linear(16, 10)

        self.relu = nn.ReLU()

    def forward(self, x_sensory: torch.Tensor):
        activ_hl1 = self.relu(self.fc_sensory_to_hl1(x_sensory))

        activ_indexer = self.relu(self.fc_hl1_to_indexer(activ_hl1))

        # For now, just have the recaller be a fixed value of all 0s.
        # Pad x_sensory with zeros to make it 794 units long for the recaller.
        # activ_recaller = torch.nn.functional.pad(x_sensory, (0, 10), "constant", 0)
        activ_recaller = torch.zeros_like(x_sensory)

        activ_hl2 = self.relu(
            self.fc_hl1_to_hl2(activ_hl1)
            + self.fc_indexer_to_hl2(activ_indexer)
            + self.fc_recaller_to_hl2(activ_recaller)
        )

        activ_output = self.fc_hl2_to_output(activ_hl2)

        if self.training:
            indexerlist = activ_indexer.tolist()
            sensorylist = x_sensory.tolist()

            keyvalpairs = list(zip(indexerlist, sensorylist))
            for key, val in keyvalpairs:
                self.eidetic_mem.insert(key, val)
                print(len(self.eidetic_mem))

        return activ_output
