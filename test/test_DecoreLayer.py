import torch, torch.nn as nn
from decore import DecoreAgent, DecoreLayer

class TestDecoreLayer:
    @staticmethod
    def init_layer():
        '''Initialises a DecoreLayer'''
        module = nn.Conv2d(10, 20, kernel_size=5)
        agents = [DecoreAgent(0, channel_num, 6.9) for channel_num in range(module.weight.shape[0])]
        return DecoreLayer(module, "conv1", agents)

    def test_initial_sparsity(self):
        layer = TestDecoreLayer.init_layer()
        assert layer.calc_sparsity() == 0.0

    def test_layer_policy(self):
        module = nn.Conv2d(2, 4, kernel_size=5)
        # Initialise layer with agents of alternating weights.
        agents = [DecoreAgent(0, channel_num, (6.9 if (channel_num % 2 ==0) else -6.9)) for channel_num in range(module.weight.shape[0])]
        layer  =  DecoreLayer(module, "conv1", agents)

        layer_mask, layer_probs = layer.layer_policy()
     
        assert torch.allclose(layer_mask, torch.tensor([1.0, 0.0, 1.0, 0.0]))
        assert torch.allclose(layer_probs, torch.tensor([0.99, 0, 0.99, 0]), atol=1e-2)

    def test_decore_pruning(self):
        module = nn.Conv2d(2, 4, kernel_size=5)
        # Initialise layer with agents of alternating weights.
        agents = [DecoreAgent(0, channel_num, (6.9 if (channel_num % 2 ==0) else -6.9)) for channel_num in range(module.weight.shape[0])]
        layer  =  DecoreLayer(module, "conv1", agents)

        layer.decore_prune()
        assert layer.calc_sparsity() == 50.0

