from attr import s
from numpy import dtype
import torch, torch.nn.utils.prune as prune

class Agent:
    # TODO: Refactor this out once it works. 
    def __init__(self, module, module_name):
        assert isinstance(module, torch.nn.Conv2d), "Only supports pruning Conv2D layers."
        self.module      = module
        self.module_name = module_name
        self.name        = "weight"

        # Weight vector with size (num_channels).
        # Default 6.99 so that prob(keep_channel) ~= 0.99 after sigmoid. 
        self.weights = torch.zeros(module.out_channels) + 6.99

        # Either 1 to keep channel, or 0 to prune it. 
        self.action, self.probs  = None, None
        self.reward  = 0
    
    def policy(self):
        '''
        Returns
        action : The mask that chooses which channels to keep. 
        probs  : The probabilities of keeping each channel in the layer. 
        '''
        importance_scores  = torch.zeros(self.module.out_channels, self.module.in_channels, self.module.kernel_size[0], self.module.kernel_size[1])
        importance_scores *= self.weights.view(-1, 1, 1, 1)
        self.probs  = torch.sigmoid(importance_scores)
        self.action = torch.bernoulli(self.probs)

        return self.action, self.probs

    def reinforce():
        pass

    def calcReward():
        pass

class DecorePruningStrategy(prune.BasePruningMethod):
    '''
    Prune using the given importance scores
    '''
    def compute_mask(self, importance_scores, default_mask):
        '''
        Overriden method, applies a channel mask to the layer.
        Importance scores is the actual mask used. 
        '''
        return importance_scores
    
__all__ = [DecorePruningStrategy,  Agent]
