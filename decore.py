import torch, torch.nn.utils.prune as prune

class Agent:
    # TODO: Refactor this out once it works. 
    def __init__(self, module, module_name):
        self.reward  = 0

        # Weight vector with size (num_channels)
        self.weights = torch.zeros(module.out_channels)
        
        assert isinstance(module, torch.nn.Conv2d)
        self.module = module
        self.name   = module_name

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


def decore_pruning(module, name, agent):
    '''
    Prunes tensor corresponding to parameter called `name` in `module`
    
    Args:
        module (nn.Module): module containing the tensor to prune
        name (string): parameter name within `module` on which pruning
                will act.
        agent (Agent): The RL agent that chooses channels to prune in the layer. 

    Returns:
       mask  : The mask that chooses which channels to keep. 
       probs : The probabilities of keeping each channel in the layer. 
    '''
    importance_scores  = torch.zeros(module.out_channels, module.in_channels, module.kernel_size[0], module.kernel_size[1])
    importance_scores *= agent.weights.view(-1, 1, 1, 1)
    probs = torch.sigmoid(importance_scores)
    mask  = torch.bernoulli(probs)

    # Applies decore mask to tensor in place. 
    DecorePruningStrategy.apply(module, name, importance_scores=mask)
    return mask, probs
    
__all__ = [DecorePruningStrategy, decore_pruning, Agent]
