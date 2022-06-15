import torch, torch.nn.utils.prune as prune

class DecorePruningStrategy(prune.BasePruningMethod):
    '''
    Prune every other entry in a tensor
    '''
    def compute_mask(self, t_unused, default_mask):
        '''
        Overriden method, applies a channel mask to the layer.
        '''
        channel_mask =  self.decore_channel_mask(default_mask)
        mask = default_mask.clone()
        for channel_num, chn_mask in enumerate(channel_mask):
            mask[channel_num] *= chn_mask
        return mask
        
    def decore_channel_mask(self, default_mask):
        '''
        Compute the channel mask given some weights. 
        1. A weight Wj is given for each channel in the layer.
        2. Init each channel weight=6.99 so prob of keeping ~= 1.
        3. Use these probs to draw either 0, 1 from a Bernoulli dist. 
        '''
        weights = torch.zeros(default_mask.shape[0]) + 6.99
        probs   = torch.sigmoid(weights)
        return torch.bernoulli(probs)


def decore_structured(module, name):
    '''
    Prunes tensor corresponding to parameter called `name` in `module`
    
    Args:
        module (nn.Module): module containing the tensor to prune
        name (string): parameter name within `module` on which pruning
                will act.

    Returns:
        module (nn.Module): modified (i.e. pruned) version of the input
            module
    '''

    DecorePruningStrategy.apply(module, name)
    return module
    
__all__ = [DecorePruningStrategy, decore_structured]
