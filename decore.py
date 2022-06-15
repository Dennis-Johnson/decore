import torch.nn.utils.prune as prune

class DecorePruningStrategy(prune.BasePruningMethod):
    '''
    Prune every other entry in a tensor
    '''
    PRUNING_TYPE = 'unstructured'

    def compute_mask(self, t, default_mask):
        mask = default_mask.clone()
        mask.view(-1)[::2] = 0
        return mask

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
