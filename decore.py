import torch.nn.utils.prune as prune
import torch

class DecorePruningStrategy(prune.BasePruningMethod):
    '''
    Prune every other entry in a tensor
    '''
    PRUNING_TYPE = 'unstructured'

    def compute_mask(self, t, default_mask):
        assert(default_mask.dim() == 4), f"Expected 4 dims (OIFF), got {default_mask.dim()}"
        mask = default_mask.clone()

        for out_dim in range(default_mask.shape[0]):
            if out_dim % 2 != 0:
                mask[out_dim] = 0
                assert (torch.numel(mask[out_dim]) - torch.count_nonzero(mask[out_dim])) == torch.numel(mask[out_dim])
            else:
                assert torch.count_nonzero(mask[out_dim]) == torch.numel(mask[out_dim])

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
