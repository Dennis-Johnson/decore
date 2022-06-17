import torch.nn.utils.prune as prune

class DecorePruningStrategy(prune.BasePruningMethod):
    '''
    Prune using the given importance scores
    '''
    PRUNING_TYPE = "unstructured"
    
    def compute_mask(self, importance_scores, default_mask):
        '''
        Overriden method, applies a channel mask to the layer.
        Importance scores is the actual mask used. 
        '''
        self.dim = 0
        return importance_scores
    
__all__ = [DecorePruningStrategy]
