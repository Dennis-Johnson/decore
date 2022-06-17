import torch 

class DecoreAgent:
    def __init__(self, module, module_name):
        assert isinstance(module, torch.nn.Conv2d), "Only supports pruning Conv2D layers."
        self.module      = module
        self.module_name = module_name
        self.name        = "weight"

        # Weight vector with size (num_channels).
        # Default 6.99 so that prob(keep_channel) ~= 0.99 after sigmoid. 
        self.weights = torch.zeros(module.out_channels, requires_grad=True)
        torch.add(self.weights, 6.99)
        # self.weights.register_hook(lambda x: None)

        # Either 1 to keep channel, or 0 to prune it. 
        self.action = self.probs  = torch.ones_like(self.weights), torch.ones_like(self.weights)
        self.policy()
        # Large negative penalty as reward for wrong predictions.
        self.penalty     = -10
    
    def policy(self):
        '''
        Returns
        mask   : The mask that chooses which channels to keep. Same size as self.module
        probs  : The probabilities of keeping each channel in the layer. 
        '''
        # Multiply --> ImpScores (shape is [O,I, K, K]) * Weights(shape is [O])
        importance_scores  = torch.ones(self.module.out_channels, self.module.in_channels, self.module.kernel_size[0], self.module.kernel_size[1])
        importance_scores *= self.weights.view(-1, 1, 1, 1)

        self.probs  = torch.sigmoid(importance_scores)[:,0,0,0]
        self.action = torch.bernoulli(self.probs)

        # print(self.action, self.probs)
        return torch.bernoulli(torch.sigmoid(importance_scores)), self.probs

    def reinforce(self, batchRewards):
        print(self.module.named_buffers())  # to verify that all masks exist
        print(self.module.named_parameters())  # to verify that all masks exist

        agent_loss_term = 0
        joint_prob = torch.log(torch.prod(self.probs))
        for reward in batchRewards:
            agent_loss_term += joint_prob * reward
        agent_loss_term /= len(batchRewards)
        
        return agent_loss_term


    def calcRewards(self, predictionWasCorrect: bool):
        return torch.sum(self.action == 0) * (1 if predictionWasCorrect else self.penalty)

__all__ = [DecoreAgent]