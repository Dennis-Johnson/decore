import torch, torch.nn as nn
from PruningStrategies import DecorePruningStrategy
import torch.distributions as dist

class DecoreAgent:
    def __init__(self, layer_num:int, channel_num: int, init_weight:float=6.9):
        self.channel_num = channel_num
        self.layer_num   = layer_num

        # Initially 6.9 so that probability(keep_channel) ~= 0.99
        self.weight = torch.tensor(init_weight, requires_grad=True)

        # Initialise probs and actions.
        self.policy()

    def policy(self):
        self.logits = torch.sigmoid(self.weight)
        self.dist   = dist.Bernoulli(logits = self.logits)
        self.action = torch.tensor(self.dist.sample().item())
        return self.action
    
    def get_log_prob(self):
        return self.dist.log_prob(self.action)

class DecoreLayer:
    # Rewards for right and wrong predicitons. 
    REWARD_RIGHT =  1
    REWARD_WRONG = -4

    def __init__(self, module:nn.Module, module_name: str, agents: list):
        self.module      = module
        self.module_name = module_name
        self.agents      = agents
        self.layer_mask  = None
        self.layer_probs = None

    def layer_policy(self):
        mask  = []
        log_probs = []

        for agent in self.agents:
            action   = agent.policy()
            log_prob = agent.get_log_prob()
            mask.append(action)
            log_probs.append(log_prob)
        
        
        self.layer_mask      = torch.tensor(mask)
        self.layer_log_probs = torch.tensor(log_probs)
        return self.layer_mask, self.layer_log_probs

    def layer_reward(self, predictionWasCorrect:bool):
        droppedChannels = 0
        for agent in self.agents:
            droppedChannels = droppedChannels + (1 - agent.action)
        
        return droppedChannels * (self.REWARD_RIGHT if predictionWasCorrect else self.REWARD_WRONG)

    def calc_sparsity(self):
        notDropped = 0
        for agent in self.agents:
            notDropped += agent.action
        return 100.0 * (float(len(self.agents) - notDropped)) / float(len(self.agents))

    def decore_prune(self):
        channel_mask, _ = self.layer_policy()

        importance_scores  = torch.ones((self.module.weight.shape))

        if isinstance(self.module, torch.nn.Conv2d):
            importance_scores *= channel_mask.view(-1, 1, 1, 1)

        elif isinstance(self.module, torch.nn.Linear):
            importance_scores *= channel_mask.view(-1, 1)

        
        DecorePruningStrategy.apply(self.module, name="weight", importance_scores=importance_scores)
        print(f"Sparsity ({self.module_name}) : {self.calc_sparsity()}%")

__all__ = [DecoreAgent, DecoreLayer]