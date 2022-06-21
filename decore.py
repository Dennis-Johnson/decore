import torch, torch.nn as nn

class DecoreAgent:
    def __init__(self, layer_num:int, channel_num: int):
        self.channel_num = channel_num
        self.layer_num   = layer_num

        # Initially 6.9 so that probability(keep_channel) ~= 0.99
        self.weight = torch.tensor(0.0, requires_grad=True)

        # Initialise probs and actions.
        self.policy()

    def policy(self):
        self.prob   = torch.sigmoid(self.weight)
        self.action = torch.bernoulli(self.prob)
        return self.action, self.prob

class DecoreLayer:
    # Rewards for right and wrong predicitons. 
    REWARD_RIGHT =  1
    REWARD_WRONG = -1

    def __init__(self, module:nn.Module, module_name: str, agents: list):
        self.module      = module
        self.module_name = module_name
        self.agents      = agents
        self.layer_mask  = None
        self.layer_probs = None

    def layer_policy(self):
        mask  = []
        probs = []

        for agent in self.agents:
            action, prob = agent.policy()
            mask.append(action)
            probs.append(prob)
        
        
        self.layer_mask  = torch.tensor(mask)
        self.layer_probs = torch.tensor(probs)
        return self.layer_mask, self.layer_probs

    def layer_reward(self, predictionWasCorrect:bool):
        droppedChannels = 0
        for agent in self.agents:
            droppedChannels = droppedChannels * (1 - agent.action)
        
        return droppedChannels * (self.REWARD_RIGHT if predictionWasCorrect else self.REWARD_WRONG)

    def calc_sparsity(self):
        notDropped = 0
        for agent in self.agents:
            notDropped += agent.action
        return 100.0 * (float(len(self.agents) - notDropped)) / float(len(self.agents))

    
__all__ = [DecoreAgent, DecoreLayer]