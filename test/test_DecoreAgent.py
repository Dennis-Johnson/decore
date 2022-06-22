from decore import DecoreAgent
import torch 
class TestDecoreAgent:
    def test_setup(self):
        agent = DecoreAgent(layer_num=1, channel_num=1)
        assert agent.action == 1.0 or agent.action == 0.0

    def test_policy(self):
        # Ideally start with no channels pruned. 
        agent = DecoreAgent(layer_num=1, channel_num=1, init_weight=0.0)
        action = agent.policy()
        assert action == 1.0 or action == 0.0
    
    def test_log_prob(self):
        agent = DecoreAgent(layer_num=1, channel_num=1, init_weight=0.0)
        action   = agent.policy()
        log_prob = agent.get_log_prob()
        assert log_prob == agent.dist.log_prob(action)