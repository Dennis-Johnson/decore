from decore import DecoreAgent

class TestDecoreAgent:
    def test_setup(self):
        agent = DecoreAgent(layer_num=1, channel_num=1)
        assert agent.action == 1

    def test_policy(self):
        # Ideally start with no channels pruned. 
        agent = DecoreAgent(layer_num=1, channel_num=1, init_weight=6.9)
        action, prob = agent.policy()
        assert action == 1 and prob > 0.9

        # 0.5 chance of pruning each channel from the start. 
        agent = DecoreAgent(layer_num=1, channel_num=1, init_weight=0.0)
        action, prob = agent.policy()
        assert prob > 0.4 and prob < 0.6

        # All channels are pruned from the start. 
        agent = DecoreAgent(layer_num=1, channel_num=1, init_weight=-6.9)
        action, prob = agent.policy()
        assert action == 0 and prob < 0.1


    