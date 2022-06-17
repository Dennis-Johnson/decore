from CNN import CNN
from utils import train, test
from DecoreAgent import DecoreAgent
from dataloaders import get_dataloaders
from PruningStrategies import DecorePruningStrategy
import torch, torch.optim as optim, torch.nn.functional as F 

# Parameters for the CNN
n_epochs         = 1
batch_size_train = 64
batch_size_test  = 1000
learning_rate    = 0.01
momentum         = 0.5
random_seed      = 1
torch.manual_seed(random_seed)

# Use Metal Performance Shader backend if available.
if not torch.backends.mps.is_available() and not torch.backends.mps.is_built():
        print("MPS not available because the current PyTorch install was not "
              "built with MPS enabled.")
        exit()
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

network = torch.load("./models/baseline_98.pt")
agents  = [] 

train_loader, test_loader = get_dataloaders(batch_size_train, batch_size_test)
examples = enumerate(test_loader) 

# Prune channels using decore and initialise agents.
for name, module in network.named_modules():
    if isinstance(module, torch.nn.Conv2d):
      # Create an RL agent for each conv2d layer.
      agent = DecoreAgent(module, name)
      _mask, _probs = agent.policy()
      agents.append(agent)

      # Applies decore mask to tensor in place. 
      DecorePruningStrategy.apply(agent.module, agent.name, importance_scores=agent.action)
      
      print( "Initial sparsity in {}: {:.2f}%".format(name, 
          100. * float(torch.sum(agent.action == 0))
        / float(agent.module.weight.nelement())
      ))

print("\nPruning Conv2D complete ********************")

optimizer = optim.SGD(network.parameters(), lr=learning_rate,
                      momentum=momentum)
training_stats = []
train_losses   = []
train_counter  = []
test_losses    = []
test_counter   = [ i * len(train_loader.dataset) for i in range(n_epochs + 1) ]

# Train and evaluate performance. 
test_losses = test(network, test_loader, test_losses)
for epoch in range(1, n_epochs + 1):
    # Take an action : Apply a channel mask.
    _mask, _probs = agent.policy()
    DecorePruningStrategy.apply(agent.module, agent.name, importance_scores=agent.action)

    # Take a step : Fine-tune the pruned network.
    train_losses, train_counter = train(epoch, network, train_loader, optimizer, train_losses, train_counter, log_interval = 10)

    # Get reward : Get results of an inference run.
    test_losses = test(network, test_loader, test_losses)

    # Update the policy.
    agent.reinforce()
