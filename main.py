from CNN import CNN
from utils import train, test
from DecoreAgent import DecoreAgent
from dataloaders import get_dataloaders
from PruningStrategies import DecorePruningStrategy
import torch, torch.optim as optim, torch.nn.functional as F 

# Parameters for the CNN
n_epochs         = 10
batch_size_train = 64
batch_size_test  = 1000
learning_rate    = 0.01
momentum         = 0.5
random_seed      = 1
load_baseline    = False
torch.manual_seed(random_seed)

# Use Metal Performance Shader backend if available.
if not torch.backends.mps.is_available() and not torch.backends.mps.is_built():
        print("MPS not available because the current PyTorch install was not "
              "built with MPS enabled.")
        exit()
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

if load_baseline:
  network = torch.load("./models/baseline_98.pt")
else:
  network = CNN()
agents  = [] 

train_loader, test_loader = get_dataloaders(batch_size_train, batch_size_test)
examples = enumerate(test_loader) 

# Prune channels using decore and initialise agents.
for name, module in network.named_modules():
    if isinstance(module, torch.nn.Conv2d):
      # Create an RL agent for each conv2d layer.
      agent = DecoreAgent(module, name)
      agents.append(agent)
      
      print( "Initial sparsity in {}: {:.2f}%".format( name, 
        100. * torch.sum(agent.action == 0) / agent.action.shape[0]
      ))

print("\nPruning Conv2D complete ********************")

training_stats = []
train_losses   = []
train_counter  = []
test_losses    = []
test_counter   = [ i * len(train_loader.dataset) for i in range(n_epochs + 1) ]

optimizer    = optim.SGD(network.parameters(), lr=learning_rate,momentum=momentum)
rl_optimiser = optim.Adam([agent.weights for agent in agents], lr=0.001)

# Train and evaluate performance. 
test_losses, predictions = test(network, test_loader, test_losses)
for epoch in range(1, n_epochs + 1):

    # Take an action : Apply a channel mask.
    for agent in agents:
      mask, probs = agent.policy()
      DecorePruningStrategy.apply(agent.module, agent.name, importance_scores=mask)
      print( "Sparsity in {}: {:.2f}%".format( name, 
        100. * torch.sum(agent.action == 0) / agent.action.shape[0]
      ))
    
    # Take a step : Fine-tune the pruned network.
    train_losses, train_counter = train(epoch, network, train_loader, optimizer, train_losses, train_counter, log_interval = 10)

    # Get reward : Get results of an inference run.
    test_losses, predictions = test(network, test_loader, test_losses)
    
    total_loss = 0
    rl_optimiser.zero_grad()
    
    for agent in agents:
      batchRewards = []
      for prediction in predictions:
        batchRewards.append(agent.calcRewards(prediction))

      # Update the policy.
      total_loss += agent.reinforce(batchRewards)
      
    total_loss.backward()
    rl_optimiser.step()
