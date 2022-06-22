from pyexpat import model
import re
from CNN import CNN
from utils import train, test
from decore import DecoreAgent, DecoreLayer
from dataloaders import get_dataloaders
import torch, torch.optim as optim, torch.nn.functional as F
import torch.nn.utils.prune as prune

# Parameters for the CNN
n_epochs         = 40
batch_size_train = 256
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

optim_params = []
layers       = []
layer_num    = 0

for mod_name, module in network.named_modules():
    # Don't prune the last layer.
    if mod_name == "fc2":
      break

    # Initialise agents for each channel in each layer.
    if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
      agents = [DecoreAgent(layer_num, channel_num, init_weight=6.9) for channel_num in range(module.weight.shape[0])]

      # The RL Optimiser will optimise the agent weights. 
      optim_params.extend([agent.weight for agent in agents])

      layer = DecoreLayer(module, mod_name, agents)
      layers.append(layer)
      layer_num += 1
  
print("\nAgents Initialised ********************")

training_stats = []
train_losses   = []
train_counter  = []
test_losses    = []
test_counter   = [ i * len(train_loader.dataset) for i in range(n_epochs + 1) ]
best_test_loss = 10000
optimizer    = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)
rl_optimizer = optim.Adam(optim_params, lr=0.01)

#### Train and evaluate performance. 
test_losses, predictions, acc_ = test(network, test_loader, test_losses)


for epoch in range(1, n_epochs + 1):
    #### Take an action : Apply a channel mask to each layer.
    for layer in layers:
      layer.decore_prune()

    #### Take a step : Fine-tune the pruned network.
    train_losses, train_counter = train(epoch, network, train_loader, optimizer, train_losses, train_counter, log_interval = 100)

    #### Get reward : Get results of an inference run.
    test_losses, predictions, accuracy = test(network, test_loader, test_losses)

    # Save the best model yet
    test_loss = test_losses[-1]
    if test_loss < best_test_loss:
      best_test_loss = test_loss
      print(f"Saved best model, test_loss {test_loss}")
      torch.save(network, f"./models/best_acc{accuracy}_ep{epoch}.pt")

    rl_optimizer.zero_grad()
    loss = torch.tensor(0.0, requires_grad=True)

    #### Update RL agent weights.
    for layer in layers:
      mask_, log_probs = layer.layer_mask, layer.layer_log_probs

      for prediction in predictions:
        reward = layer.layer_reward(prediction)
        loss = torch.add(loss, -1 * torch.sum(log_probs) * reward) 
      loss = torch.div(loss, len(predictions))
      print(f"RL Loss: {loss}")

      #### Remove the previous mask to avoid cascading.
      prune.remove(layer.module, name="weight")

    #### Update the policy  
    loss.backward()
    rl_optimizer.step()


    
