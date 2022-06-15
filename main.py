import time
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn.utils.prune as prune
import matplotlib.pyplot as plt
from dataloaders import get_dataloaders
from CNN import CNN

# Parameters for the CNN
n_epochs         = 1
batch_size_train = 64
batch_size_test  = 1000
learning_rate    = 0.01
momentum         = 0.5
log_interval     = 10
plot_data        = False
random_seed      = 1
torch.manual_seed(random_seed)


# Check that MPS is available
if not torch.backends.mps.is_available() and not torch.backends.mps.is_built():
        print("MPS not available because the current PyTorch install was not "
              "built with MPS enabled.")
        exit()

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

train_loader, test_loader = get_dataloaders(batch_size_train, batch_size_test)

examples = enumerate(test_loader)
batch_idx, (example_data, example_targets) = next(examples)


if plot_data: 
  fig = plt.figure()
  for i in range(6):
    plt.subplot(2,3,i+1)
    plt.tight_layout()
    plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
    plt.title("Ground Truth: {}".format(example_targets[i]))
    plt.xticks([])
    plt.yticks([])
  plt.show()

network   = torch.load("./models/baseline_98.pt")

# Prune multiple modules. 
for name, module in network.named_modules():
    # prune 20% of connections in all 2D-conv layers
    if isinstance(module, torch.nn.Conv2d):
        prune.ln_structured(module, name="weight", amount=0.2, n=2, dim=0)
    # prune 40% of connections in all linear layers
    elif isinstance(module, torch.nn.Linear):
        prune.ln_structured(module, name="weight", amount=0.2, n=2, dim=0)
print("Pruning complete.")

optimizer = optim.SGD(network.parameters(), lr=learning_rate,
                      momentum=momentum)

train_losses  = []
train_counter = []
test_losses   = []
test_counter  = [i*len(train_loader.dataset) for i in range(n_epochs + 1)]

def train(epoch):
  network.train()

  for batch_idx, (data, target) in enumerate(train_loader):
    optimizer.zero_grad()
    output = network(data)
    loss = F.nll_loss(output, target) 
    loss.backward()
    optimizer.step()

    if batch_idx % log_interval == 0:
      print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        epoch, batch_idx * len(data), len(train_loader.dataset),
        100. * batch_idx / len(train_loader), loss.item()))

      train_losses.append(loss.item())
      train_counter.append((batch_idx*64) + ((epoch-1)*len(train_loader.dataset)))
  torch.save(network.state_dict(), './results/model.pth')
  torch.save(optimizer.state_dict(), './results/optimizer.pth')

def test():
  network.eval()
  test_loss = 0
  correct   = 0

  with torch.no_grad():
    for data, target in test_loader:
      output     = network(data)
      test_loss += F.nll_loss(output, target, size_average=False).item()
      pred       = output.data.max(1, keepdim=True)[1]
      correct   += pred.eq(target.data.view_as(pred)).sum()

  test_loss /= len(test_loader.dataset)
  test_losses.append(test_loss)

  print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    test_loss, correct, len(test_loader.dataset),
    100. * correct / len(test_loader.dataset)))

training_stats = []

# Train and benchmark M1 CPU vs GPU
startEndtoEnd = time.perf_counter()
test()
for epoch in range(1, n_epochs + 1):
    startEpoch = time.perf_counter()
    train(epoch)
    training_stats.append(time.perf_counter() - startEpoch)

    test()

print(f"End-to-end training MNIST time: {time.perf_counter() - startEndtoEnd}s")
print(f"Timings: {training_stats}\nAvg: {sum(training_stats) / n_epochs}\nMin: {min(training_stats)}\nMax: {max(training_stats)}")


if plot_data: 
  fig = plt.figure()
  plt.plot(train_counter, train_losses, color='blue')
  plt.scatter(test_counter, test_losses, color='red')
  plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
  plt.xlabel('number of training examples seen')
  plt.ylabel('negative log likelihood loss')
  plt.show()