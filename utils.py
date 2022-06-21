import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


def train(epoch: int, network: nn.Module, train_loader, optimizer, train_losses, train_counter, log_interval = 10):
  ''' Train on dataset for one epoch. Returns accumulated losses and a counter. 
  '''
  network.train()
  for batch_idx, (data, target) in enumerate(train_loader):
    optimizer.zero_grad()
    output = network(data)
    loss   = F.nll_loss(output, target) 
    loss.backward(retain_graph=True)
    optimizer.step()

    if batch_idx % log_interval == 0:
      print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        epoch, batch_idx * len(data), len(train_loader.dataset),
        100. * batch_idx / len(train_loader), loss.item()))

      train_losses.append(loss.item())
      train_counter.append((batch_idx*64) + ((epoch-1)*len(train_loader.dataset)))
  
  torch.save(network.state_dict(),   './results/model.pth')
  torch.save(optimizer.state_dict(), './results/optimizer.pth')

  return train_losses, train_counter


def test(network: nn.Module, test_loader, test_losses):
  ''' Run inference on test data.
  '''
  network.eval()
  test_loss = 0
  correct   = 0
  predictions = []

  with torch.no_grad():
    for data, target in test_loader:
      output     = network(data)
      test_loss += F.nll_loss(output, target, size_average=False).item()
      pred       = output.data.max(1, keepdim = True)[1]
      correct   += pred.eq(target.data.view_as(pred)).sum()
      predictions.extend(pred.eq(target.data.view_as(pred)).flatten())

  test_loss /= len(test_loader.dataset)
  test_losses.append(test_loss)
  accuracy = 100. * correct / len(test_loader.dataset)
  print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    test_loss, correct, len(test_loader.dataset),
    accuracy))

  return test_losses, predictions, accuracy


def plot_data(example_data, example_targets):
  ''' Plot MNIST data in a grid layout. 

  Get sample inputs like this: batch_idx, (example_data, example_targets) = next(examples)
  '''
  fig = plt.figure()
  for i in range(6):
    plt.subplot(2,3,i+1)
    plt.tight_layout()
    plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
    plt.title("Ground Truth: {}".format(example_targets[i]))
    plt.xticks([])
    plt.yticks([])
  plt.show()

def plot_loss(train_losses, train_counter, test_losses, test_counter):
  ''' Plot training and testing losses. 
  '''
  fig = plt.figure()
  plt.plot(train_counter, train_losses, color='blue')
  plt.scatter(test_counter, test_losses, color='red')
  plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
  plt.xlabel('number of training examples seen')
  plt.ylabel('negative log likelihood loss')
  plt.show()