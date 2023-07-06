import torch

# NOTE: model(x) returns the loss
class Trainer:
    def __init__(self, model, lr, weight_decay):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), 
                                          lr=lr, weight_decay=weight_decay)
    
    def train_loop(self, train_loader, num_epochs, device, print_every=500):
        loss_list = []
        self.model = self.model.to(device)
        for epoch in range(num_epochs):
            loss = self.train_one_epoch(train_loader, device)
            if (epoch + 1) % print_every == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
            loss_list.append(loss.item())
        return loss_list
    
    def train_one_epoch(self, train_loader, device):
        for batc_idx, (features, labels) in enumerate(train_loader):
            features, labels = features.to(device), labels.to(device)
            self.optimizer.zero_grad()
            loss = self.model(features, labels)
            loss.backward()
            self.optimizer.step()
        return loss
