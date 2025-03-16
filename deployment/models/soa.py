import torch

# Define PyTorch-based models
class TorchStandardScaler(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mean = None
        self.std = None
    
    def fit(self, X):
        X_tensor = torch.tensor(X, dtype=torch.float32)
        self.mean = X_tensor.mean(0, keepdim=True)
        self.std = X_tensor.std(0, keepdim=True) + 1e-7
        return self
    
    def transform(self, X):
        X_tensor = torch.tensor(X, dtype=torch.float32)
        return (X_tensor - self.mean) / self.std
    
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

class TorchSVR(torch.nn.Module):
    def __init__(self, C=1.0, epsilon=0.1):
        super().__init__()
        self.C = C
        self.epsilon = epsilon
        self.model = None
        
    def fit(self, X, y):
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32).reshape(-1)
        n_features = X_tensor.shape[1]
        
        self.model = torch.nn.Sequential(
            torch.nn.Linear(n_features, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 1)
        )
        
        criterion = torch.nn.SmoothL1Loss(beta=self.epsilon)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01, weight_decay=1/self.C)
        
        for epoch in range(100):
            optimizer.zero_grad()
            outputs = self.model(X_tensor).reshape(-1)
            loss = criterion(outputs, y_tensor)
            loss.backward()
            optimizer.step()
        
        return self
    
    def predict(self, X):
        X_tensor = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            return self.model(X_tensor).cpu().numpy()

class TorchDecisionTree(torch.nn.Module):
    def __init__(self, max_depth=100):
        super().__init__()
        self.max_depth = max_depth
        self.model = None
        
    def fit(self, X, y):
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)
        n_features = X_tensor.shape[1]
        
        self.model = torch.nn.Sequential(
            torch.nn.Linear(n_features, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1)
        )
        
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        
        for epoch in range(100):
            optimizer.zero_grad()
            outputs = self.model(X_tensor)
            loss = criterion(outputs, y_tensor)
            loss.backward()
            optimizer.step()
        
        return self
    
    def predict(self, X):
        X_tensor = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            return self.model(X_tensor).cpu().numpy()

class TorchMLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = None
        
    def fit(self, X, y):
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)
        n_features = X_tensor.shape[1]
        
        self.model = torch.nn.Sequential(
            torch.nn.Linear(n_features, 100),
            torch.nn.ReLU(),
            torch.nn.Linear(100, 100),
            torch.nn.ReLU(),
            torch.nn.Linear(100, 100),
            torch.nn.ReLU(),
            torch.nn.Linear(100, 1)
        )
        
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        
        for epoch in range(100):
            optimizer.zero_grad()
            outputs = self.model(X_tensor)
            loss = criterion(outputs, y_tensor)
            loss.backward()
            optimizer.step()
        
        return self
    
    def predict(self, X):
        X_tensor = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            return self.model(X_tensor).cpu().numpy()

class TorchKNN(torch.nn.Module):
    def __init__(self, n_neighbors=7):
        super().__init__()
        self.n_neighbors = n_neighbors
        self.X_train = None
        self.y_train = None
        
    def fit(self, X, y):
        self.X_train = torch.tensor(X, dtype=torch.float32)
        self.y_train = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)
        return self
    
    def predict(self, X):
        X_tensor = torch.tensor(X, dtype=torch.float32)
        
        # Calculate distances
        distances = torch.cdist(X_tensor, self.X_train)
        
        # Get indices of k nearest neighbors
        _, indices = torch.topk(distances, self.n_neighbors, dim=1, largest=False)
        
        # Get predictions
        neighbor_values = torch.stack([self.y_train[idx] for idx in indices])
        predictions = torch.mean(neighbor_values, dim=1)
        
        return predictions.cpu().numpy()

class TorchLinearRegression(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = None
        
    def fit(self, X, y):
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)
        n_features = X_tensor.shape[1]
        
        self.model = torch.nn.Linear(n_features, 1)
        
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        
        for epoch in range(100):
            optimizer.zero_grad()
            outputs = self.model(X_tensor)
            loss = criterion(outputs, y_tensor)
            loss.backward()
            optimizer.step()
        
        return self
    
    def predict(self, X):
        X_tensor = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            return self.model(X_tensor).cpu().numpy()

class TorchPipeline:
    def __init__(self, steps):
        self.steps = steps
        
    def fit(self, X, y):
        data = X
        for name, transform in self.steps[:-1]:
            data = transform.fit_transform(data)
        
        name, model = self.steps[-1]
        model.fit(data, y)
        return self
    
    def predict(self, X):
        data = X
        for name, transform in self.steps[:-1]:
            data = transform.transform(data)
        
        name, model = self.steps[-1]
        return model.predict(data)