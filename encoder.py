import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


class IPCEncoder(nn.Module):
    """
    CNN-based encoder for predicting IPC curves from pore network geometry.
    Uses only convolutional layers with adaptive pooling, no fully connected layers.
    """
    
    def __init__(self, output_size):
        """
        Parameters
        ----------
        output_size : int
            Size of the flattened IPC curve output
        """
        super(IPCEncoder, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            
            nn.Conv3d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            
            nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            
            nn.AdaptiveAvgPool3d((2, 2, 2)),
            nn.Conv3d(64, 1, kernel_size=1),
        )
        
        self.output_size = output_size
        
    def forward(self, x):
        """
        Forward pass through the encoder.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, 1, 9, 9, 9)
        
        Returns
        -------
        out : torch.Tensor
            Output tensor of shape (batch, output_size)
        """
        x = self.features(x)
        x = x.flatten(start_dim=1)
        
        # Handle output size mismatch
        if x.shape[1] != self.output_size:
            if x.shape[1] < self.output_size:
                pad_size = self.output_size - x.shape[1]
                x = torch.nn.functional.pad(x, (0, pad_size))
            else:
                x = x[:, :self.output_size]
        
        return x


def prepare_ipc_data(pressure_data, saturation_data, target_size=None):
    """
    Prepare IPC curves as fixed-size vectors.
    
    Parameters
    ----------
    pressure_data : np.ndarray
        Array of pressure curves (variable length)
    saturation_data : np.ndarray
        Array of saturation curves (variable length)
    target_size : int, optional
        Target size for each curve (default: max observed length)
    
    Returns
    -------
    ipc_vectors : np.ndarray
        Flattened IPC vectors of shape (n_samples, target_size*2)
    """
    # Find max length if not specified
    if target_size is None:
        max_pressure_len = max(len(p) if isinstance(p, np.ndarray) else len(p) for p in pressure_data)
        max_sat_len = max(len(s) if isinstance(s, np.ndarray) else len(s) for s in saturation_data)
        target_size = max(max_pressure_len, max_sat_len)
    
    ipc_vectors = []
    
    for i in range(len(pressure_data)):
        pressure = pressure_data[i] if isinstance(pressure_data[i], np.ndarray) else np.array(pressure_data[i])
        saturation = saturation_data[i] if isinstance(saturation_data[i], np.ndarray) else np.array(saturation_data[i])
        
        # Pad or truncate to target size
        pressure_padded = np.zeros(target_size)
        pressure_padded[:len(pressure)] = pressure[:target_size]
        
        saturation_padded = np.zeros(target_size)
        saturation_padded[:len(saturation)] = saturation[:target_size]
        
        # Concatenate pressure and saturation
        ipc_vector = np.concatenate([pressure_padded, saturation_padded])
        ipc_vectors.append(ipc_vector)
    
    return np.array(ipc_vectors)


def train_model(model, train_loader, test_loader, epochs=100, device='cpu'):
    """
    Train the encoder model.
    
    Parameters
    ----------
    model : IPCEncoder
        The model to train
    train_loader : torch.utils.data.DataLoader
        Training data loader
    test_loader : torch.utils.data.DataLoader
        Test data loader
    epochs : int
        Number of training epochs
    device : str
        Device to train on
    
    Returns
    -------
    train_losses : list
        Training loss per epoch
    test_losses : list
        Test loss per epoch
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    train_losses = []
    test_losses = []
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Testing
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                test_loss += loss.item()
        
        test_loss /= len(test_loader)
        test_losses.append(test_loss)
        
        if (epoch + 1) % 20 == 0:
            print(f'Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.6f}, Test Loss: {test_loss:.6f}')
    
    return train_losses, test_losses


if __name__ == '__main__':
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Load data
    print('\nLoading data...')
    input_tensors = np.load('data/input.npy')
    ipc_data = np.load('data/ipc.npz')
    pressure = ipc_data['pressure']
    saturation = ipc_data['saturation']
    
    print(f'Input shape: {input_tensors.shape}')
    print(f'Pressure shape: {[p.shape for p in pressure]}')
    print(f'Saturation shape: {[s.shape for s in saturation]}')
    
    # Prepare IPC output vectors
    ipc_vectors = prepare_ipc_data(pressure, saturation)
    output_size = ipc_vectors.shape[1]
    print(f'IPC vector size: {output_size}')
    
    # Add channel dimension to input
    input_tensors = np.expand_dims(input_tensors, axis=1)  # (n, 1, 9, 9, 9)
    
    # Convert to torch tensors
    X = torch.from_numpy(input_tensors).float()
    y = torch.from_numpy(ipc_vectors).float()
    
    # Train/test split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f'\nTrain set size: {X_train.shape[0]}')
    print(f'Test set size: {X_test.shape[0]}')
    
    # Create data loaders
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    test_dataset = torch.utils.data.TensorDataset(X_test, y_test)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    # Initialize model
    print('\nInitializing model...')
    model = IPCEncoder(output_size=output_size)
    model.to(device)
    print(model)
    
    # Train model
    print('\nTraining model...')
    train_losses, test_losses = train_model(model, train_loader, test_loader, epochs=100, device=device)
    
    # Plot training curve
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss', linewidth=2)
    plt.plot(test_losses, label='Test Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Training Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('results/training_curve.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Generate predictions
    model.eval()
    with torch.no_grad():
        predictions = model(X.to(device)).cpu().numpy()
    
    # Plot actual vs predicted IPC curves
    target_size = output_size // 2
    
    for idx in range(len(y)):
        actual = y[idx].numpy()
        predicted = predictions[idx]
        
        actual_pressure = actual[:target_size]
        actual_saturation = actual[target_size:]
        
        pred_pressure = predicted[:target_size]
        pred_saturation = predicted[target_size:]
        
        plt.figure(figsize=(10, 6))
        plt.plot(actual_saturation, actual_pressure, 'b-', linewidth=2, label='Actual', marker='o', markersize=4)
        plt.plot(pred_saturation, pred_pressure, 'r--', linewidth=2, label='Predicted', marker='s', markersize=4)
        plt.xlabel('Saturation')
        plt.ylabel('Pressure')
        plt.title(f'IPC Curve Comparison - Sample {idx}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(f'results/ipc_curve_{idx}.png', dpi=150, bbox_inches='tight')
        plt.show()

