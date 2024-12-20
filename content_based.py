import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os


def parse_args():
    parser = argparse.ArgumentParser(description='Sparse Autoencoder for Movie Recommendations')

    # Model architecture
    parser.add_argument('--input-dim', type=int, default=None,
                        help='Input dimension (will be inferred from data if not provided)')
    parser.add_argument('--encoding-dim', type=int, default=512,
                        help='Dimension of the sparse encoding')
    parser.add_argument('--hidden-dim', type=int, default=256,
                        help='Dimension of the hidden layer')

    # Sparsity parameters
    parser.add_argument('--sparsity-param', type=float, default=0.05,
                        help='Target activation (ρ) for hidden neurons')
    parser.add_argument('--beta', type=float, default=0.05,
                        help='Weight of the sparsity penalty term')

    # Training parameters
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                        help='Learning rate for optimizer')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='Dropout rate')

    # Data and output
    parser.add_argument('--data-path', type=str, default='torch_tensor.pt',
                        help='Path to input tensor file')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory (default: timestamped directory)')

    # Additional parameters
    parser.add_argument('--save-every', type=int, default=10,
                        help='Save visualizations every N epochs')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')

    return parser.parse_args()


class SparseMovieAutoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim,
                 sparsity_param=0.05, beta=3.0, dropout=0.2):
        super().__init__()

        self.encoder = nn.Sequential(
            # nn.BatchNorm1d(input_dim),
            # nn.Linear(input_dim, hidden_dim),
            # nn.ReLU(),
            # nn.BatchNorm1d(hidden_dim),
            # nn.Dropout(dropout),
            nn.Linear(input_dim, encoding_dim),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            # nn.Linear(encoding_dim, hidden_dim),
            # nn.ReLU(),
            # nn.BatchNorm1d(hidden_dim),
            # nn.Dropout(dropout),
            nn.Linear(encoding_dim, input_dim),
            nn.Sigmoid()
        )

        self.sparsity_param = sparsity_param
        self.beta = beta

        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Initialize weights with smaller values to prevent vanishing/exploding gradients
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        self.sparsity_param = sparsity_param
        self.beta = beta
        self.dropout = dropout

    def forward(self, x):
        encoding = self.encoder(x)
        decoding = self.decoder(encoding)
        return encoding, decoding

    def get_sparsity_loss(self, encodings):
        # Original KL divergence
        eps = 1e-10
        rho_hat = torch.mean(encodings, dim=0).clamp(eps, 1 - eps)

        kl_div = self.sparsity_param * torch.log((self.sparsity_param + eps) / rho_hat) + \
                 (1 - self.sparsity_param) * torch.log((1 - self.sparsity_param + eps) / (1 - rho_hat))

        return torch.sum(kl_div)


def get_movie_recommendations(model, reference_movies, movie_features, n_recommendations=5):
    """
    Get recommendations using sparse encodings
    """
    model.eval()
    with torch.no_grad():
        # Get encodings for reference movies
        ref_encodings, _ = model(movie_features[reference_movies])
        user_profile = torch.mean(ref_encodings, dim=0)

        # Get encodings for all movies
        all_encodings, _ = model(movie_features)

        # Calculate similarities
        similarities = F.cosine_similarity(all_encodings, user_profile.unsqueeze(0))

        # Get top recommendations (excluding reference movies)
        similarities[reference_movies] = float('-inf')
        _, indices = torch.topk(similarities, n_recommendations)

        return indices.tolist()


def analyze_encodings(encodings):
    """Analyze encoding statistics"""
    with torch.no_grad():
        # Move to CPU for analysis
        encodings_cpu = encodings.cpu()

        # Calculate sparsity ratio (proportion of near-zero values)
        sparsity_threshold = 1e-3
        sparsity_ratio = (torch.abs(encodings_cpu) < sparsity_threshold).float().mean().item()

        # Calculate average activation
        avg_activation = torch.mean(torch.abs(encodings_cpu)).item()

        # Calculate standard deviation across features
        encoding_std = torch.std(encodings_cpu, dim=0).mean().item()

        return sparsity_ratio, avg_activation, encoding_std


def plot_training_history():
    """Plot training metrics"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Loss plots
    axes[0, 0].plot(training_history['total_loss'], label='Total Loss')
    axes[0, 0].plot(training_history['reconstruction_loss'], label='Reconstruction Loss')
    axes[0, 0].plot(training_history['sparsity_loss'], label='Sparsity Loss')
    axes[0, 0].set_title('Training Losses')
    axes[0, 0].legend()

    # Sparsity ratio
    axes[0, 1].plot(training_history['sparsity_ratio'])
    axes[0, 1].set_title('Sparsity Ratio')

    # Average activation
    axes[1, 0].plot(training_history['avg_activation'])
    axes[1, 0].set_title('Average Activation')

    # Feature STD
    axes[1, 1].plot(training_history['encoding_std'])
    axes[1, 1].set_title('Feature Standard Deviation')

    plt.tight_layout()
    plt.savefig(f'{args.output_dir}/training_history.png')
    plt.close()


def visualize_encodings(encodings, epoch):
    """Create visualizations for encodings"""
    with torch.no_grad():
        encodings_cpu = encodings.cpu().numpy()

        # Activation heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(encodings_cpu[:100, :100], cmap='viridis')
        plt.title(f'Encoding Activation Heatmap (Epoch {epoch})')
        plt.savefig(f'{args.output_dir}/heatmap_epoch_{epoch}.png')
        plt.close()

        # Activation distribution
        plt.figure(figsize=(10, 6))
        plt.hist(encodings_cpu.flatten(), bins=50)
        plt.title(f'Encoding Value Distribution (Epoch {epoch})')
        plt.savefig(f'{args.output_dir}/distribution_epoch_{epoch}.png')
        plt.close()


if __name__ == '__main__':

    # Parse arguments
    args = parse_args()

    # Set random seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # Create output directory
    if args.output_dir is None:
        args.output_dir = f'autoencoder_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    os.makedirs(args.output_dir, exist_ok=True)

    # Save configuration
    with open(os.path.join(args.output_dir, 'config.txt'), 'w') as f:
        for arg in vars(args):
            f.write(f'{arg}: {getattr(args, arg)}\n')

    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Data loading
    print("Loading data...")
    X_tensor = torch.load(args.data_path)
    X_tensor = X_tensor.to(device)

    if args.input_dim is None:
        args.input_dim = X_tensor.shape[1]

    # Model setup
    sae = SparseMovieAutoencoder(
        input_dim=args.input_dim,
        encoding_dim=args.encoding_dim,
        hidden_dim=args.hidden_dim,
        sparsity_param=args.sparsity_param,
        beta=args.beta,
        dropout=args.dropout
    )
    sae = sae.to(device)

    # Training setup
    dataset = TensorDataset(X_tensor)
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    optimizer = torch.optim.Adam(sae.parameters(), lr=args.learning_rate)
    reconstruction_criterion = nn.MSELoss()

    # Tracking metrics
    training_history = {
        'total_loss': [],
        'reconstruction_loss': [],
        'sparsity_loss': [],
        'avg_activation': [],
        'sparsity_ratio': [],
        'encoding_std': []
    }

    # Training loop
    print("Starting training...")
    for epoch in range(args.epochs):
        total_loss = 0
        recon_loss = 0
        sparse_loss = 0
        epoch_encodings = []

        for batch in train_loader:
            optimizer.zero_grad()

            # Forward pass
            encodings, reconstructions = sae(batch[0])
            epoch_encodings.append(encodings.detach())

            # Calculate losses
            reconstruction_loss = reconstruction_criterion(reconstructions, batch[0])
            sparsity_loss = sae.get_sparsity_loss(encodings)
            loss = reconstruction_loss + sae.beta * sparsity_loss

            # Inside training loop, before loss.backward()
            if torch.isnan(loss):
                print(f"NaN detected!")
                print(f"Reconstruction loss: {reconstruction_loss.item()}")
                print(f"Sparsity loss: {sparsity_loss.item()}")
                print(f"Encoding stats - min: {encodings.min()}, max: {encodings.max()}, mean: {encodings.mean()}")

            # Backward pass
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            recon_loss += reconstruction_loss.item()
            sparse_loss += sparsity_loss.item()

        # Combine all batch encodings
        epoch_encodings = torch.cat(epoch_encodings, dim=0)
        sparsity_ratio, avg_activation, encoding_std = analyze_encodings(epoch_encodings)

        # Store metrics
        training_history['total_loss'].append(total_loss / len(train_loader))
        training_history['reconstruction_loss'].append(recon_loss / len(train_loader))
        training_history['sparsity_loss'].append(sparse_loss / len(train_loader))
        training_history['sparsity_ratio'].append(sparsity_ratio)
        training_history['avg_activation'].append(avg_activation)
        training_history['encoding_std'].append(encoding_std)

        # Print progress
        print(f"Epoch {epoch}: Loss={total_loss / len(train_loader):.4f}, "
              f"Sparsity Ratio={sparsity_ratio:.4f}, "
              f"Avg Activation={avg_activation:.4f}")

        # Create visualizations every 10 epochs
        if epoch % 10 == 0:
            visualize_encodings(epoch_encodings, epoch)
            plot_training_history()

        # Save final encodings
        if epoch == 99:
            torch.save(epoch_encodings.cpu(), f'{args.output_dir}/final_encodings.pt')
            np.save(f'{args.output_dir}/final_encodings.npy', epoch_encodings.cpu().numpy())

    print("Training completed. Results saved in:", args.output_dir)


# ##################### From Notebooks #####################
def one_hot_encode(df, column_name, prefix, keeponly=100):
    all_cat = {}
    for cat in df[column_name]:

        if isinstance(cat, str):
            cat = [el.strip() for el in cat.split(',')]

        if not isinstance(cat, list):
            continue
        for el in cat:
            if el not in all_cat:
                all_cat[el] = 0
            all_cat[el] += 1

    sorted_cat = {k: v for k, v in sorted(all_cat.items(), reverse=True, key=lambda item: item[1])}

    all_cat = {v[0]: v[1] for v in (list(sorted_cat.items())[:keeponly])}

    print(f' The dataframe has {len(all_cat)} different categories of {column_name}. Let\' one hot encode !')
    columns = []

    for cat in all_cat:
        columns.append(pd.DataFrame(
            {f'{prefix}_{cat}': df[column_name].apply(lambda x: 1 if (not isinstance(x, float) and cat in x) else 0)}))

    return all_cat, pd.concat([df, *columns], axis=1)


def normalize_without_nan(df, column_name, nan_val=None, clip=None):
    mask = df[column_name].isna()
    if nan_val is not None:
        mask = mask | (df[column_name] == nan_val)
    mask = ~mask

    col = df.loc[mask, column_name]

    if clip is not None:
        col = np.clip(col, np.mean(col) - clip * np.std(col), np.mean(col) + clip * np.std(col))

    df.loc[mask, column_name] = (col - col.mean()) / col.std()

    df.fillna({column_name: 0}, inplace=True)

    return df[column_name]


def analyze_encodings(encodings):
    """Analyze encoding statistics"""
    with torch.no_grad():
        # Move to CPU for analysis
        encodings_cpu = encodings.cpu()

        # Calculate sparsity ratio (proportion of near-zero values)
        sparsity_threshold = 1e-3
        sparsity_ratio = (torch.abs(encodings_cpu) < sparsity_threshold).float().mean().item()

        # Calculate average activation
        avg_activation = torch.mean(torch.abs(encodings_cpu)).item()

        # Calculate standard deviation across features
        encoding_std = torch.std(encodings_cpu, dim=0).mean().item()

        return sparsity_ratio, avg_activation, encoding_std

