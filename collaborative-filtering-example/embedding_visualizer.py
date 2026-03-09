import numpy as np

try:
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE
    DEPENDENCIES_MET = True
except ImportError:
    DEPENDENCIES_MET = False

class EmbeddingVisualizer:
    """
    Visualizes user and item embedding matrices from Matrix Factorization
    using t-SNE for dimensionality reduction to 2D.
    """
    
    def __init__(self, random_state=42, perplexity=5):
        self.random_state = random_state
        self.perplexity = perplexity
        
    def visualize(self, P, U, user_names, item_names, layout='joint', save_path=None):
        """
        P: User matrix (m x d) - list of lists or numpy array
        U: Item matrix (d x n) - list of lists or numpy array
        user_names: list of m strings
        item_names: list of n strings
        layout: 'joint' (same plot) or 'separate' (two subplots)
        """
        if not DEPENDENCIES_MET:
            print("ERROR: EmbeddingVisualizer requires 'matplotlib' and 'scikit-learn'.")
            print("Please install them via: pip install matplotlib scikit-learn")
            return
            
        # Convert to numpy arrays if not already
        P_np = np.array(P)
        U_np = np.array(U)
        
        # U is (d x n), we need it to be (n x d) to match P for concatenation
        U_np_t = U_np.T
        
        if layout == 'joint':
            self._plot_joint(P_np, U_np_t, user_names, item_names, save_path)
        elif layout == 'separate':
            self._plot_separate(P_np, U_np_t, user_names, item_names, save_path)
        else:
            print(f"Unknown layout: {layout}")
            
    def _plot_joint(self, P_np, U_np_t, user_names, item_names, save_path):
        # Stack matrices: Users first, then Items
        combined = np.vstack([P_np, U_np_t])
        
        # Adjust perplexity if dataset is very small
        n_samples = combined.shape[0]
        actual_perplexity = min(self.perplexity, max(1, n_samples - 1))
        
        tsne = TSNE(n_components=2, perplexity=actual_perplexity, 
                    random_state=self.random_state, init='pca', learning_rate='auto')
        embeddings_2d = tsne.fit_transform(combined)
        
        user_2d = embeddings_2d[:len(user_names)]
        item_2d = embeddings_2d[len(user_names):]
        
        plt.figure(figsize=(10, 8))
        
        # Plot Users (Blue Circles)
        plt.scatter(user_2d[:, 0], user_2d[:, 1], c='blue', marker='o', s=100, label='Users', alpha=0.7)
        for i, name in enumerate(user_names):
            plt.annotate(name, (user_2d[i, 0], user_2d[i, 1]), xytext=(5, 5), textcoords='offset points', color='darkblue')
            
        # Plot Items (Red Squares)
        plt.scatter(item_2d[:, 0], item_2d[:, 1], c='red', marker='s', s=100, label='Movies', alpha=0.7)
        for i, name in enumerate(item_names):
            plt.annotate(name, (item_2d[i, 0], item_2d[i, 1]), xytext=(5, -15), textcoords='offset points', color='darkred')
            
        plt.title('Joint Latent Space of Users and Movies (t-SNE)')
        plt.xlabel('t-SNE Dimension 1')
        plt.ylabel('t-SNE Dimension 2')
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.legend()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            print(f"Saved visualization to {save_path}")
        else:
            plt.show()
            
    def _plot_separate(self, P_np, U_np_t, user_names, item_names, save_path):
        # Even when separate, doing t-SNE together ensures they share the same relative manifold projection,
        # but plotting separately prevents label collision.
        combined = np.vstack([P_np, U_np_t])
        
        n_samples = combined.shape[0]
        actual_perplexity = min(self.perplexity, max(1, n_samples - 1))
        
        tsne = TSNE(n_components=2, perplexity=actual_perplexity, 
                    random_state=self.random_state, init='pca', learning_rate='auto')
        embeddings_2d = tsne.fit_transform(combined)
        
        user_2d = embeddings_2d[:len(user_names)]
        item_2d = embeddings_2d[len(user_names):]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
        
        # Plot Users
        ax1.scatter(user_2d[:, 0], user_2d[:, 1], c='blue', marker='o', s=100, alpha=0.7)
        for i, name in enumerate(user_names):
            ax1.annotate(name, (user_2d[i, 0], user_2d[i, 1]), xytext=(5, 5), textcoords='offset points')
        ax1.set_title('User Embedding Space')
        ax1.grid(True, linestyle='--', alpha=0.3)
        
        # Plot Items
        ax2.scatter(item_2d[:, 0], item_2d[:, 1], c='red', marker='s', s=100, alpha=0.7)
        for i, name in enumerate(item_names):
            ax2.annotate(name, (item_2d[i, 0], item_2d[i, 1]), xytext=(5, 5), textcoords='offset points')
        ax2.set_title('Movie Embedding Space')
        ax2.grid(True, linestyle='--', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            print(f"Saved visualization to {save_path}")
        else:
            plt.show()
