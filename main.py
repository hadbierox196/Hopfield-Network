"""
Assignment 3: Hopfield Network Implementation
Clean code without extensive comments
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from scipy.spatial.distance import hamming
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-whitegrid')
np.random.seed(42)

print("="*70)
print("HOPFIELD NETWORK IMPLEMENTATION")
print("="*70)

#%% PART 1: HOPFIELD NETWORK CLASS

class HopfieldNetwork:
    def __init__(self, n_neurons):
        self.n_neurons = n_neurons
        self.weights = np.zeros((n_neurons, n_neurons))
        self.patterns = []
        
    def train(self, patterns):
        patterns = np.array(patterns)
        self.patterns = patterns
        n_patterns = len(patterns)
        
        patterns_bipolar = 2 * patterns - 1
        self.weights = np.zeros((self.n_neurons, self.n_neurons))
        
        for pattern in patterns_bipolar:
            self.weights += np.outer(pattern, pattern)
        
        self.weights /= n_patterns
        np.fill_diagonal(self.weights, 0)
        
        print(f"✓ Trained network with {n_patterns} patterns")
        
    def energy(self, state):
        state_bipolar = 2 * state - 1
        energy = -0.5 * np.dot(state_bipolar, np.dot(self.weights, state_bipolar))
        return energy
    
    def update_neuron(self, state, i):
        state_bipolar = 2 * state - 1
        activation = np.dot(self.weights[i], state_bipolar)
        return 1 if activation >= 0 else 0
    
    def retrieve(self, probe, max_iter=100, mode='async'):
        state = np.array(probe).copy()
        energy_trajectory = [self.energy(state)]
        
        for iteration in range(max_iter):
            old_state = state.copy()
            
            if mode == 'async':
                order = np.random.permutation(self.n_neurons)
                for i in order:
                    state[i] = self.update_neuron(state, i)
            else:
                for i in range(self.n_neurons):
                    state[i] = self.update_neuron(old_state, i)
            
            energy_trajectory.append(self.energy(state))
            
            if np.array_equal(state, old_state):
                return state, True, iteration + 1, energy_trajectory
        
        return state, False, max_iter, energy_trajectory
    
    def pattern_overlap(self, state, pattern):
        return 1 - hamming(state, pattern)


#%% PART 2: BASIC DEMONSTRATION

def create_demo_patterns(n_neurons=100, n_patterns=3):
    patterns = []
    for i in range(n_patterns):
        pattern = (np.random.rand(n_neurons) < 0.3).astype(int)
        patterns.append(pattern)
    return np.array(patterns)

n_neurons = 100
demo_patterns = create_demo_patterns(n_neurons, n_patterns=3)

hopfield = HopfieldNetwork(n_neurons)
hopfield.train(demo_patterns)

print("\nStored patterns:")
for i, pattern in enumerate(demo_patterns):
    print(f"  Pattern {i+1}: {np.sum(pattern)}/{n_neurons} neurons active")

print("\nTesting retrieval with noisy cues:")
test_pattern_idx = 0
original = demo_patterns[test_pattern_idx]

noise_levels = [0.1, 0.2, 0.3, 0.4]

for noise_level in noise_levels:
    noisy = original.copy()
    n_flips = int(noise_level * n_neurons)
    flip_indices = np.random.choice(n_neurons, n_flips, replace=False)
    noisy[flip_indices] = 1 - noisy[flip_indices]
    
    retrieved, converged, iterations, energy = hopfield.retrieve(noisy)
    overlap = hopfield.pattern_overlap(retrieved, original)
    
    print(f"\nNoise level: {noise_level*100:.0f}%")
    print(f"  Initial overlap: {hopfield.pattern_overlap(noisy, original):.3f}")
    print(f"  Retrieved overlap: {overlap:.3f}")
    print(f"  Converged: {converged} (in {iterations} iterations)")


#%% PART 3: NOISE ROBUSTNESS ANALYSIS

def test_noise_robustness(n_neurons=100, n_patterns=5, n_trials=20):
    patterns = create_demo_patterns(n_neurons, n_patterns)
    hopfield = HopfieldNetwork(n_neurons)
    hopfield.train(patterns)
    
    noise_levels = np.linspace(0, 0.5, 11)
    results = {
        'noise_levels': noise_levels,
        'mean_overlap': [],
        'std_overlap': [],
        'convergence_rate': []
    }
    
    print(f"\nTesting with {n_patterns} patterns, {n_trials} trials per noise level...")
    
    for noise_level in tqdm(noise_levels, desc="Noise levels"):
        overlaps = []
        converged_count = 0
        
        for trial in range(n_trials):
            pattern_idx = np.random.randint(n_patterns)
            original = patterns[pattern_idx]
            
            noisy = original.copy()
            n_flips = int(noise_level * n_neurons)
            if n_flips > 0:
                flip_indices = np.random.choice(n_neurons, n_flips, replace=False)
                noisy[flip_indices] = 1 - noisy[flip_indices]
            
            retrieved, converged, _, _ = hopfield.retrieve(noisy, max_iter=100)
            overlap = hopfield.pattern_overlap(retrieved, original)
            overlaps.append(overlap)
            
            if converged:
                converged_count += 1
        
        results['mean_overlap'].append(np.mean(overlaps))
        results['std_overlap'].append(np.std(overlaps))
        results['convergence_rate'].append(converged_count / n_trials)
    
    return results

noise_results = test_noise_robustness(n_neurons=100, n_patterns=5, n_trials=20)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.errorbar(noise_results['noise_levels'] * 100, 
             noise_results['mean_overlap'],
             yerr=noise_results['std_overlap'],
             marker='o', capsize=5, linewidth=2, markersize=8,
             color='steelblue', label='Mean ± SD')

ax1.axhline(y=1.0, color='green', linestyle='--', alpha=0.5, label='Perfect recall')
ax1.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Chance level')
ax1.set_xlabel('Noise Level (%)', fontsize=12)
ax1.set_ylabel('Retrieval Overlap', fontsize=12)
ax1.set_title('Recall Accuracy vs. Noise Level\n(5 patterns, 100 neurons)', fontsize=13, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_ylim(-0.05, 1.05)

ax2.plot(noise_results['noise_levels'] * 100,
         noise_results['convergence_rate'],
         marker='s', linewidth=2, markersize=8, color='coral')
ax2.set_xlabel('Noise Level (%)', fontsize=12)
ax2.set_ylabel('Convergence Rate', fontsize=12)
ax2.set_title('Network Convergence vs. Noise', fontsize=13, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.set_ylim(-0.05, 1.05)

plt.tight_layout()
plt.savefig('hopfield_noise_robustness.png', dpi=150, bbox_inches='tight')
plt.show()


#%% PART 4: ENERGY LANDSCAPE VISUALIZATION

def visualize_energy_landscape_2D(n_neurons=20):
    print(f"\nCreating 2-pattern network with {n_neurons} neurons...")
    
    pattern1 = np.zeros(n_neurons, dtype=int)
    pattern1[:n_neurons//2] = 1
    
    pattern2 = np.zeros(n_neurons, dtype=int)
    pattern2[n_neurons//2:] = 1
    
    patterns = np.array([pattern1, pattern2])
    
    hopfield = HopfieldNetwork(n_neurons)
    hopfield.train(patterns)
    
    print("\nSampling state space...")
    n_samples = 2000
    states = []
    energies = []
    
    for _ in tqdm(range(n_samples), desc="Sampling states"):
        state = np.random.randint(0, 2, n_neurons)
        states.append(state)
        energies.append(hopfield.energy(state))
    
    states = np.array(states)
    energies = np.array(energies)
    
    for pattern in patterns:
        states = np.vstack([states, pattern])
        energies = np.append(energies, hopfield.energy(pattern))
    
    pca = PCA(n_components=2)
    states_2d = pca.fit_transform(states)
    
    fig = plt.figure(figsize=(15, 5))
    
    ax1 = fig.add_subplot(131)
    scatter = ax1.scatter(states_2d[:, 0], states_2d[:, 1], 
                         c=energies, cmap='viridis_r', 
                         s=20, alpha=0.6)
    
    pattern_2d = pca.transform(patterns)
    ax1.scatter(pattern_2d[:, 0], pattern_2d[:, 1], 
               c='red', s=200, marker='*', 
               edgecolors='black', linewidths=2,
               label='Stored patterns', zorder=5)
    
    ax1.set_xlabel('PC1', fontsize=11)
    ax1.set_ylabel('PC2', fontsize=11)
    ax1.set_title('Energy Landscape (2D Projection)', fontsize=12, fontweight='bold')
    ax1.legend()
    plt.colorbar(scatter, ax=ax1, label='Energy')
    
    ax2 = fig.add_subplot(132)
    
    from scipy.interpolate import griddata
    x_min, x_max = states_2d[:, 0].min() - 1, states_2d[:, 0].max() + 1
    y_min, y_max = states_2d[:, 1].min() - 1, states_2d[:, 1].max() + 1
    
    xi = np.linspace(x_min, x_max, 100)
    yi = np.linspace(y_min, y_max, 100)
    xi, yi = np.meshgrid(xi, yi)
    zi = griddata((states_2d[:, 0], states_2d[:, 1]), energies, 
                  (xi, yi), method='cubic')
    
    contour = ax2.contourf(xi, yi, zi, levels=20, cmap='viridis_r', alpha=0.8)
    ax2.scatter(pattern_2d[:, 0], pattern_2d[:, 1], 
               c='red', s=200, marker='*', 
               edgecolors='black', linewidths=2, zorder=5)
    
    ax2.set_xlabel('PC1', fontsize=11)
    ax2.set_ylabel('PC2', fontsize=11)
    ax2.set_title('Energy Contours', fontsize=12, fontweight='bold')
    plt.colorbar(contour, ax=ax2, label='Energy')
    
    ax3 = fig.add_subplot(133, projection='3d')
    surf = ax3.plot_surface(xi, yi, zi, cmap='viridis_r', 
                           alpha=0.8, edgecolor='none')
    ax3.scatter(pattern_2d[:, 0], pattern_2d[:, 1], 
               [hopfield.energy(p) for p in patterns],
               c='red', s=200, marker='*', 
               edgecolors='black', linewidths=2, zorder=5)
    
    ax3.set_xlabel('PC1', fontsize=10)
    ax3.set_ylabel('PC2', fontsize=10)
    ax3.set_zlabel('Energy', fontsize=10)
    ax3.set_title('3D Energy Surface', fontsize=12, fontweight='bold')
    ax3.view_init(elev=25, azim=45)
    
    plt.tight_layout()
    plt.savefig('hopfield_energy_landscape.png', dpi=150, bbox_inches='tight')
    plt.show()

visualize_energy_landscape_2D(n_neurons=20)


#%% PART 5: CAPACITY ANALYSIS

def test_capacity(n_neurons=100, max_patterns=30, n_trials=10):
    pattern_numbers = np.arange(1, max_patterns + 1)
    results = {
        'n_patterns': pattern_numbers,
        'mean_accuracy': [],
        'std_accuracy': [],
        'mean_overlap': [],
        'std_overlap': []
    }
    
    print(f"\nTesting capacity with {n_neurons} neurons...")
    
    for n_patterns in tqdm(pattern_numbers, desc="Pattern count"):
        trial_accuracies = []
        trial_overlaps = []
        
        for trial in range(n_trials):
            patterns = create_demo_patterns(n_neurons, n_patterns)
            hopfield = HopfieldNetwork(n_neurons)
            hopfield.train(patterns)
            
            overlaps = []
            perfect_recalls = 0
            
            for i, pattern in enumerate(patterns):
                noisy = pattern.copy()
                n_flips = int(0.2 * n_neurons)
                flip_indices = np.random.choice(n_neurons, n_flips, replace=False)
                noisy[flip_indices] = 1 - noisy[flip_indices]
                
                retrieved, _, _, _ = hopfield.retrieve(noisy, max_iter=50)
                overlap = hopfield.pattern_overlap(retrieved, pattern)
                overlaps.append(overlap)
                
                if overlap > 0.95:
                    perfect_recalls += 1
            
            trial_accuracies.append(perfect_recalls / n_patterns)
            trial_overlaps.append(np.mean(overlaps))
        
        results['mean_accuracy'].append(np.mean(trial_accuracies))
        results['std_accuracy'].append(np.std(trial_accuracies))
        results['mean_overlap'].append(np.mean(trial_overlaps))
        results['std_overlap'].append(np.std(trial_overlaps))
    
    return results

capacity_results = test_capacity(n_neurons=100, max_patterns=30, n_trials=10)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.errorbar(capacity_results['n_patterns'], 
             capacity_results['mean_accuracy'],
             yerr=capacity_results['std_accuracy'],
             marker='o', capsize=5, linewidth=2, markersize=6,
             color='darkblue', label='Empirical accuracy')

theoretical_limit = 0.14 * 100
ax1.axvline(x=theoretical_limit, color='red', linestyle='--', 
           linewidth=2, label=f'Theoretical limit (0.14N = {theoretical_limit:.0f})')

ax1.axhline(y=0.8, color='green', linestyle=':', alpha=0.5, 
           label='80% accuracy threshold')

ax1.set_xlabel('Number of Stored Patterns', fontsize=12)
ax1.set_ylabel('Recall Accuracy (20% noise)', fontsize=12)
ax1.set_title('Hopfield Network Capacity\n100 neurons, 20% noise', 
             fontsize=13, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_ylim(-0.05, 1.05)

ax2.errorbar(capacity_results['n_patterns'] / 100,
             capacity_results['mean_overlap'],
             yerr=capacity_results['std_overlap'],
             marker='s', capsize=5, linewidth=2, markersize=6,
             color='darkgreen', label='Mean overlap')

ax2.axvline(x=0.14, color='red', linestyle='--', linewidth=2, 
           label='Theoretical limit (0.14N)')
ax2.axhline(y=0.95, color='blue', linestyle=':', alpha=0.5,
           label='95% overlap threshold')

ax2.set_xlabel('Pattern Load (P/N)', fontsize=12)
ax2.set_ylabel('Mean Retrieval Overlap', fontsize=12)
ax2.set_title('Capacity Curve (Normalized)', fontsize=13, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_ylim(0, 1.05)

plt.tight_layout()
plt.savefig('hopfield_capacity_analysis.png', dpi=150, bbox_inches='tight')
plt.show()

accuracy_threshold = 0.8
empirical_capacity = capacity_results['n_patterns'][
    np.where(np.array(capacity_results['mean_accuracy']) > accuracy_threshold)[0][-1]
]

print("\n" + "="*70)
print("CAPACITY ANALYSIS RESULTS")
print("="*70)
print(f"\nTheoretical capacity (0.14N): {theoretical_limit:.0f} patterns")
print(f"Empirical capacity (80% accuracy): {empirical_capacity} patterns")
print(f"Empirical/Theoretical ratio: {empirical_capacity/theoretical_limit:.2f}")


#%% PART 6: PATTERN INTERFERENCE VISUALIZATION

def visualize_pattern_interference():
    n_neurons = 100
    fig, axes = plt.subplots(3, 4, figsize=(15, 10))
    
    test_loads = [2, 5, 10, 20]
    
    for col, n_patterns in enumerate(test_loads):
        patterns = create_demo_patterns(n_neurons, n_patterns)
        hopfield = HopfieldNetwork(n_neurons)
        hopfield.train(patterns)
        
        test_pattern = patterns[0]
        
        noisy = test_pattern.copy()
        n_flips = int(0.3 * n_neurons)
        flip_indices = np.random.choice(n_neurons, n_flips, replace=False)
        noisy[flip_indices] = 1 - noisy[flip_indices]
        
        retrieved, converged, iterations, energy = hopfield.retrieve(noisy, max_iter=100)
        overlap = hopfield.pattern_overlap(retrieved, test_pattern)
        
        ax = axes[0, col]
        ax.imshow(test_pattern.reshape(10, 10), cmap='binary', interpolation='nearest')
        ax.set_title(f'{n_patterns} patterns\nOriginal', fontsize=10)
        ax.axis('off')
        
        ax = axes[1, col]
        ax.imshow(noisy.reshape(10, 10), cmap='binary', interpolation='nearest')
        ax.set_title(f'Noisy (30%)', fontsize=10)
        ax.axis('off')
        
        ax = axes[2, col]
        ax.imshow(retrieved.reshape(10, 10), cmap='binary', interpolation='nearest')
        color = 'green' if overlap > 0.9 else ('orange' if overlap > 0.7 else 'red')
        ax.set_title(f'Retrieved\nOverlap: {overlap:.2f}', 
                    fontsize=10, color=color, fontweight='bold')
        ax.axis('off')
    
    plt.suptitle('Pattern Interference at Different Network Loads',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('hopfield_interference.png', dpi=150, bbox_inches='tight')
    plt.show()

visualize_pattern_interference()


#%% PART 7: SPARSE HOPFIELD NETWORK (BONUS)

class SparseHopfieldNetwork(HopfieldNetwork):
    def __init__(self, n_neurons, sparsity=0.1):
        super().__init__(n_neurons)
        self.sparsity = sparsity
        
    def train_sparse(self, patterns):
        patterns = np.array(patterns)
        self.patterns = patterns
        n_patterns = len(patterns)
        
        self.weights = np.zeros((self.n_neurons, self.n_neurons))
        mean_activity = np.mean(patterns, axis=0)
        
        for pattern in patterns:
            centered = pattern - mean_activity
            self.weights += np.outer(centered, centered)
        
        self.weights /= n_patterns
        np.fill_diagonal(self.weights, 0)
        
        print(f"✓ Trained sparse network with {n_patterns} patterns")

def compare_sparse_vs_dense():
    n_neurons = 100
    max_patterns = 40
    n_trials = 10
    
    results = {
        'dense': {'n_patterns': [], 'accuracy': []},
        'sparse': {'n_patterns': [], 'accuracy': []}
    }
    
    pattern_numbers = np.arange(2, max_patterns + 1, 2)
    
    print("\nComparing Dense vs Sparse Hopfield Networks...")
    
    for n_patterns in tqdm(pattern_numbers, desc="Testing capacity"):
        dense_accuracies = []
        for trial in range(n_trials):
            patterns_dense = (np.random.rand(n_patterns, n_neurons) < 0.5).astype(int)
            hopfield_dense = HopfieldNetwork(n_neurons)
            hopfield_dense.train(patterns_dense)
            
            perfect = 0
            for pattern in patterns_dense:
                noisy = pattern.copy()
                n_flips = int(0.2 * n_neurons)
                flip_indices = np.random.choice(n_neurons, n_flips, replace=False)
                noisy[flip_indices] = 1 - noisy[flip_indices]
                
                retrieved, _, _, _ = hopfield_dense.retrieve(noisy, max_iter=50)
                if hopfield_dense.pattern_overlap(retrieved, pattern) > 0.95:
                    perfect += 1
            
            dense_accuracies.append(perfect / n_patterns)
        
        sparse_accuracies = []
        for trial in range(n_trials):
            patterns_sparse = (np.random.rand(n_patterns, n_neurons) < 0.1).astype(int)
            hopfield_sparse = SparseHopfieldNetwork(n_neurons, sparsity=0.1)
            hopfield_sparse.train_sparse(patterns_sparse)
            
            perfect = 0
            for pattern in patterns_sparse:
                noisy = pattern.copy()
                n_flips = int(0.2 * n_neurons)
                flip_indices = np.random.choice(n_neurons, n_flips, replace=False)
                noisy[flip_indices] = 1 - noisy[flip_indices]
                
                retrieved, _, _, _ = hopfield_sparse.retrieve(noisy, max_iter=50)
                if hopfield_sparse.pattern_overlap(retrieved, pattern) > 0.95:
                    perfect += 1
            
            sparse_accuracies.append(perfect / n_patterns)
        
        results['dense']['n_patterns'].append(n_patterns)
        results['dense']['accuracy'].append(np.mean(dense_accuracies))
        
        results['sparse']['n_patterns'].append(n_patterns)
        results['sparse']['accuracy'].append(np.mean(sparse_accuracies))
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(results['dense']['n_patterns'], 
           results['dense']['accuracy'],
           marker='o', linewidth=2, markersize=8,
           label='Dense (50% active)', color='blue')
    
    ax.plot(results['sparse']['n_patterns'], 
           results['sparse']['accuracy'],
           marker='s', linewidth=2, markersize=8,
           label='Sparse (10% active)', color='red')
    
    ax.axhline(y=0.8, color='gray', linestyle='--', alpha=0.5,
              label='80% accuracy threshold')
    
    ax.axvline(x=14, color='blue', linestyle=':', alpha=0.5,
              label='Dense limit (0.14N)')
    
    ax.set_xlabel('Number of Stored Patterns', fontsize=12)
    ax.set_ylabel('Recall Accuracy', fontsize=12)
    ax.set_title('Sparse vs. Dense Hopfield Networks\n100 neurons, 20% noise',
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)
    
    plt.tight_layout()
    plt.savefig('hopfield_sparse_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    dense_capacity = results['dense']['n_patterns'][
        np.where(np.array(results['dense']['accuracy']) > 0.8)[0][-1]
    ]
    sparse_capacity = results['sparse']['n_patterns'][
        np.where(np.array(results['sparse']['accuracy']) > 0.8)[0][-1]
    ]
    
    print("\n" + "="*70)
    print("SPARSE vs DENSE COMPARISON")
    print("="*70)
    print(f"\nDense network capacity: {dense_capacity} patterns ({dense_capacity/n_neurons:.2f}N)")
    print(f"Sparse network capacity: {sparse_capacity} patterns ({sparse_capacity/n_neurons:.2f}N)")
    print(f"\nCapacity improvement: {(sparse_capacity/dense_capacity - 1)*100:.0f}%")

compare_sparse_vs_dense()


#%% PART 8: BASINS OF ATTRACTION

def visualize_basins_of_attraction():
    n_neurons = 50
    
    pattern1 = np.zeros(n_neurons, dtype=int)
    pattern1[:n_neurons//3] = 1
    
    pattern2 = np.zeros(n_neurons, dtype=int)
    pattern2[n_neurons//3:2*n_neurons//3] = 1
    
    pattern3 = np.zeros(n_neurons, dtype=int)
    pattern3[2*n_neurons//3:] = 1
    
    patterns = np.array([pattern1, pattern2, pattern3])
    
    hopfield = HopfieldNetwork(n_neurons)
    hopfield.train(patterns)
    
    n_samples = 500
    states = []
    convergence_targets = []
    
    print(f"\nTesting {n_samples} random initial states...")
    
    for _ in tqdm(range(n_samples), desc="Sampling basins"):
        state = np.random.randint(0, 2, n_neurons)
        states.append(state)
        
        retrieved, _, _, _ = hopfield.retrieve(state, max_iter=50)
        
        overlaps = [hopfield.pattern_overlap(retrieved, p) for p in patterns]
        best_match = np.argmax(overlaps)
        convergence_targets.append(best_match if max(overlaps) > 0.7 else -1)
    
    states = np.array(states)
    convergence_targets = np.array(convergence_targets)
    
    pca = PCA(n_components=2)
    states_2d = pca.fit_transform(states)
    patterns_2d = pca.transform(patterns)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = ['red', 'blue', 'green', 'gray']
    labels = ['Pattern 1', 'Pattern 2', 'Pattern 3', 'No convergence']
    
    for i in range(-1, 3):
        mask = convergence_targets == i
        if np.sum(mask) > 0:
            ax.scatter(states_2d[mask, 0], states_2d[mask, 1],
                      c=colors[i], alpha=0.5, s=30,
                      label=labels[i])
    
    ax.scatter(patterns_2d[:, 0], patterns_2d[:, 1],
              c=['red', 'blue', 'green'], s=500, marker='*',
              edgecolors='black', linewidths=3,
              label='Stored patterns', zorder=10)
    
    ax.set_xlabel('PC1', fontsize=12)
    ax.set_ylabel('PC2', fontsize=12)
    ax.set_title('Basins of Attraction\n(Colors show convergence targets)',
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('hopfield_basins_of_attraction.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\n" + "="*70)
    print("BASIN OF ATTRACTION ANALYSIS")
    print("="*70)
    for i in range(3):
        count = np.sum(convergence_targets == i)
        percentage = 100 * count / n_samples
        print(f"Pattern {i+1} basin: {count}/{n_samples} states ({percentage:.1f}%)")

visualize_basins_of_attraction()

print("\n" + "="*70)
print("ALL ANALYSES COMPLETE")
print("="*70)
