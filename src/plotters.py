import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import collections as mc
from .tools import ewma, freeze
import ot
import seaborn as sns

import torch
import gc

def plot_images(X, Y, T):
    freeze(T);
    with torch.no_grad():
        T_X = T(X)
        imgs = torch.cat([X, T_X, Y]).to('cpu').permute(0,2,3,1).mul(0.5).add(0.5).numpy().clip(0,1)

    fig, axes = plt.subplots(3, 10, figsize=(15, 4.5), dpi=150)
    for i, ax in enumerate(axes.flatten()):
        ax.imshow(imgs[i])
        ax.get_xaxis().set_visible(False)
        ax.set_yticks([])
        
    axes[0, 0].set_ylabel('X', fontsize=24)
    axes[1, 0].set_ylabel('T(X)', fontsize=24)
    axes[2, 0].set_ylabel('Y', fontsize=24)
    
    fig.tight_layout(pad=0.001)
    torch.cuda.empty_cache(); gc.collect()
    return fig, axes

def plot_random_images(X_sampler, Y_sampler, T):
    X = X_sampler.sample(10)
    Y = Y_sampler.sample(10)
    return plot_images(X, Y, T)

def plot_Z_images(XZ, Y, T):
    freeze(T);
    with torch.no_grad():
        T_XZ = T(
            XZ.flatten(start_dim=0, end_dim=1)
        ).permute(1,2,3,0).reshape(Y.shape[1], Y.shape[2], Y.shape[3], 10, 4).permute(4,3,0,1,2).flatten(start_dim=0, end_dim=1)
        imgs = torch.cat([XZ[:,0,:Y.shape[1]], T_XZ, Y]).to('cpu').permute(0,2,3,1).mul(0.5).add(0.5).numpy().clip(0,1)

    fig, axes = plt.subplots(6, 10, figsize=(15, 9), dpi=150)
    for i, ax in enumerate(axes.flatten()):
        ax.imshow(imgs[i])
        ax.get_xaxis().set_visible(False)
        ax.set_yticks([])
        
    axes[0, 0].set_ylabel('X', fontsize=24)
    for i in range(4):
        axes[i+1, 0].set_ylabel('T(X,Z)', fontsize=24)
    axes[-1, 0].set_ylabel('Y', fontsize=24)
    
    fig.tight_layout(pad=0.001)
    torch.cuda.empty_cache(); gc.collect()
    return fig, axes

def plot_random_Z_images(X_sampler, ZC, Z_STD, Y_sampler, T):
    X = X_sampler.sample(10)[:,None].repeat(1,4,1,1,1)
    with torch.no_grad():
        Z = torch.randn(10, 4, ZC, X.size(3), X.size(4), device='cuda') * Z_STD
        XZ = torch.cat([X, Z], dim=2)
    X = X_sampler.sample(10)
    Y = Y_sampler.sample(10)
    return plot_Z_images(XZ, Y, T)


def plot_bar_and_stochastic_2D(X_sampler, Y_sampler, T, ZD, Z_STD, plot_discrete=True):
    DIM = 2
    freeze(T)
    
    DISCRETE_OT = 1024
    
    PLOT_X_SIZE_LEFT = 64
    PLOT_Z_COMPUTE_LEFT = 256

    PLOT_X_SIZE_RIGHT = 32
    PLOT_Z_SIZE_RIGHT = 4
    
    assert PLOT_Z_COMPUTE_LEFT >= PLOT_Z_SIZE_RIGHT
    assert PLOT_X_SIZE_LEFT >= PLOT_X_SIZE_RIGHT
    assert DISCRETE_OT >= PLOT_X_SIZE_LEFT
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2), dpi=150, sharex=True, sharey=True, )
    for i in range(2):
        axes[i].set_xlim(-2.5, 2.5); axes[i].set_ylim(-2.5, 2.5)
        
    axes[0].set_title(r'Map $x\mapsto \overline{T}(x)=\int_{\mathcal{Z}}T(x,z)d\mathbb{S}(z)$', fontsize=22, pad=10)
    axes[1].set_title(r'Stochastic map $x\mapsto T(x,z)$', fontsize=20, pad=10)    
    axes[2].set_title(r'DOT map $x\mapsto \int y d\pi^{*}(y|x)$', fontsize=18, pad=10)
    
    # Computing and plotting discrete OT bar map
    X, Y = X_sampler.sample(DISCRETE_OT), Y_sampler.sample(DISCRETE_OT)
    
    if plot_discrete:
        X_np, Y_np = X.cpu().numpy(), Y.cpu().numpy()
        pi = ot.weak.weak_optimal_transport(X_np, Y_np)
        T_X_bar_np = pi @ Y_np * len(X)

        lines = list(zip(X_np[:PLOT_X_SIZE_LEFT], T_X_bar_np[:PLOT_X_SIZE_LEFT]))
        lc = mc.LineCollection(lines, linewidths=1, color='black')
        axes[2].add_collection(lc)
        axes[2].scatter(
            X_np[:PLOT_X_SIZE_LEFT, 0], X_np[:PLOT_X_SIZE_LEFT, 1], c='darkseagreen', edgecolors='black',
            zorder=2, label=r'$x\sim\mathbb{P}$'
        )
        axes[2].scatter(
            T_X_bar_np[:PLOT_X_SIZE_LEFT, 0], T_X_bar_np[:PLOT_X_SIZE_LEFT, 1],
            c='slateblue', edgecolors='black', zorder=2, label=r'$\overline{T}(x)$', marker='v'
        )
        axes[2].legend(fontsize=16, loc='lower right', framealpha=1)
    
    # Our method results
    with torch.no_grad():
        X = X[:PLOT_X_SIZE_LEFT].reshape(-1, 1, DIM).repeat(1, PLOT_Z_COMPUTE_LEFT, 1)
        Y = Y[:PLOT_X_SIZE_LEFT]
        
        Z = torch.randn(PLOT_X_SIZE_LEFT, PLOT_Z_COMPUTE_LEFT, ZD, device='cuda') * Z_STD
        XZ = torch.cat([X, Z], dim=2)
        T_XZ = T(
            XZ.flatten(start_dim=0, end_dim=1)
        ).permute(1, 0).reshape(DIM, -1, PLOT_Z_COMPUTE_LEFT).permute(1, 2, 0)

    X_np = X[:, 0].cpu().numpy()
    Y_np = Y.cpu().numpy()
    T_XZ_np = T_XZ.cpu().numpy()

    lines = list(zip(X_np[:PLOT_X_SIZE_LEFT], T_XZ_np.mean(axis=1)[:PLOT_X_SIZE_LEFT]))
    lc = mc.LineCollection(lines, linewidths=1, color='black')
    axes[0].add_collection(lc)
    axes[0].scatter(
        X_np[:PLOT_X_SIZE_LEFT, 0], X_np[:PLOT_X_SIZE_LEFT, 1], c='darkseagreen', edgecolors='black',
        zorder=2, label=r'$x\sim\mathbb{P}$'
    )
    axes[0].scatter(
        T_XZ_np.mean(axis=1)[:PLOT_X_SIZE_LEFT, 0], T_XZ_np.mean(axis=1)[:PLOT_X_SIZE_LEFT, 1],
        c='tomato', edgecolors='black', zorder=2, label=r'$\overline{T}(x)$', marker='v'
    )
    axes[0].legend(fontsize=16, loc='lower right', framealpha=1)

    lines = []
    for i in range(PLOT_X_SIZE_RIGHT):
        for j in range(PLOT_Z_SIZE_RIGHT):
            lines.append((X_np[i], T_XZ_np[i, j]))
    lc = mc.LineCollection(lines, linewidths=0.5, color='black')
    axes[1].add_collection(lc)
    axes[1].scatter(
        X_np[:PLOT_X_SIZE_RIGHT, 0], X_np[:PLOT_X_SIZE_RIGHT, 1], c='darkseagreen', edgecolors='black',
        zorder=2,  label=r'$x\sim\mathbb{P}$'
    )
    axes[1].scatter(
        T_XZ_np[:PLOT_X_SIZE_RIGHT, :PLOT_Z_SIZE_RIGHT, 0].flatten(),
        T_XZ_np[:PLOT_X_SIZE_RIGHT, :PLOT_Z_SIZE_RIGHT, 1].flatten(),
        c='wheat', edgecolors='black', zorder=3,  label=r'$T(x,z)$'
    )
    axes[1].legend(fontsize=16, loc='lower right', framealpha=1)

    fig.tight_layout()
    return fig, axes

def plot_generated_2D(X_sampler, Y_sampler, T, ZD, Z_STD):
    DIM = 2
    freeze(T)

    PLOT_SIZE = 512
    X = X_sampler.sample(PLOT_SIZE).reshape(-1, 1, DIM).repeat(1, 1, 1)
    Y = Y_sampler.sample(PLOT_SIZE)

    with torch.no_grad():
        Z = torch.randn(PLOT_SIZE, 1, ZD, device='cuda') * Z_STD
        XZ = torch.cat([X, Z], dim=2)
        T_XZ = T(
            XZ.flatten(start_dim=0, end_dim=1)
        ).permute(1, 0).reshape(DIM, -1, 1).permute(1, 2, 0)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.4), sharex=True, sharey=True, dpi=150)

    X_np = X[:,0].cpu().numpy()
    Y_np = Y.cpu().numpy()
    T_XZ_np = T_XZ[:,0].cpu().numpy()

    for i in range(3):
        axes[i].set_xlim(-2.5, 2.5); axes[i].set_ylim(-2.5, 2.5)
        axes[i].grid(True)

    axes[0].scatter(X_np[:, 0], X_np[:, 1], c='darkseagreen', edgecolors='black')
    axes[1].scatter(Y_np[:, 0], Y_np[:, 1], c='peru', edgecolors='black')
    axes[2].scatter(T_XZ_np[:, 0], T_XZ_np[:, 1], c='wheat', edgecolors='black')

    axes[0].set_title(r'Input $x\sim\mathbb{P}$', fontsize=22, pad=10)
    axes[1].set_title(r'Target $y\sim\mathbb{Q}$', fontsize=22, pad=10)
    axes[2].set_title(r'Fitted $T(x,z)_{\#}(\mathbb{P}\times\mathbb{S})$', fontsize=22, pad=10)

    fig.tight_layout()
    return fig, axes

def plot_1D(X_sampler, Y_sampler, T, ZD, Z_STD, num_samples=1024):
    DIM = 1; freeze(T)
    
    X, Y = X_sampler.sample(num_samples), Y_sampler.sample(num_samples)
    X_np, Y_np = X.cpu().numpy(), Y.cpu().numpy()
    
    fig, axes = plt.subplots(1, 4, figsize=(12, 3), dpi=150)
    
    axes[2].set_xlim(-2.5, 2.5); axes[2].set_ylim(-2.5, 2.5)
    
    axes[0].set_axisbelow(True); axes[0].grid(axis='x')
    axes[1].set_axisbelow(True); axes[1].grid(axis='y')
    axes[3].set_axisbelow(True); axes[3].grid(axis='y')
    axes[0].set_xlim(-2.5, 2.5); axes[0].set_ylim(0, 0.7)
    axes[1].set_ylim(-2.5, 2.5); axes[1].set_xlim(0, 0.7)
    axes[3].set_ylim(-2.5, 2.5); axes[3].set_xlim(0, 0.7)
    
    # Plotting X
    sns.kdeplot(
        X_np[:, 0], color='darkseagreen', shade=True,
        edgecolor='black', alpha=0.95,
        ax=axes[0], label=r'$x\sim\mathbb{P}$'
    )  
    axes[0].legend(fontsize=12, loc='upper left', framealpha=1)
#     axes[0].set_xlabel(r"$x$", fontsize=12)
    axes[0].set_title(r"Input $\mathbb{P}$ (1D)", fontsize=14)
    
    # Plotting Y
    sns.kdeplot(
        y=Y_np[:, 0], color='wheat', shade=True,
        edgecolor='black', alpha=0.95,
        ax=axes[1], label=r'$y\sim\mathbb{Q}$'
    )
    axes[1].legend(fontsize=12, loc='upper left', framealpha=1)   
#     axes[1].set_ylabel(r"$y$", fontsize=12)
    axes[1].set_title(r"Target $\mathbb{Q}$ (1D)", fontsize=14)
        
    # Computing and plotting our OT map
    with torch.no_grad():
        X = X.reshape(-1, 1, DIM).repeat(1, 1, 1)
        Z = torch.randn(X.size(0), 1, ZD, device='cuda') * Z_STD
        XZ = torch.cat([X, Z], dim=2)
        T_XZ = T(
            XZ.flatten(start_dim=0, end_dim=1)
        )
    
    X_np = X[:, 0].cpu().numpy()
    T_XZ_np = T_XZ.cpu().numpy()
    
    sns.kdeplot(
        X_np[:, 0], T_XZ_np[:, 0], xlim=(-2.5, 2.5), ylim=(-2.5, 2.5),
        color='black', alpha=1., ax=axes[2],
        label=r'$(x,\hat{T}(x,z))\sim \hat{\pi}$'
    )
    sns.kdeplot(
        X_np[:, 0], T_XZ_np[:, 0], xlim=(-2.5, 2.5), ylim=(-2.5, 2.5),
        color='white', alpha=.3, ax=axes[2], shade=True
    )
    axes[2].set_title(r"Learned $\hat{\pi}$ (2D), ours", fontsize=14)
    
    # Plotting T(X,Z)
    sns.kdeplot(
        y=T_XZ_np[:, 0], color='sandybrown', shade=True,
        edgecolor='black', alpha=0.95,
        ax=axes[3], label=r'$T(x,z)\sim T_{\sharp}(\mathbb{P}\times\mathbb{S})$'
    )
    axes[3].legend(fontsize=12, loc='upper left', framealpha=1)   
#     axes[1].set_ylabel(r"$y$", fontsize=12)
    axes[3].set_title(r"Mapped $T_{\sharp}(\mathbb{P}\times\mathbb{S})$ (1D)", fontsize=14)
    
    # Computing and plotting our bar. proj
    X = X_sampler.sample(num_samples)
    with torch.no_grad():
        Z_SIZE = 16
        X = X.reshape(-1, 1, DIM).repeat(1, Z_SIZE, 1)
        Z = torch.randn(X.size(0), Z_SIZE, ZD, device='cuda') * Z_STD
        XZ = torch.cat([X, Z], dim=2)
        T_XZ = T(
            XZ.flatten(start_dim=0, end_dim=1)
        ).reshape(-1, Z_SIZE, 1)
    
    X_np = X[:, 0].cpu().numpy()
    T_bar_np = T_XZ.mean(dim=1).cpu().numpy()
    X_T_bar = np.concatenate([X_np, T_bar_np], axis=1)
    X_T_bar.sort(axis=0)
    axes[2].plot(
        X_T_bar[:,0], X_T_bar[:,1], color='sandybrown', 
        linewidth=3, label=r'$x\mapsto \overline{T}(x)$'
    )
    axes[2].legend(fontsize=12, loc='upper left', framealpha=1)

    fig.tight_layout(pad=0.01)
    
    return fig, axes


def plot_1D_discrete(X_sampler, Y_sampler, num_samples=1024):
    DIM = 1;
    
    fig, axes = plt.subplots(1, 4, figsize=(12, 3), dpi=150)
    for i in range(4):
        axes[i].set_xlim(-2.5, 2.5); axes[i].set_ylim(-2.5, 2.5)
#         axes[i].set_axisbelow(True)
        axes[i].grid(True)
    
    for j in range(4):
        # Computing and plotting discrete OT bar map
        X, Y = X_sampler.sample(num_samples), Y_sampler.sample(num_samples)
        X_np, Y_np = X.cpu().numpy(), Y.cpu().numpy()
          
        G0 = np.abs(np.random.normal(size=(len(X_np), len(Y_np))))
        G0 /= G0.sum()
                        
        pi = ot.weak.weak_optimal_transport(X_np, Y_np, G0=G0)
        T_X_bar_np = pi @ Y_np * len(X_np)

        idx, XY = [], []
        for i in range(num_samples):
            idx.append(np.random.choice(list(range(len(Y))), p=pi[i]/pi[i].sum()))
            XY.append((X_np[i][0], Y_np[idx[-1]][0]))
        XY = np.array(XY)
        sns.kdeplot(
            XY[:, 0], XY[:, 1], xlim=(-2.5, 2.5), ylim=(-2.5, 2.5),
            color='black', alpha=1., ax=axes[j], label=r'$(x,y)\sim\pi^{*}$'
        ) 
        sns.kdeplot(
            XY[:, 0], XY[:, 1], xlim=(-2.5, 2.5), ylim=(-2.5, 2.5),
            color='lemonchiffon', alpha=0.5, ax=axes[j], shade=True
        )
        
        
        X_T_bar = np.concatenate([X_np, T_X_bar_np], axis=1)
        X_T_bar.sort(axis=0)
        axes[j].plot(
            X_T_bar[:,0], X_T_bar[:,1], color='orangered', 
            linewidth=3, label=r'$x\mapsto \nabla\psi^{*}(x)$'
        )
             
        axes[j].legend(fontsize=12, loc='upper left', framealpha=1)
        axes[j].set_title(r"DOT plan $\pi^{*}$ (2D)", fontsize=14)

    fig.tight_layout(pad=0.01)
    
    return fig, axes