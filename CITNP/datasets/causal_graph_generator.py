"""
File that contains the class that generates causal graphs.
"""

import igraph as ig
import numpy as np


def generate_synthetic_dag(d, s0, graph_type):
    """Simulate random DAG with some expected number of edges.

    Args:
        d (int): num of nodes
        s0 (int): expected num of edges
        graph_type (str): ER, SF, BP

    Returns:
        B (np.ndarray): [d, d] binary adj matrix of DAG, where
        B[i, j] = 1 if i -> j
    """

    def _random_permutation(M):
        # np.random.permutation permutes first axis only
        P = np.random.permutation(np.eye(M.shape[0]))
        return P.T @ M @ P

    def _random_acyclic_orientation(B_und):
        return np.tril(_random_permutation(B_und), k=-1)

    def _graph_to_adjmat(G):
        return np.array(G.get_adjacency().data)

    if graph_type == "ER":
        # Erdos-Renyi
        G_und = ig.Graph.Erdos_Renyi(n=d, m=s0)
        B_und = _graph_to_adjmat(G_und)
        B = _random_acyclic_orientation(B_und)
    elif graph_type == "SF":
        # Scale-free, Barabasi-Albert
        G = ig.Graph.Barabasi(n=d, m=int(round(s0 / d)), directed=True)
        B = _graph_to_adjmat(G)
    elif graph_type == "BP":
        # Bipartite, Sec 4.1 of (Gu, Fu, Zhou, 2018)
        top = int(0.2 * d)
        G = ig.Graph.Random_Bipartite(top, d - top, m=s0, directed=True, neimode=ig.OUT)
        B = _graph_to_adjmat(G)
    elif graph_type == "FC":
        # Fully connected DAG
        M = np.zeros((d, d))
        M[np.triu_indices(d, k=1)] = 1
        B = _random_permutation(M)

    else:
        raise ValueError("unknown graph type")
    # Make B upper triangular
    B_perm = B.T
    assert ig.Graph.Adjacency(B_perm.tolist()).is_dag()
    return B_perm


if __name__ == "__main__":
    # Check degree of DAG
    # dag = generate_random_dag(2, )
    # print(np.mean(np.sum(dag, axis=1)))
    # Get 2 node DAG
    num_nodes = 9
    density = 4
    for i in range(1):
        dag = generate_synthetic_dag(num_nodes, num_nodes * density, "ER")
        print(dag)
