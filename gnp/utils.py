from typing import Tuple

import torch
import torch_geometric as tg
from torch_scatter import scatter_add, scatter_max


class QueryTorchGeometric:
    """
    A wrapper for performing k-nearest neighbor and radius queries on a point
    cloud using PyTorch Geometric.

    Parameters
    ----------
    x : torch.Tensor
        The point cloud data to be queried, of shape (N, D).
    device : str, optional
        The device to store the point cloud on, by default "cpu".
    """

    def __init__(self, x: torch.Tensor, device="cpu"):
        self.x = x.to(device)
        self.device = torch.device(device)

    def query_knn(
        self, queries: torch.Tensor, k: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Find the k-nearest neighbors for a set of query points.

        Parameters
        ----------
        queries : torch.Tensor
            The query points, of shape (M, D).
        k : int
            The number of nearest neighbors to find.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            A tuple containing:
            - The distances to the k-nearest neighbors, shape (M, k).
            - The indices of the k-nearest neighbors, shape (M, k).
        """

        if not queries.device == self.device:
            queries = queries.to(self.device)
        index_y, index_x = tg.nn.knn(self.x, queries, k=k)
        distances = (self.x[index_x.view(-1, k)] - queries[index_y.view(-1, k)]).norm(
            dim=-1
        )

        return distances, index_x.view(-1, k)

    @staticmethod
    def query_radius(
        x: torch.Tensor,
        y: torch.Tensor,
        radius: float,
        max_num_neighbors: int = 100,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Find all points in `x` within a given radius of points in `y`.

        This method finds all pairs (i, j) such that the distance between
        x[i] and y[j] is less than the radius.

        Parameters
        ----------
        x : torch.Tensor
            The point cloud to search within (the "haystack").
        y : torch.Tensor
            The query points (the "needles").
        radius : float
            The search radius.
        max_num_neighbors : int, optional
            The maximum number of neighbors to return for each query point,
            by default 100.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            A tuple containing:
            - `index_x`: Indices of the found points in `x`.
            - `index_y`: Indices of the corresponding query points in `y`.
        """

        assert radius >= 0
        index_y, index_x = tg.nn.radius(
            x=x, y=y, r=radius, max_num_neighbors=max_num_neighbors
        )
        return index_x, index_y


def subsample_points_by_radius(
    x: torch.Tensor, radius: float, protected_indices: torch.Tensor = None
) -> torch.Tensor:
    """
    Subsample points using a parallel maximal independent set approach.

    Parameters
    ----------
    x : torch.Tensor
        The input points to subsample, shape (N, D).
    radius : float
        The radius for defining neighborhoods.
    protected_indices : torch.Tensor, optional
        1-D long tensor of point indices that must never be removed.
        Protected points are assigned a rank above the random range so
        they always win the local-max competition in the MIS algorithm.
        When two points are within ``radius`` and one is protected, the
        free point is the one that gets removed.

    Returns
    -------
    torch.Tensor
        A tensor containing the indices of the subsampled points.
    """
    device = x.device
    num_points = x.shape[0]

    random_rank = torch.rand(num_points, device=device)
    if protected_indices is not None:
        random_rank[protected_indices] = 2.0
    neighbor_indices, query_indices = QueryTorchGeometric.query_radius(
        x, x, radius, max_num_neighbors=1024
    )

    mask = torch.ones(x.shape[0], device=device).bool()
    mask_sum = mask.sum() - 1

    while mask_sum != mask.sum():
        random_rank = torch.rand(num_points, device=device)
        if protected_indices is not None:
            random_rank[protected_indices] = 2.0
        mask_sum = mask.sum()
        max_neighbor_rank, _ = scatter_max(
            random_rank[neighbor_indices], query_indices, dim=0, dim_size=num_points
        )

        max_neighbor_mask = (random_rank == max_neighbor_rank) & mask
        neighbor_sum_mask = scatter_add(
            max_neighbor_mask[neighbor_indices], query_indices, dim=0, dim_size=num_points
        )
        mask = (max_neighbor_mask | ~neighbor_sum_mask) & mask
        edge_mask = mask[neighbor_indices] & mask[query_indices]
        neighbor_indices = neighbor_indices[edge_mask]
        query_indices = query_indices[edge_mask]
        
    if (~mask.any()):
        print(f"Removed {(1 - mask).sum()} points.")
    return torch.where(mask)[0]


def smooth_values_by_gaussian(
    x: torch.Tensor, values: torch.Tensor, radius: float
) -> torch.Tensor:
    """
    Smooth values on a point cloud using a truncated Gaussian kernel.

    For each point, this function computes a weighted average of the values of
    its neighbors within a given radius. The weights are determined by a
    Gaussian function of the distance.

    Parameters
    ----------
    x : torch.Tensor
        The input data points, shape (N, D).
    values : torch.Tensor
        The values associated with each point to be smoothed, shape (N,).
    radius : float
        The truncation radius for the Gaussian kernel. The standard deviation
        of the Gaussian is set to one-third of this radius.

    Returns
    -------
    torch.Tensor
        A tensor of shape (N,) containing the smoothed values.
    """

    neighbor_indices, query_indices = QueryTorchGeometric.query_radius(
        x, x, radius, max_num_neighbors=1024
    )

    distances = (x[neighbor_indices] - x[query_indices]).norm(dim=1)
    weights = torch.exp(-distances.pow(2) / (2 * (radius / 3) ** 2))
    weight_sums = scatter_add(weights, index=query_indices, dim=0, dim_size=x.size(0))
    weights_normalized = weights / weight_sums[query_indices]

    values_averaged = scatter_add(
        weights_normalized * values[neighbor_indices],
        query_indices,
        dim=0,
        dim_size=x.size(0),
    )
    return values_averaged
