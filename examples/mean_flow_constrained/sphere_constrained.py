"""
Constrained mean curvature flow on a sphere PCD.

The equatorial circle is held fixed (the "wire loop"), and the rest of
the surface evolves under mean curvature flow toward the minimal surface
spanning that loop — a flat disk, analogous to a soap bubble.

Physical analogy: Plateau's problem — find the minimal surface bounded
by a given closed curve. Here the curve is the unit-circle equator.
"""

import torch
import numpy as np
from tqdm import tqdm

from gnp import GeometryEstimator
from gnp.utils import smooth_values_by_gaussian, subsample_points_by_radius


# ---------------------------------------------------------------------------
# Point cloud generation
# ---------------------------------------------------------------------------

def generate_sphere_pcd(
    n_points: int = 5000,
    radius: float = 1.0,
    equator_bandwidth: float = 0.05,
    device: str = "cpu",
) -> dict:
    """
    Generate a near-uniformly sampled sphere via Fibonacci (golden-angle) sampling.

    Parameters
    ----------
    n_points : int
        Total number of points on the sphere.
    radius : float
        Sphere radius.
    equator_bandwidth : float
        Half-width (in z) of the equatorial band treated as the constraint loop.
        Points with |z| < equator_bandwidth * radius are pinned.
    device : str
        Torch device.

    Returns
    -------
    dict with keys:
        "xyz"               : (N, 3) float tensor  — point positions
        "normals"           : (N, 3) float tensor  — outward unit normals
        "constraint_indices": (K,)  long tensor   — indices of equator points
    """
    golden = (1.0 + np.sqrt(5.0)) / 2.0
    i = np.arange(n_points)

    # Polar angle from uniform z sampling
    z = 1.0 - 2.0 * i / (n_points - 1)          # z in [-1, 1]
    r = np.sqrt(np.clip(1.0 - z**2, 0.0, None))  # cylindrical radius
    phi = 2.0 * np.pi * i / golden               # azimuthal angle

    xyz = radius * np.stack([r * np.cos(phi), r * np.sin(phi), z], axis=1)
    normals = xyz / radius  # outward normals on a sphere

    constraint_mask = np.abs(z) < equator_bandwidth
    constraint_indices = np.where(constraint_mask)[0]

    return {
        "xyz": torch.tensor(xyz, dtype=torch.float32, device=device),
        "normals": torch.tensor(normals, dtype=torch.float32, device=device),
        "constraint_indices": torch.tensor(constraint_indices, dtype=torch.long, device=device),
    }


# ---------------------------------------------------------------------------
# Constrained flow step
# ---------------------------------------------------------------------------

def constrained_flow_step(
    estimator: GeometryEstimator,
    constraint_indices: torch.Tensor,
    delta_t: float,
    subsample_radius: float,
    smooth_radius: float,
    smooth_x: bool = False,
) -> dict:
    """
    Single mean curvature flow step with fixed-point constraints.

    Normal and curvature estimation is fully delegated to the GNP estimator.
    The only addition over the standard ``flow_step`` is:
      1. Zero out the displacement for constrained points before moving them.
      2. Pass constraint_indices as protected_indices to subsampling so they
         are never removed.
      3. Re-index constraint_indices into the post-subsampled frame.

    Parameters
    ----------
    estimator : GeometryEstimator
        GNP estimator (already initialised with the current PCD + normals).
    constraint_indices : torch.Tensor
        1-D long tensor of indices (into estimator.pcd) that must not move.
    delta_t : float
        Flow time step.
    subsample_radius : float
        Radius for point-cloud density control after the step.
    smooth_radius : float
        Gaussian smoothing radius applied to mean curvature before the step.
    smooth_x : bool
        If True, coordinates are first regularised via the GNP surface fit.

    Returns
    -------
    dict with keys "x", "normals", "mean_curvature", "constraint_indices"
        constraint_indices here refers to positions in the returned "x" array.
    """
    if smooth_x:
        est = estimator.estimate_quantities(["xyz_coordinates"])
        estimator.pcd = est["xyz_coordinates"]
        estimator.data["x"] = estimator.pcd

    # GNP estimates normals and mean curvature from the current PCD
    estimate = estimator.estimate_quantities(["normals", "mean_curvature"])
    x = estimator.pcd
    normals = estimate["normals"]
    mean_curvature = smooth_values_by_gaussian(
        x=x, values=estimate["mean_curvature"], radius=smooth_radius
    )

    # Compute displacement; pin the constrained points
    displacement = delta_t * mean_curvature.view(-1, 1) * normals
    displacement[constraint_indices] = 0.0
    new_x = x + displacement

    # Subsample for density control; protected_indices ensures constraints survive
    subsampled_indices = subsample_points_by_radius(
        new_x, subsample_radius, protected_indices=constraint_indices
    )

    new_x = new_x[subsampled_indices]
    new_normals = normals[subsampled_indices]
    new_mean_curvature = mean_curvature[subsampled_indices]

    # Re-index constraints into the new (subsampled) frame
    new_constraint_indices = torch.isin(
        subsampled_indices, constraint_indices
    ).nonzero(as_tuple=True)[0]

    return {
        "x": new_x.contiguous(),
        "normals": new_normals.contiguous(),
        "mean_curvature": new_mean_curvature.contiguous(),
        "constraint_indices": new_constraint_indices,
    }


# ---------------------------------------------------------------------------
# Constrained mean curvature flow loop
# ---------------------------------------------------------------------------

def constrained_mean_flow(
    estimator: GeometryEstimator,
    constraint_indices: torch.Tensor,
    num_steps: int,
    save_data_per_step: int,
    delta_t: float,
    subsample_radius: float,
    smooth_radius: float,
    smooth_x: bool = False,
) -> list:
    """
    Run constrained mean curvature flow for ``num_steps`` iterations.

    Parameters
    ----------
    estimator : GeometryEstimator
        Initialised with the starting PCD + normals.
    constraint_indices : torch.Tensor
        Indices of points that must remain fixed throughout the flow.
    num_steps : int
        Total number of flow steps.
    save_data_per_step : int
        Save a snapshot every this many steps.
    delta_t, subsample_radius, smooth_radius, smooth_x :
        Passed through to ``constrained_flow_step``.

    Returns
    -------
    list of dict
        Each entry has "x", "normals", "mean_curvature", "constraint_indices"
        for one saved snapshot.
    """
    save_data = []
    for i in tqdm(range(num_steps)):
        new_data = constrained_flow_step(
            estimator=estimator,
            constraint_indices=constraint_indices,
            delta_t=delta_t,
            subsample_radius=subsample_radius,
            smooth_radius=smooth_radius,
            smooth_x=smooth_x,
        )

        # Update estimator state
        estimator.data = {k: v for k, v in new_data.items() if k != "constraint_indices"}
        estimator.pcd = new_data["x"]
        estimator.orientation = new_data["normals"]

        # Thread constraint_indices forward (they point into the new x)
        constraint_indices = new_data["constraint_indices"]

        if not torch.isfinite(new_data["x"]).all():
            print(f"NaN/Inf detected at step {i}, stopping early.")
            return save_data

        if i % save_data_per_step == 0:
            save_data.append({k: v.clone() for k, v in new_data.items()})

    return save_data


# ---------------------------------------------------------------------------
# Quick demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # --- Generate sphere PCD -------------------------------------------------
    data = generate_sphere_pcd(n_points=4000, radius=1.0, equator_bandwidth=0.05, device=device)
    xyz, normals, c_idx = data["xyz"], data["normals"], data["constraint_indices"]
    print(f"Sphere: {xyz.shape[0]} points, {c_idx.shape[0]} equator constraints")

    # --- Run constrained flow ------------------------------------------------
    estimator = GeometryEstimator(xyz, orientation=normals, device=device)

    history = constrained_mean_flow(
        estimator=estimator,
        constraint_indices=c_idx,
        num_steps=400,
        save_data_per_step=50,
        delta_t=0.0002,
        subsample_radius=0.04,
        smooth_radius=0.08,
        smooth_x=False,
    )

    # --- Verify constraint satisfaction -------------------------------------
    print("\nConstraint check (equator points should stay at z~0, r~1):")
    for snap in history:
        pts = snap["x"][snap["constraint_indices"]]
        z_vals = pts[:, 2]
        r_vals = pts[:, :2].norm(dim=1)
        print(
            f"  step {history.index(snap) * 50:4d} | "
            f"z: [{z_vals.min():.4f}, {z_vals.max():.4f}] | "
            f"r: [{r_vals.min():.4f}, {r_vals.max():.4f}]"
        )

    # --- 3-D plot (first and last snapshot) ----------------------------------
    fig = plt.figure(figsize=(12, 5))
    for col, snap in enumerate([history[0], history[-1]]):
        ax = fig.add_subplot(1, 2, col + 1, projection="3d")
        pts = snap["x"].cpu().numpy()
        cidx = snap["constraint_indices"].cpu().numpy()

        free_mask = np.ones(len(pts), dtype=bool)
        free_mask[cidx] = False

        ax.scatter(pts[free_mask, 0], pts[free_mask, 1], pts[free_mask, 2],
                   s=1, alpha=0.4, c="steelblue", label="free")
        ax.scatter(pts[cidx, 0], pts[cidx, 1], pts[cidx, 2],
                   s=8, c="red", label="constrained")
        ax.set_title("Initial sphere" if col == 0 else "After constrained MCF")
        ax.set_xlim(-1.2, 1.2); ax.set_ylim(-1.2, 1.2); ax.set_zlim(-1.2, 1.2)
        ax.legend(markerscale=4)

    plt.tight_layout()
    plt.savefig("sphere_constrained_flow.png", dpi=150)
    print("\nSaved sphere_constrained_flow.png")
