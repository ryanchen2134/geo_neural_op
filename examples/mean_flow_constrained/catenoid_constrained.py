"""
Constrained mean curvature flow on a cylinder PCD → catenoid.

The two end-circles of the cylinder are held fixed. Mean curvature flow
evolves the lateral surface toward the minimal surface of revolution
spanning those two circles — the catenoid:

    r(z) = a * cosh(z / a)

where ``a`` is chosen so the catenoid passes through the boundary circles.

Physical analogy: dip a wire frame (two parallel rings) into soap solution
and pull it out — the soap film settles into a catenoid.
"""

import torch
import numpy as np
from tqdm import tqdm

from gnp import GeometryEstimator
from gnp.utils import smooth_values_by_gaussian, subsample_points_by_radius


# ---------------------------------------------------------------------------
# Point cloud generation
# ---------------------------------------------------------------------------

def generate_cylinder_pcd(
    n_lateral: int = 3000,
    n_ring: int = 100,
    radius: float = 1.0,
    half_height: float = 0.6,
    end_bandwidth: float = 0.05,
    device: str = "cpu",
) -> dict:
    """
    Generate a cylinder lateral surface with two end-circle constraints.

    Points are sampled uniformly in angle and z on the cylinder r=radius,
    z in [-half_height, half_height]. The two end bands (z near ±half_height)
    are marked as constraints.

    Parameters
    ----------
    n_lateral : int
        Number of points on the lateral surface (excl. dense end rings).
    n_ring : int
        Number of extra points added explicitly on each end circle to ensure
        a well-resolved constraint boundary.
    radius : float
        Cylinder radius (= target catenoid radius at the end circles).
    half_height : float
        Half the cylinder height. The catenoid exists only if
        ``half_height / radius`` is below ~0.6627 (Goldschmidt limit);
        the default 0.6 is safely below that.
    end_bandwidth : float
        Fraction of half_height: points with |z| > (1 - end_bandwidth) * half_height
        are treated as the end-circle constraints.
    device : str

    Returns
    -------
    dict with keys:
        "xyz"               : (N, 3) float tensor
        "normals"           : (N, 3) float tensor  — outward radial normals
        "constraint_indices": (K,)  long tensor
    """
    # --- Lateral surface (uniform in angle × z) ----------------------------
    angles_lat = 2.0 * np.pi * np.random.rand(n_lateral)
    z_lat = np.random.uniform(-half_height, half_height, n_lateral)
    x_lat = radius * np.cos(angles_lat)
    y_lat = radius * np.sin(angles_lat)

    # --- Dense end rings (guaranteed constraint points) --------------------
    angles_ring = np.linspace(0, 2.0 * np.pi, n_ring, endpoint=False)
    x_top = radius * np.cos(angles_ring)
    y_top = radius * np.sin(angles_ring)
    z_top = np.full(n_ring, half_height)
    x_bot = x_top.copy()
    y_bot = y_top.copy()
    z_bot = np.full(n_ring, -half_height)

    x = np.concatenate([x_lat, x_top, x_bot])
    y = np.concatenate([y_lat, y_top, y_bot])
    z = np.concatenate([z_lat, z_top, z_bot])
    xyz = np.stack([x, y, z], axis=1)

    # Outward radial normals on a cylinder (no z component)
    nx = np.cos(np.arctan2(y, x))
    ny = np.sin(np.arctan2(y, x))
    nz = np.zeros_like(nx)
    normals = np.stack([nx, ny, nz], axis=1)

    # Constraint: both end bands
    threshold = (1.0 - end_bandwidth) * half_height
    constraint_mask = np.abs(z) > threshold
    # Also always include the explicitly placed ring points
    constraint_mask[n_lateral:] = True
    constraint_indices = np.where(constraint_mask)[0]

    return {
        "xyz": torch.tensor(xyz, dtype=torch.float32, device=device),
        "normals": torch.tensor(normals, dtype=torch.float32, device=device),
        "constraint_indices": torch.tensor(constraint_indices, dtype=torch.long, device=device),
    }


def analytic_catenoid(radius: float, half_height: float, n_pts: int = 200) -> np.ndarray:
    """
    Return points on the analytic catenoid r = a*cosh(z/a) for reference.

    ``a`` is found numerically so that r(half_height) = radius.
    Returns an (n_pts, 3) array sampled uniformly in angle, at a single z slice
    — use for visual overlay only.
    """
    from scipy.optimize import brentq

    def eq(a):
        return a * np.cosh(half_height / a) - radius

    # Goldschmidt limit: solution exists for half_height/radius < 0.6627
    ratio = half_height / radius
    if ratio >= 0.6627:
        raise ValueError(
            f"half_height/radius={ratio:.4f} exceeds the Goldschmidt limit (~0.6627); "
            "no catenoid exists for these boundary circles."
        )
    a = brentq(eq, 1e-6, radius)

    z_vals = np.linspace(-half_height, half_height, n_pts)
    r_vals = a * np.cosh(z_vals / a)
    angles = np.linspace(0, 2 * np.pi, n_pts, endpoint=False)

    # Build a grid of (z, angle) for a ribbon plot
    pts = []
    for z_val, r_val in zip(z_vals, r_vals):
        for ang in angles[:4]:          # sparse — just for shape reference
            pts.append([r_val * np.cos(ang), r_val * np.sin(ang), z_val])
    return np.array(pts)


# ---------------------------------------------------------------------------
# Constrained flow step  (identical logic to sphere_constrained.py)
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
    Single MCF step with fixed-point constraints.

    Normal and curvature estimation is delegated entirely to the GNP estimator
    via ``estimate_quantities``. The new logic is only:
      1. Zero the MCF displacement at constraint_indices.
      2. Restore any constraint points dropped by subsampling.
      3. Re-index constraint_indices into the post-subsampled frame.
    """
    if smooth_x:
        est = estimator.estimate_quantities(["xyz_coordinates"])
        estimator.pcd = est["xyz_coordinates"]
        estimator.data["x"] = estimator.pcd

    estimate = estimator.estimate_quantities(["normals", "mean_curvature"])
    x = estimator.pcd
    normals = estimate["normals"]
    mean_curvature = smooth_values_by_gaussian(
        x=x, values=estimate["mean_curvature"], radius=smooth_radius
    )

    displacement = delta_t * mean_curvature.view(-1, 1) * normals
    displacement[constraint_indices] = 0.0
    new_x = x + displacement

    subsampled_indices = subsample_points_by_radius(new_x, subsample_radius)

    dropped = constraint_indices[~torch.isin(constraint_indices, subsampled_indices)]
    if dropped.numel() > 0:
        subsampled_indices = torch.cat([subsampled_indices, dropped]).sort().values

    new_x = new_x[subsampled_indices]
    new_normals = normals[subsampled_indices]
    new_mean_curvature = mean_curvature[subsampled_indices]

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

    Returns
    -------
    list of dict
        Snapshots with keys "x", "normals", "mean_curvature", "constraint_indices".
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

        estimator.data = {k: v for k, v in new_data.items() if k != "constraint_indices"}
        estimator.pcd = new_data["x"]
        estimator.orientation = new_data["normals"]
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

    RADIUS = 1.0
    HALF_H = 0.6

    # --- Generate cylinder PCD -----------------------------------------------
    data = generate_cylinder_pcd(
        n_lateral=3000, n_ring=120,
        radius=RADIUS, half_height=HALF_H,
        end_bandwidth=0.05, device=device,
    )
    xyz, normals, c_idx = data["xyz"], data["normals"], data["constraint_indices"]
    print(f"Cylinder: {xyz.shape[0]} points, {c_idx.shape[0]} end-circle constraints")

    # --- Run constrained flow ------------------------------------------------
    estimator = GeometryEstimator(xyz, orientation=normals, device=device)

    history = constrained_mean_flow(
        estimator=estimator,
        constraint_indices=c_idx,
        num_steps=600,
        save_data_per_step=100,
        delta_t=0.0002,
        subsample_radius=0.04,
        smooth_radius=0.10,
        smooth_x=False,
    )

    # --- Verify boundary stays put ------------------------------------------
    print("\nBoundary check (end circles should stay at |z| ~ half_height, r ~ radius):")
    for idx, snap in enumerate(history):
        pts = snap["x"][snap["constraint_indices"]]
        z_vals = pts[:, 2]
        r_vals = pts[:, :2].norm(dim=1)
        print(
            f"  step {idx * 100:4d} | "
            f"z: [{z_vals.min():.3f}, {z_vals.max():.3f}] | "
            f"r: [{r_vals.min():.3f}, {r_vals.max():.3f}]"
        )

    # --- Analytic catenoid profile for comparison ---------------------------
    try:
        cat_pts = analytic_catenoid(RADIUS, HALF_H, n_pts=60)
    except Exception as e:
        cat_pts = None
        print(f"Could not compute analytic catenoid: {e}")

    # --- 3-D plot (initial cylinder vs final evolved surface) ---------------
    fig = plt.figure(figsize=(14, 6))
    titles = ["Initial cylinder", "After constrained MCF (→ catenoid)"]
    for col, snap in enumerate([history[0], history[-1]]):
        ax = fig.add_subplot(1, 2, col + 1, projection="3d")
        pts = snap["x"].cpu().numpy()
        cidx = snap["constraint_indices"].cpu().numpy()

        free_mask = np.ones(len(pts), dtype=bool)
        free_mask[cidx] = False

        ax.scatter(pts[free_mask, 0], pts[free_mask, 1], pts[free_mask, 2],
                   s=1, alpha=0.4, c="steelblue", label="free")
        ax.scatter(pts[cidx, 0], pts[cidx, 1], pts[cidx, 2],
                   s=6, c="red", zorder=5, label="constrained")

        if col == 1 and cat_pts is not None:
            ax.scatter(cat_pts[:, 0], cat_pts[:, 1], cat_pts[:, 2],
                       s=4, c="gold", alpha=0.7, label="analytic catenoid")

        ax.set_title(titles[col])
        lim = RADIUS * 1.3
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.set_zlim(-HALF_H * 1.2, HALF_H * 1.2)
        ax.legend(markerscale=4)

    plt.tight_layout()
    plt.savefig("catenoid_constrained_flow.png", dpi=150)
    print("\nSaved catenoid_constrained_flow.png")

    # --- Radial profile plot ------------------------------------------------
    fig2, ax2 = plt.subplots(figsize=(7, 4))
    colors = plt.cm.viridis(np.linspace(0, 1, len(history)))
    for snap, color in zip(history, colors):
        pts = snap["x"].cpu().numpy()
        cidx = snap["constraint_indices"].cpu().numpy()
        free_mask = np.ones(len(pts), dtype=bool)
        free_mask[cidx] = False
        z = pts[free_mask, 2]
        r = np.linalg.norm(pts[free_mask, :2], axis=1)
        ax2.scatter(z, r, s=0.5, color=color, alpha=0.3)

    if cat_pts is not None:
        z_cat = cat_pts[:, 2]
        r_cat = np.linalg.norm(cat_pts[:, :2], axis=1)
        ax2.plot(z_cat, r_cat, "k--", lw=2, label="analytic catenoid")
        ax2.legend()

    ax2.set_xlabel("z"); ax2.set_ylabel("r = sqrt(x²+y²)")
    ax2.set_title("Radial profile evolution (dark = later steps)")
    plt.tight_layout()
    plt.savefig("catenoid_radial_profile.png", dpi=150)
    print("Saved catenoid_radial_profile.png")
