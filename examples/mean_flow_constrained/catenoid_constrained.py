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
      2. Pass constraint_indices as protected_indices to subsampling so they
         are never removed.
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

    subsampled_indices = subsample_points_by_radius(
        new_x, subsample_radius, protected_indices=constraint_indices
    )

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
    HALF_H = 1.0

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

    # --- PyVista plot (all snapshots) ----------------------------------------
    import pyvista as pv

    ncols = 3
    n_total = len(history) + 1  # +1 for initial cylinder
    nrows = (n_total + ncols - 1) // ncols

    camera_pos = [
        (RADIUS * 5, RADIUS * 2, HALF_H * 2),
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 1.0),
    ]

    all_snaps = (
        [{"x": xyz, "constraint_indices": c_idx,
          "mean_curvature": None, "label": "Initial cylinder"}]
        + [{"x": s["x"], "constraint_indices": s["constraint_indices"],
            "mean_curvature": s["mean_curvature"],
            "label": f"Step {(i + 1) * 100}"}
           for i, s in enumerate(history)]
    )

    cat_poly = pv.PolyData(cat_pts) if cat_pts is not None else None
    plotter = pv.Plotter(shape=(nrows, ncols), window_size=(600 * ncols, 600 * nrows))

    for i, snap in enumerate(all_snaps):
        row, col = divmod(i, ncols)
        plotter.subplot(row, col)

        pts = snap["x"].cpu().numpy()
        cidx = snap["constraint_indices"].cpu().numpy()
        free_mask = np.ones(len(pts), dtype=bool)
        free_mask[cidx] = False

        mc = snap["mean_curvature"]
        if mc is not None:
            plotter.add_points(
                pv.PolyData(pts[free_mask]),
                point_size=5, render_points_as_spheres=True,
                cmap="bwr",
                scalars=mc.cpu().numpy()[free_mask],
                clim=(-5 * mc.std().item(), 5 * mc.std().item()),
            )
            plotter.remove_scalar_bar()
        else:
            plotter.add_points(
                pv.PolyData(pts[free_mask]),
                color="steelblue", point_size=5, render_points_as_spheres=True,
            )

        plotter.add_points(
            pv.PolyData(pts[cidx]),
            color="red", point_size=8, render_points_as_spheres=True,
        )

        if cat_poly is not None:
            plotter.add_points(cat_poly, color="gold", point_size=6, render_points_as_spheres=True)

        plotter.add_text(snap["label"], font_size=12)
        plotter.camera_position = camera_pos

    plotter.show()
    plotter.screenshot("catenoid_constrained_flow.png")
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
