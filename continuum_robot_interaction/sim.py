import numpy as np
from collections import defaultdict
import elastica as ea
import pickle

# TODO: many sim environment parameters are hardcoded

CYL_START = np.array([0, -0.70, 0.1])
CYL_DIRECTION = np.array([0.0, 1.0, 0.0])  # axis along z ("pin" out of plane)
CYL_NORMAL = np.array([1.0, 0.0, 0.0])
CYL_LENGTH = 1.40
CYL_RADIUS = 0.03
CYL_DENSITY = 10e6
CYL_CENTER_COORD = CYL_LENGTH / 2


# def rod_hits_cylinder(positions, rod_radius):
#     for idx in range(positions.shape[1]):
#         point = positions[:, idx]
#         relative = point - CYL_START
#         axis_coord = np.clip(np.dot(relative, CYL_DIRECTION), 0.0, CYL_LENGTH)
#         closest_point = CYL_START + axis_coord * CYL_DIRECTION
#         if np.linalg.norm(point - closest_point) <= CYL_RADIUS + rod_radius:
#             return True
#     return False

# only considered positive if below the obstacle center
def rod_hits_cylinder(positions, rod_radius):
    for idx in range(positions.shape[1]):
        point = positions[:, idx]
        relative = point - CYL_START
        axis_coord = np.clip(np.dot(relative, CYL_DIRECTION), 0.0, CYL_LENGTH)
        # Only consider points below the cylinder center
        if axis_coord < CYL_CENTER_COORD:
            closest_point = CYL_START + axis_coord * CYL_DIRECTION
            if np.linalg.norm(point - closest_point) <= CYL_RADIUS + rod_radius:
                return True
    return False


def pcc_centerline(kappa, phi, length, n_points, start_T):
    """
    Compute centerline points for a constant curvature section.
    """
    # Unbent section points along -y to match the simulated hanging rod.
    t0 = np.array([0.0, -1.0, 0.0])
    bend_dir = np.array([np.cos(phi), 0.0, np.sin(phi)])

    def _local_pose_at(s):
        if abs(kappa) < 1e-8:
            R = np.eye(3)
            p = t0 * s
            return R, p

        omega = kappa * bend_dir
        k_mag = np.linalg.norm(omega)
        v = omega / k_mag
        K = np.array(
            [
                [0.0, -v[2], v[1]],
                [v[2], 0.0, -v[0]],
                [-v[1], v[0], 0.0],
            ]
        )
        K2 = K @ K
        theta = k_mag * s

        sin_t = np.sin(theta)
        cos_t = np.cos(theta)
        R = np.eye(3) + sin_t * K + (1.0 - cos_t) * K2

        # Closed-form integral... p(s) = int_0^s R(u) t0 du
        p = (
            s * np.eye(3) + ((1.0 - cos_t) / k_mag) * K + ((theta - sin_t) / k_mag) * K2
        ) @ t0
        return R, p

    s_vals = np.linspace(0, length, n_points)
    pts = []

    for s in s_vals:
        _, p_local = _local_pose_at(s)
        p_world = start_T[:3, :3] @ p_local + start_T[:3, 3]
        pts.append(p_world)

    pts = np.array(pts).T

    # Local end transform from the same kinematics.
    R_end, p_end = _local_pose_at(length)
    T = np.eye(4)
    T[:3, :3] = R_end
    T[:3, 3] = p_end

    T_world = start_T @ T
    return pts, T_world


def pcc_robot_targets(inputs, n_elem, base_length=0.25):
    k1, phi1, k2, phi2 = inputs
    pts1, T1 = pcc_centerline(k1, phi1, base_length, n_elem + 1, start_T=np.eye(4))
    pts2, _ = pcc_centerline(k2, phi2, base_length, n_elem + 1, start_T=T1)
    return pts1, pts2


# Statics is the equilibrium mechanics of rigid bodies not experiencing an acceleration due to motion

# Not an exact statics formulation
# Damped proportional controller driving the system toward a quasi-static equilibrium
# Stiffness affects the final equilibrium when other forces are present


class PCCController(ea.NoForces):
    def __init__(self, targets, gain, damping):
        super().__init__()
        self.targets = targets
        self.k = gain
        self.d = damping

    def apply_forces(self, system, time=0.0):
        pos = system.position_collection
        vel = system.velocity_collection
        error = self.targets - pos
        # error[:,0] = 0.0
        system.external_forces += self.k * error - self.d * vel


# Simulation
class ContinuumSimulator(
    ea.BaseSystemCollection,
    ea.Constraints,
    ea.Connections,
    ea.Forcing,
    ea.Damping,
    ea.CallBacks,
    ea.Contact,
):
    pass


def run(inputs, output_path, params={}, final_time=4.0):
    sim = ContinuumSimulator()
    inputs = np.asarray(inputs, dtype=float)
    n_elem = 10  # elements per section
    base_length = 0.25
    base_radius = 0.025
    density = 1000
    E = 1e7
    shear_modulus = E / 1.5

    hang_dir = np.array([0.0, -1.0, 0.0])  # arm hangs downward
    arm_normal = np.array([0.0, 0.0, 1.0])  # cross-section normal (z-axis)

    def make_rod(start):
        return ea.CosseratRod.straight_rod(
            n_elem,
            start,
            hang_dir,
            arm_normal,
            base_length,
            base_radius,
            density,
            youngs_modulus=E,
            shear_modulus=shear_modulus,
        )

    pivot = np.array([0.0, 0.0, 0.0])  # fixed suspension point
    rod1 = make_rod(pivot)  # upper section
    rod2 = make_rod(pivot + hang_dir * base_length)  # lower section
    sim.append(rod1)
    sim.append(rod2)

    # Fix the suspension point (top node of rod1)
    sim.constrain(rod1).using(
        ea.FixedConstraint,
        constrained_position_idx=(0,),
        constrained_director_idx=(0,),
    )

    # Keep the base of rod2 attached to the tip of rod1
    sim.connect(rod1, rod2, first_connect_idx=-1, second_connect_idx=0).using(
        ea.FixedJoint, k=5e6, nu=100.0, kt=5e3
    )

    period = 1.0

    # Forward kinematics for piecewise constant curvature
    target1, target2 = pcc_robot_targets(inputs, n_elem, base_length)

    gain = params.get("gain", 4e4)
    damping = params.get("damping", 2e2)

    sim.add_forcing_to(rod1).using(
        PCCController, gain=gain, damping=damping, targets=target1
    )

    sim.add_forcing_to(rod2).using(
        PCCController, gain=gain, damping=damping, targets=target2
    )

    # Static obstacle cylinder (rigid body, fully constrained)
    obstacle = ea.Cylinder(
        CYL_START,
        CYL_DIRECTION,
        CYL_NORMAL,
        CYL_LENGTH,
        CYL_RADIUS,
        CYL_DENSITY,
    )
    sim.append(obstacle)
    sim.constrain(obstacle).using(
        ea.FixedConstraint,
        constrained_position_idx=(0,),
        constrained_director_idx=(0,),
    )

    dt = 5.0e-5 * period
    damping_constant = 10.0
    sim.dampen(rod1).using(
        ea.AnalyticalLinearDamper, damping_constant=damping_constant, time_step=dt
    )
    sim.dampen(rod2).using(
        ea.AnalyticalLinearDamper, damping_constant=damping_constant, time_step=dt
    )

    class StepCallBack(ea.CallBackBaseClass):
        def __init__(self, step_skip: int, callback_params: dict):
            ea.CallBackBaseClass.__init__(self)
            self.every = step_skip
            self.callback_params = callback_params
            self.collision_reported = False

        def make_callback(self, system, time, current_step: int):
            if current_step % self.every == 0:
                self.callback_params["time"].append(time)
                self.callback_params["position"].append(
                    system.position_collection.copy()
                )
                if not self.collision_reported and rod_hits_cylinder(system.position_collection, base_radius):
                    self.callback_params["collision"] = self.collision_reported

    pp1 = defaultdict(list)
    pp2 = defaultdict(list)
    sim.collect_diagnostics(rod1).using(
        StepCallBack, step_skip=100, callback_params=pp1
    )
    sim.collect_diagnostics(rod2).using(
        StepCallBack, step_skip=100, callback_params=pp2
    )

    # Instantiate contact features last so all bodies, constraints, and forcing are in place first.
    for rod in (rod1, rod2):
        sim.detect_contact_between(rod, obstacle).using(
            ea.RodCylinderContact,
            k=5e5,
            nu=5e2,
            velocity_damping_coefficient=5e2,
        )

    sim.finalize()
    timestepper = ea.PositionVerlet()
    total_steps = int(final_time / dt)
    print("Total steps", total_steps)
    ea.integrate(timestepper, sim, final_time, total_steps)

    with open(output_path, "wb") as file:
        pickle.dump({"rod1": pp1, "rod2": pp2}, file)
        
    collision = pp1.get("collision", False) or pp2.get("collision", False)
    return collision
