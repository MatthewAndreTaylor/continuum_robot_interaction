from continuum_robot_interaction import sim, renderer
import numpy as np
import os


sim_output_path = "sim.dat"

## Inputs
# kappa (𝜅), phi (𝜙), ( l (ℓ) ignored )
# inputs = (κ1, φ1, κ2, φ2)

# params is physical params of the robot

# final time is the time to reach equilbrium, (0.1s is usually enough)

def main(inputs=None, params={}, final_time=0.1):
    HEADLESS = False
    
    if inputs is None:
        kappa = np.random.uniform(5.0, 10.0, 2)
        phi = np.random.uniform(-np.pi, np.pi, 2)
        inputs = np.array([kappa[0], phi[0], kappa[1], phi[1]])
    
    # inputs = np.array([0.0, 0.0, 0.0, 0.0])  # straight configuration for testing fixed
    OVERRIDE_SIM_DATA = True
    
    if OVERRIDE_SIM_DATA:
        print(inputs)
        collision_detected = sim.run(inputs, sim_output_path, params, final_time)
        print("Collision detected during simulation:", collision_detected)

    if not HEADLESS:
        renderer.render_sim(
            data_path=sim_output_path,
            scene=renderer.RenderScene(HEADLESS)
        )
        return True

    if not collision_detected:
        return False
    
    if inputs.shape == (4,):
        screenshot_filename = (
            f"param_k1_{inputs[0]:.2f}_phi1_{inputs[1]:.2f}_"
            f"k2_{inputs[2]:.2f}_phi2_{inputs[3]:.2f}.png"
        )
    else:
        raise ValueError("Expected inputs shape (4,), got {}".format(inputs.shape))
    
    image_path = os.path.join("synthetic_dataset", screenshot_filename)
    renderer.render_sim(headless=True, screenshot_path=image_path)

    print("Saved screenshot:", image_path)
    return True


if __name__ == "__main__":
    # os.makedirs("synthetic_dataset", exist_ok=True)
    
    # list synthetic_dataset directory contents
    # existing_files = os.listdir("synthetic_dataset_og")
    # or use parameters from a CSV file if available
    
    # for filename in existing_files:
    #     if filename.endswith(".png"):
    #         # parse parameters from filename
    #         parts = filename[:-4].split("_")
    #         k1 = float(parts[2])
    #         phi1 = float(parts[4])
    #         k2 = float(parts[6])
    #         phi2 = float(parts[8])
            
    #         inputs = np.array([k1, phi1, k2, phi2])
    #         main(inputs)
    
    # for _ in range(100000):
    #     main()
    
    
    # Some dataset examples for visualization:
    inputs = [5.03,-2.65,6.59,1.79]
    
    # inputs = [7.6,2.7,6.64,1.4]
    
    # main(inputs=inputs, params={"gain": 5e3, "damping": 5e2}, final_time=4.0)
    
    main(inputs=inputs)