import os
import pathlib
from typing import List

import matplotlib as mpl
import matplotlib.animation as animation  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import numpy as np
from matplotlib.colors import to_rgb
from matplotlib.patches import Circle
from mpl_toolkits.axes_grid1 import make_axes_locatable

mpl.rcParams.update(
    {
        "font.family": "STIXGeneral",
        "mathtext.fontset": "stix",
        "font.size": 24,
        "axes.labelsize": 35,
        "axes.titlesize": 35,
        "xtick.labelsize": 22,
        "ytick.labelsize": 22,
        "legend.fontsize": 20,
    }
)

from basic_boundary_function.env import Env
from basic_boundary_function.onMan_approximation import OnMan_Approx  # type: ignore
from configs import (
    CircularRobotSpecification,
    EnvContrConfiguration,
    GPDFConfiguration,
)
from control.cbf_control import OnManCBFController  # type: ignore

from pkg_moving_object.human_trajs import dx_c, x_c
from pkg_moving_object.moving_object import RobotObject

root_dir = pathlib.Path(__file__).resolve().parents[1]
yaml_path = os.path.join(root_dir, "config", "env_controller.yaml")
econf = EnvContrConfiguration.from_yaml(yaml_path)

yaml_path = os.path.join(root_dir, "config", "gpdf.yaml")
gconf = GPDFConfiguration.from_yaml(yaml_path)


spec_robot_fpath = os.path.join(root_dir, "config", "spec_robot.yaml")
spec_robot = CircularRobotSpecification.from_yaml(spec_robot_fpath)



def animate(n_iter):
    """Call repeatedly by FuncAnimation."""
    if n_iter % 20 == 0:
        print(n_iter)

    fig.clear()
    ax = fig.add_subplot(111, aspect="equal", autoscale_on=True)

    if econf.env_name == "hospital":
        human_pos = x_c(59 + n_iter, steps=200)
        human_vel = dx_c(59 + n_iter, steps=200)
    elif econf.env_name == "social_nav_dynamic":
        human_pos = x_c(n_iter + 1, steps=30 / (spec_robot.ts / 0.05), dt=spec_robot.ts)
        human_vel = dx_c(
            n_iter + 1, steps=30 / (spec_robot.ts / 0.05), dt=spec_robot.ts
        )
    if env.num_dyn_circle:
        env.update_xc(human_pos[:, :2], human_vel[:, :2])
        for i in range(env.num_dyn_circle):
            ax.add_patch(
                Circle(
                    (env.xc[i, 0], env.xc[i, 1]),
                    env.radius,
                    color="orange",
                    fill=True,
                )
            )

    ax.set_aspect("equal", adjustable="box")
            


    c_bar = None
    rob_color = ["#135e08"] * num_rob
    for n_rob in range(num_rob):
        robot = robot_list[n_rob]
        if not isSuccess[n_rob]:
            u_nom = controller_list[n_rob].get_nominal_ctrl(k_p=1.0)
            if CONTROLLER == "cbf":
                controller = controller_list[n_rob]
                controller.set_state(robot.state)
                u_mod, controller_status = controller.run_step(
                    n_iter, om, om.env.env_name, u_nom=u_nom
                )
            elif CONTROLLER is None:
                u_mod = u_nom
                controller_status = {}
                controller_status["isInfeasible"] = False

            all_u_list[:, n_iter, n_rob] = u_mod
            all_x_list[:, n_iter, n_rob] = robot.state
            try:
                dist_2_goal = np.linalg.norm(robot.state[:2] - econf.target)
            except:
                print(robot.state)
                print(econf.target)
                breakpoint()

            robot.one_step(u_mod)

            if dist_2_goal <= 0.4 and (iterations[n_rob] == max_iter):
                iterations[n_rob] = n_iter
                isSuccess[n_rob] = 1
            elif (dist_2_goal > 0.4) and (iterations[n_rob] == max_iter):
                ### For visualization
                if (env.h_set(all_x_list[:2, n_iter, n_rob].reshape(1, 2)) <= 0).any():
                    rob_color[n_rob] = "r"
                    isSafe[n_rob] = 0
                else:
                    rob_color[n_rob] = "#135e08"  # basically green
                if controller_status["isInfeasible"]:
                    rob_color[n_rob] = "b"
                    isinfeasible[n_rob] += 1
                t = np.linspace(0, 1, n_iter)
                fade_strength = np.clip(n_iter / 500, 0.1, 10)

                # old points closer to 0, new points = 1
                t_faded = t**fade_strength

                colors = plt.cm.Greens(0.2 + 0.5 * t_faded)

                traj_step = 10
                ax.scatter(
                    all_x_list[0, :n_iter, n_rob][::traj_step],
                    all_x_list[1, :n_iter, n_rob][::traj_step],
                    c=colors[::traj_step],
                    marker="o",
                    s=400,
                )

            pass
    ax.plot(
        econf.target[0], econf.target[1], "X", color="k", markersize=50, label="goal"
    )

    for n_rob in range(num_rob):
        ax.add_patch(
            Circle(
                (robot_list[n_rob].state[0], robot_list[n_rob].state[1]),
                0.2,
                color=rob_color[n_rob],
                fill=True,
                zorder=100000,
            )
        )

        if econf.dynamics == "differential":
            ax.plot(
                [
                    robot_list[n_rob].state[0],
                    robot_list[n_rob].state[0]
                    + 0.3 * np.cos(robot_list[n_rob].state[2]),
                ],
                [
                    robot_list[n_rob].state[1],
                    robot_list[n_rob].state[1]
                    + 0.3 * np.sin(robot_list[n_rob].state[2]),
                ],
                color="k",
                zorder=200000,
            )
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")

    if c_bar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.2)
        cbar = fig.colorbar(c_bar, cax=cax)
        cbar.ax.tick_params(labelsize=23)
    if econf.env_name[:10] == "social_nav":
        env.plot_env_standard(
            ax, "k", plot_grad_dir=vis_env_only, dynamic_obstacle=True
        )
    else:
        env.plot_env_standard(ax, "k", plot_grad_dir=vis_env_only)


if __name__ == "__main__":
    import sys

    CONTROLLER = "cbf"  # Controller selection:
    #   "cbf"       - Proposed MCBF-QP or standard CBF-QP,
    #                 depending on the MCBF flag in config/env_controller.yaml.
    #   None        - No controller/safety filters.


    vis_env_only = (
        False  # If True, visualize the environment without running the controllers.
    )
    max_iter = 50

    if econf.dynamics == "differential":
        num_rob = 10
        state_num = 3
    else:
        num_rob = 1
        state_num = 2

    max_speed = 1.5
    all_x_list = np.zeros((state_num, max_iter + 1, num_rob))
    all_u_list = np.zeros((2, max_iter, num_rob))
    all_execution_times = np.zeros((num_rob, max_iter))
    iterations = max_iter * np.ones((num_rob,))
    isSuccess = np.zeros((num_rob,))
    isSafe = np.ones((num_rob,))
    isinfeasible = np.zeros((num_rob,))

    env = Env(
        load_env=True,
        env_name=econf.env_name,
        rho=gconf.rho,
        radius=econf.radius,
        num_dyn_circle=econf.num_dyn_circle,
    )

    if CONTROLLER == "cbf":
        om = OnMan_Approx(env, w=econf.w, hold_time=econf.hold_time)

    controller_list: List[OnManCBFController] = []
    robot_list: List[RobotObject] = []
    ini_states = np.zeros((num_rob, state_num))

    for nr in range(num_rob):
        ini_state = np.array(
            [econf.init_state[0], econf.init_state[1], int(nr) * 2 * np.pi / num_rob]
        )
        ini_states[nr] = ini_state

        if CONTROLLER == "cbf":
            controller = OnManCBFController(
                threeD_controller=econf.threeD_controller,
                autotune=econf.autotune,
                dynamics=econf.dynamics,
            )
            controller.set_params(
                target=econf.target,
                nominal_speed=econf.nominal_speed,
                sampling_time=spec_robot.ts,
                target_range=econf.target_range,
                ve=econf.ve,
                beta_coef=econf.beta_coef,
                om_expand_threshold=econf.om_expand_threshold,
                om_range=econf.om_range,
                MCBF=econf.MCBF,
                dir_num=econf.dir_num,
            )

            controller.set_init_data((ini_state[:state_num]), max_iter)

        if CONTROLLER is not None:
            controller_list.append(controller)

        robot = RobotObject.from_yaml(
            ini_state[:state_num], spec_robot_fpath, dynamics=econf.dynamics
        )  # we only need one robot for now
        robot_list.append(robot)
    # ----------------------------------------------------------------------------------------------------------------
    ### Animation

    if vis_env_only:
        fig, ax = plt.subplots()
        env.plot_env_standard(ax, "k")
        ax.set_aspect("equal", adjustable="box")
        plt.show()
        sys.exit(0)

    writervideo = animation.FFMpegWriter(fps=60)
    fig, ax = plt.subplots(figsize=(15, 15))

    fig.subplots_adjust(left=0.06, right=0.98, bottom=0.06, top=0.98)

    anim = animation.FuncAnimation(
        fig, animate, frames=max_iter, interval=spec_robot.ts * 1000, repeat=False
    )
    fig.canvas.draw()
    anim.event_source.stop()
    scenario_num = 1
    if CONTROLLER == "cbf":
        anim.save(
            "../examples/"
            + CONTROLLER
            + "_"
            + str(env.env_name)
            + str(scenario_num)
            + "_3D"
            + str(controller.threeD_controller)
            + "_mcbf"
            + str(econf.MCBF)
            + "_obs"
            + str(econf.num_dyn_circle)
            + "_ts"
            + str(spec_robot.ts)
            + "_autotune"
            + str(econf.autotune)
            + ".gif",
            writer=writervideo,
        )
    else:
        anim.save(
            "../examples/"
            + str(CONTROLLER)
            + "_"
            + str(env.env_name)
            + str(scenario_num)
            + ".gif",
            writer=writervideo,
        )

    for nr in range(num_rob):
        print(
            f"Robot {nr} - Success: {isSuccess[nr]}, Safe: {isSafe[nr]}, Infeasible: {isinfeasible[nr]}, Iterations: {iterations[nr]}"
        )
