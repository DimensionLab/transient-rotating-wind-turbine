import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os

from sympy import Symbol, Eq, Abs, sin, cos, And, Or, Number, Function, simplify, exp, Min, log
from modulus.sym.eq.pde import PDE

import modulus
from modulus.sym.hydra import to_absolute_path, instantiate_arch, ModulusConfig
from modulus.sym.utils.io import csv_to_dict
from modulus.sym.utils.io.vtk import var_to_polyvtk
from modulus.sym.solver import SequentialSolver
from modulus.sym.domain import Domain
from modulus.sym.geometry.tessellation import Tessellation


from modulus.sym.loss.loss import CausalLossNorm


from modulus.sym.geometry.primitives_3d import Box
from modulus.sym.geometry.parameterization import OrderedParameterization, Parameterization
from modulus.sym.domain.inferencer  import VoxelInferencer

from modulus.sym.models.fully_connected import FullyConnectedArch
from modulus.sym.models.moving_time_window import MovingTimeWindowArch
from modulus.sym.domain.constraint import (
    PointwiseBoundaryConstraint,
    PointwiseInteriorConstraint,
)
from modulus.sym.domain.inferencer import PointVTKInferencer
from modulus.sym.utils.io import (
    VTKUniformGrid,
)
from modulus.sym.key import Key
from modulus.sym.node import Node
from modulus.sym.eq.pdes.navier_stokes import NavierStokes

from rotating_constraint import PointwiseRotatingBoundaryConstraint


@modulus.sym.main(config_path="conf", config_name="config") #config_fourier
def run(cfg: ModulusConfig) -> None:

    # time window parameters
    time_window_size = 1.0
    t_symbol = Symbol("t")
    amplitude = np.pi / 12 # in radians
    freq = 10.0 # Frequency in [rad/s] 
    w = 2.0 * np.pi * amplitude * cos(freq * t_symbol) # Angular displacement 
    time_range = {t_symbol: (0, time_window_size)}
    nr_time_windows = 10

    # make navier stokes equations - air of 20 deg C - nu=0.000015, rho=1.2
    ns = NavierStokes(nu=0.000015, rho=1.2, dim=3, time=True)

    # define sympy variables to parametrize domain curves
    x, y, z = Symbol("x"), Symbol("y"), Symbol("z")

    # make geometry for problem

    print(os.getcwd())
    # for now, just consider the blades
    geom_path = to_absolute_path("./stl_files")
    blades = Tessellation.from_stl(geom_path + "/blades.stl", airtight=True,
        parameterization=OrderedParameterization(time_range, key=t_symbol))

    # normalize meshes
    def normalize_mesh(mesh, center, scale):
        mesh = mesh.translate([-c for c in center])
        mesh = mesh.scale(scale) # not important right now
        return mesh

    center = (0, 0, 0)
    scale = 0.1
    blades = normalize_mesh(blades, center, scale)
    # hack to bring turbine blades into z-axis middle
    # blades = blades.translate([-c for c in (0, 0, -2)])

    channel_width = (-30.0, 30.0)
    channel_length = (-10.0, 90.0)
    channel_height = (-30.0, 30.0)
    box_bounds = {x: channel_width, y: channel_length, z: channel_height}
    print(box_bounds)

    # define interior geometry, without blades
    rec = Box(
        (channel_width[0], channel_length[0], channel_height[0]),
        (channel_width[1], channel_length[1], channel_height[1]),
        parameterization=OrderedParameterization(time_range, key=t_symbol)
    ) #+ blades

    geo = rec + blades
    geo_without_blades = rec - blades

    print(rec.bounds)
    print(geo.bounds)
    print(blades.bounds)

    # make network for current step and previous step
    flow_net = FullyConnectedArch(
        input_keys=[Key("x"), Key("y"), Key("z"), Key("t")],
        output_keys=[Key("u"), Key("v"), Key("w"), Key("p"), Key("k_star"), Key("ep_star")],
        periodicity={"x": channel_length, "y": channel_width, "z": channel_height},
        layer_size=256,
    )

    time_window_net = MovingTimeWindowArch(flow_net, time_window_size)

    # make nodes to unroll graph on
    nodes = (ns.make_nodes() 
            + [time_window_net.make_node(name="time_window_network")])

    # make initial condition domain
    ic_domain = Domain("initial_conditions")

    # make moving window domain
    window_domain = Domain("window")

    # make initial condition
    ic = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=geo_without_blades,
        bounds=box_bounds,
        outvar={
            "u": 0,
            "v": 5.0,
            "w": 0,
            "p": 0,
        },
        batch_size=cfg.batch_size.initial_condition,
        lambda_weighting={"u": 100, "v": 100, "w": 100, "p": 100},
        parameterization={t_symbol: 0},
    )
    ic_domain.add_constraint(ic, name="ic")
    
    # make constraint for matching previous windows initial condition
    ic = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=geo_without_blades,
        outvar={"u_prev_step_diff": 0, "v_prev_step_diff": 0, "w_prev_step_diff": 0},
        batch_size=cfg.batch_size.interior,
        bounds=box_bounds,
        lambda_weighting={
            "u_prev_step_diff": 100,
            "v_prev_step_diff": 100,
            "w_prev_step_diff": 100,
        },
        parameterization={t_symbol: 0},
    )
    window_domain.add_constraint(ic, name="ic")

    # inlet BC
    inletBC = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=rec,
        outvar={"u": 0, "v": 10, "w": 0},
        batch_size=cfg.batch_size.initial_condition,
        lambda_weighting={"u": 100, "v": 100, "w": 100},
        criteria=Eq(y, channel_length[0]),
        parameterization=time_range,
    )
    ic_domain.add_constraint(inletBC, "inletBC")
    window_domain.add_constraint(inletBC, "inletBC")

    # outlet BC
    outletBC = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=rec,
        outvar={"p" : 0},
        batch_size=cfg.batch_size.initial_condition,
        criteria=Eq(y, channel_length[1]),
        parameterization=time_range,
    )
    ic_domain.add_constraint(outletBC, "outletBC")
    window_domain.add_constraint(outletBC, "outletBC")

    # tunnel walls BC
    noslipBC = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=rec,
        outvar={"u": 0, "v": 0, "w": 0},
        batch_size=cfg.batch_size.initial_condition,
        parameterization=time_range,
        # criteria for all side walls
        criteria=And((y > channel_length[0]), 
                    (y < channel_length[1]), 
                    Or(
                        Or(Eq(x, channel_width[0]), Eq(x, channel_width[1])), 
                        Or(Eq(z, channel_height[0]), Eq(z, channel_height[1]))
                    )
                ),
    )
    ic_domain.add_constraint(noslipBC, "noslipBC")
    window_domain.add_constraint(noslipBC, "noslipBC")

    # blade geometry BC
    bladesBC = PointwiseRotatingBoundaryConstraint(
        nodes=nodes,
        geometry=blades,
        angular_displacement=w,
        axis="y",
        outvar={"u": 0, "v": 0, "w": 0},
        batch_size=cfg.batch_size.initial_condition,
        lambda_weighting={"u": 100, "v": 100, "w": 100},
        parameterization=OrderedParameterization(time_range, key=t_symbol),
    )
    ic_domain.add_constraint(bladesBC, "bladesBC")
    window_domain.add_constraint(bladesBC, "bladesBC")

    # make interior constraint
    interior = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=geo_without_blades,
        outvar={"continuity": 0, "momentum_x": 0, "momentum_y": 0, "momentum_z": 0},
        bounds=box_bounds,
        batch_size=1024,
    )
    ic_domain.add_constraint(interior, name="interior")
    window_domain.add_constraint(interior, name="interior")

    def mask_fn(x, y, z):
        sdf = geo_without_blades.sdf({"x": x, "y": y, "z": z}, {})
        return sdf["sdf"] < 0

    # add inference data for time slices
    for i, specific_time in enumerate(np.linspace(0, time_window_size, nr_time_windows)):
        vtk_obj = VTKUniformGrid(
            bounds=[channel_width, channel_length, channel_height],
            npoints=[64, 64, 64],
            export_map={"u": ["u", "v", "w"], "p": ["p"]},
        )
        grid_inference = PointVTKInferencer(
            vtk_obj=vtk_obj,
            nodes=nodes,
            input_vtk_map={"x": "x", "y": "y", "z": "z"},
            output_names=["u", "v", "w", "p"],
            requires_grad=False,
            invar={"t": np.full([64 ** 3, 1], specific_time)},
            mask_fn=mask_fn,
            mask_value=np.nan,
            batch_size=10000,
        )
        ic_domain.add_inferencer(grid_inference, name="time_slice_" + str(i).zfill(4))
        window_domain.add_inferencer(
            grid_inference, name="time_slice_" + str(i).zfill(4)
        )
        # # simulate rotation of blades
        # blades = blades.rotate(angle=2.0 * np.pi * amplitude * cos(freq * int(specific_time)), axis="y")
        # geo_inference = rec - blades
        # # add meshgrid inferencer
        # def mask_fn(x, y, z):
        #     sdf = geo_inference.sdf({"x": x, "y": y, "z": z}, {})
        #     return sdf["sdf"] < 0

        # voxel_inference = VoxelInferencer(
        #     bounds=[
        #         [lower, upper]
        #         for _, (
        #             lower,
        #             upper,
        #         ) in geo_inference.bounds.bound_ranges.items()
        #     ],
        #     npoints=[64, 64, 64],
        #     nodes=nodes,
        #     output_names=["u", "v", "w", "p"],
        #     export_map={"u": ["u", "v", "w"], "p": ["p"]},
        #     invar={"t": np.full([64**3, 1], specific_time)},
        #     mask_fn=mask_fn,
        #     mask_value=np.nan,
        #     batch_size=10000,
        #     requires_grad=False,
        # )
        # ic_domain.add_inferencer(voxel_inference, name="voxel_time_slice_" + str(i).zfill(4))
        # window_domain.add_constraint(voxel_inference, name="voxel_time_slice_" + str(i).zfill(4))
    
    def moving_body():
        print("\n\n\n\n")
        print(time_window_net.window_location.data)
        print("\n\n\n\n")
        blades = blades.rotate(angle=w, axis="y", parameterization=Parameterization({"t": time_window_net.window_location.data}))
        s = blades.sample_boundary(nr_points=cfg.batch_size.initial_condition, parameterization=Parameterization({"t": time_window_net.window_location.data}))
        var_to_polyvtk(s, "outputs/wind_turbine/initial_conditions/constraints/bladesBC")
        time_window_net.move_window()

    # make solver
    slv = SequentialSolver(
        cfg,
        [(1, ic_domain), (nr_time_windows, window_domain)],
        custom_update_operation=time_window_net.move_window,
    )

    # start solver
    slv.solve()


if __name__ == "__main__":
    run()