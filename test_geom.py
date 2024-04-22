
from modulus.sym.geometry.primitives_3d import Box
from modulus.sym.utils.io.vtk import var_to_polyvtk
from modulus.sym.geometry.parameterization import Parameterization, Parameter
from modulus.sym.geometry.tessellation import Tessellation
from modulus.sym.hydra import to_absolute_path
import numpy as np
from sympy import Symbol, cos

t_symbol = Parameter("t")
parameterization = Parameterization({t_symbol: (0, 20)})

geom_path = to_absolute_path("./stl_files")
blades = Tessellation.from_stl(geom_path + "/blades.stl", airtight=True, parameterization=parameterization)
blades = blades.translate([-c for c in (125, 105, 40)])

# sample geometry over entire parameter range
s = blades.sample_boundary(nr_points=10000)
var_to_polyvtk(s, "blades")

# sample specific parameter
amplitude = np.pi / 12 # in radians
freq = 10.0 # Frequency in [rad/s] 
w = 2.0 * np.pi * amplitude * cos(freq * t_symbol) # Angular displacement 
blades = blades.rotate(angle=w, axis="y", parameterization=Parameterization({"t": 3}))
s = blades.sample_boundary(nr_points=10000)
var_to_polyvtk(s, "blades_rotated")
# var_to_polyvtk(s, "outputs/wind_turbine/initial_conditions/constraints/bladesBC")

channel_width = (-5.0, 5.0)
channel_length = (-10.0, 10.0)
channel_height = (-15.0, 15.0)

rec = Box(
    (channel_width[0], channel_length[0], channel_height[0]),
    (channel_width[1], channel_length[1], channel_height[1]),
)

s = rec.sample_boundary(nr_points=10000)
var_to_polyvtk(s, "box")