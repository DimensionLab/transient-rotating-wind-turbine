from typing import Dict, List, Union, Tuple, Callable
import sympy as sp

from modulus.sym.domain.constraint.continuous import PointwiseConstraint
from modulus.sym.domain.constraint.utils import _compute_outvar, _compute_lambda_weighting
from modulus.sym.graph import Graph
from modulus.sym.key import Key
from modulus.sym.node import Node
from modulus.sym.loss import Loss, PointwiseLossNorm, IntegralLossNorm

from modulus.sym.geometry import Geometry
from modulus.sym.geometry.parameterization import Parameterization
from modulus.sym.models.arch import Arch

from modulus.sym.dataset import ContinuousPointwiseIterableDataset

class PointwiseRotatingBoundaryConstraint(PointwiseConstraint):
    """
    Pointwise Constraint applied to boundary/perimeter/surface of rotating geometry.
    For example, in 3D this will create a constraint on the surface of the
    given geometry.

    Parameters
    ----------
    nodes : List[Node]
        List of Modulus Nodes to unroll graph with.
    geometry : Geometry
        Modulus `Geometry` to apply the constraint with.
    angular_displacement : Union[float, sp.Basic]
        The angular displacement of the geometry.
    axis : str
        The axis to rotate the geometry around.
    outvar : Dict[str, Union[int, float, sp.Basic]]
        A dictionary of SymPy Symbols/Expr, floats or int.
        This is used to describe the constraint. For example,
        `outvar={'u': 0}` would specify `'u'` to be zero everywhere
        on the constraint.
    batch_size : int
        Batch size used in training.
    criteria : Union[sp.Basic, True]
        SymPy criteria function specifies to only apply constraint to areas
        that satisfy this criteria. For example, if
        `criteria=sympy.Symbol('x')>0` then only areas that have positive
        `'x'` values will have the constraint applied to them.
    lambda_weighting :  Dict[str, Union[int, float, sp.Basic]] = None
        The spatial pointwise weighting of the constraint. For example,
        `lambda_weighting={'lambda_u': 2.0*sympy.Symbol('x')}` would
        apply a pointwise weighting to the loss of `2.0 * x`.
    parameterization : Union[Parameterization, None], optional
        This allows adding parameterization or additional inputs.
    compute_sdf_derivatives: bool, optional
        Compute SDF derivatives when sampling geometery
    batch_per_epoch : int = 1000
        If `fixed_dataset=True` then the total number of points generated
        to apply constraint on is `total_nr_points=batch_per_epoch*batch_size`.
    quasirandom : bool = False
        If true then sample the points using the Halton sequence.
    num_workers : int
        Number of worker used in fetching data.
    loss : Loss
        Modulus `Loss` module that defines the loss type, (e.g. L2, L1, ...).
    shuffle : bool, optional
        Randomly shuffle examples in dataset every epoch, by default True
    """

    def __init__(
        self,
        nodes: List[Node],
        geometry: Geometry,
        outvar: Dict[str, Union[int, float, sp.Basic]],
        batch_size: int,
        angular_displacement: Union[float, sp.Basic] = 0.0,
        axis: str = "z",
        criteria: Union[sp.Basic, Callable, None] = None,
        lambda_weighting: Dict[str, Union[int, float, sp.Basic]] = None,
        parameterization: Union[Parameterization, None] = None,
        batch_per_epoch: int = 1000,
        quasirandom: bool = False,
        num_workers: int = 0,
        loss: Loss = PointwiseLossNorm(),
        shuffle: bool = True,
    ):
        self.geometry = geometry
        # invar function
        def invar_fn():
            self.geometry = self.geometry.rotate(angle=angular_displacement, axis=axis, parameterization=parameterization)
            return self.geometry.sample_boundary(
                batch_size,
                criteria=criteria,
                parameterization=parameterization,
                quasirandom=quasirandom,
            )

        # outvar function
        outvar_fn = lambda invar: _compute_outvar(invar, outvar)

        # lambda weighting function
        lambda_weighting_fn = lambda invar, outvar: _compute_lambda_weighting(
            invar, outvar, lambda_weighting
        )

        # make point dataloader
        dataset = ContinuousPointwiseIterableDataset(
            invar_fn=invar_fn,
            outvar_fn=outvar_fn,
            lambda_weighting_fn=lambda_weighting_fn,
        )

        # initialize constraint
        super().__init__(
            nodes=nodes,
            dataset=dataset,
            loss=loss,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=True,
            num_workers=num_workers,
        )