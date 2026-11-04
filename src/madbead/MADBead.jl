"""The existing MADBead 1D routines, with only their required imports.

field_1D_fit.jl omits the prior-value fallback for too few z samples: SBP uses
its shared geometry fit and fits only E_0 in the per-frequency MC stage.
"""
module MADBead
using LinearAlgebra, Statistics, Random, DataFrames, Measurements, LsqFit, SpecialFunctions
const c_const = 299792458
const ϵ0 = 8.85418782e-12
const Z0 = 376.730313667
include("bead_finite_size.jl")
include("field_1D_fit.jl")
end
