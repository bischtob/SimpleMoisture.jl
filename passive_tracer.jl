# Module that defines tooling for tracer advection extending
# FourierFlows.jl and GeophysicalFlows.jl
module TracerAdvection

export
    Problem,
    set_c!,
    updatevars!

using CUDA
using FourierFlows, GeophysicalFlows
using LinearAlgebra: mul!, ldiv!

import FourierFlows: Problem
import GeophysicalFlows.TwoDNavierStokes

"""
    Problem(dev::Device=CPU(), base_prob::FourierFlows.Problem; parameters...)

Construct a constant diffusivity problem on device `dev` using the flow from a
`GeophysicalFlows.MultiLayerQG` problem as the advecting flow. The device `CPU()`
is set as the default device.
"""
function Problem(base_prob::FourierFlows.Problem; νc, nνc, e, τc, γx=nothing, γy=nothing, stepper="FilteredRK4")
    grid = base_prob.grid

    # handle default topographic moisture source
    γx === nothing && (γx = zeros(grid.device, typeof(νc), (grid.nx, grid.ny)))
    γy === nothing && (γy = zeros(grid.device, typeof(νc), (grid.nx, grid.ny)))

    params = MoistureModelParams(νc, nνc, γx, γy, e, τc, base_prob)
    vars = Vars(grid, base_prob)
    equation = Equation(params, grid)
    dt = base_prob.clock.dt

    return FourierFlows.Problem(equation, stepper, dt, grid, vars, params)
end

abstract type AbstractTracerParams <: AbstractParams end

"""
    struct MoistureModelParams{T} <: AbstractTurbulentFlowParams

The parameters of a constant diffusivity problem with flow obtained from a
`FourierFlows.Problem` problem.
"""
struct MoistureModelParams{T,Aphys2D} <: AbstractTracerParams
    "``x``-diffusivity coefficient"
    κ::T
    "``y``-diffusivity coefficient"
    η::T
    "isotropic hyperdiffusivity coefficient"
    κh::T
    "isotropic hyperdiffusivity order"
    nκh::Int
    "saturation specific humidity gradient in y direction"
    γx::Aphys2D
    "saturation specific humidity gradient in y direction"
    γy::Aphys2D
    "evaporation rate"
    e::T
    "condentation time scale"
    τc::T
    "`FourierFlows.problem` to generate the advecting flow"
    base_prob::FourierFlows.Problem
end

"""
    MoistureModelParams(κ, η, base_prob)

Return the parameters `params` for a constant diffusivity problem with flow obtained
from a `GeophysicalFlows.TwoDNavierStokes` problem.
"""
function MoistureModelParams(νc, nνc, γx, γy, e, τc, base_prob)
    TwoDNavierStokes.updatevars!(base_prob)
    return MoistureModelParams(0νc, 0νc, νc, nνc, γx, γy, e, τc, base_prob)
end

"""
    struct Vars2D{Aphys, Atrans} <: AbstractVars

The variables of a 2D `TracerAdvectionDiffussion` problem.
"""
struct Vars2D{Aphys,Atrans} <: AbstractVars
    "tracer concentration"
    c::Aphys
    "tracer concentration ``x``-derivative, ``∂c/∂x``"
    cx::Aphys
    "tracer concentration ``y``-derivative, ``∂c/∂y``"
    cy::Aphys
    "Fourier transform of tracer concentration"
    ch::Atrans
    "Fourier transform of tracer concentration ``x``-derivative, ``∂c/∂x``"
    cxh::Atrans
    "Fourier transform of tracer concentration ``y``-derivative, ``∂c/∂y``"
    cyh::Atrans
end

"""
    Vars(dev, grid; T=Float64) 

Return the variables `vars` for a constant diffusivity problem on `grid` and device `dev`.
"""
function Vars(grid::AbstractGrid{T}, base_prob::FourierFlows.Problem) where {T}
    Dev = typeof(grid.device)

    @devzeros Dev T (grid.nx, grid.ny) c cx cy
    @devzeros Dev Complex{T} (grid.nkr, grid.nl) ch cxh cyh

    return Vars2D(c, cx, cy, ch, cxh, cyh)
end

"""
    Equation(dev, params, grid)

Return the equation for constant diffusivity problem with `params` and `grid` on device `dev`.
"""
function Equation(params::MoistureModelParams, grid)
    L = @. -params.κ * grid.kr^2 - params.η * grid.l^2 - params.κh * grid.Krsq^params.nκh
    CUDA.@allowscalar L[1, 1] = 0

    return FourierFlows.Equation(L, calcN!, grid)
end

"""
    calcN!(N, sol, t, clock, vars, params, grid)

Calculate the advective terms for a constant diffusivity `problem` with `params` and on `grid`.
"""
function calcN!(N, sol, t, clock, vars, params::AbstractTracerParams, grid)
    @. vars.cxh = im * grid.kr * sol
    @. vars.cyh = im * grid.l * sol

    # inverse transform
    ldiv!(vars.cx, grid.rfftplan, vars.cxh)
    ldiv!(vars.cy, grid.rfftplan, vars.cyh)

    u = params.base_prob.vars.u
    v = params.base_prob.vars.v

    # store N (in physical space) into vars.cx
    # add moisture source term
    @. vars.cx = -u * (vars.cx + params.γx) - v * (vars.cy + params.γy) + params.e - params.τc^(-1) * vars.c * (vars.c > 0)

    # fwd transform
    mul!(N, grid.rfftplan, vars.cx)

    return nothing
end

"""
    updatevars!(params::AbstractTracerParams, vars, grid, sol)

Update the `vars` on the `grid` with the solution in `sol` for a problem `prob`
that is being advected by a turbulent flow.     
"""
function updatevars!(params::AbstractTracerParams, vars, grid, sol)
    dealias!(sol, grid)

    @. vars.ch = sol

    ldiv!(vars.c, grid.rfftplan, deepcopy(vars.ch)) # deepcopy() since inverse real-fft destroys its input

    return nothing
end

updatevars!(prob) = updatevars!(prob.params, prob.vars, prob.grid, prob.sol)

"""
    set_c!(sol, params::AbstractTracerParams, grid, c)

Set the initial condition for tracer concentration in all layers of a
`TracerAdvectionDiffusion.Problem` that uses a `MultiLayerQG` flow to 
advect the tracer.
"""
function set_c!(sol, params::AbstractTracerParams, vars, grid, c)
    mul!(sol, grid.rfftplan, c)

    updatevars!(params, vars, grid, sol)

    return nothing
end

set_c!(prob, c) = set_c!(prob.sol, prob.params, prob.vars, prob.grid, c)

end # module
