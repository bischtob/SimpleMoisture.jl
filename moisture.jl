# A simulation of forced-dissipative two-dimensional turbulence. We solve the
# two-dimensional vorticity equation with stochastic excitation and dissipation in
# the form of linear drag and hyperviscosity.
include("passive_tracer.jl")

using GeophysicalFlows, CUDA, Random, Printf, CairoMakie

### Device
dev = CPU()

### RNG
if dev == CPU()
  Random.seed!(1234)
else
  CUDA.seed!(1234)
end
random_uniform = dev == CPU() ? rand : CUDA.rand

### Numerical, domain, and simulation parameters
n = 256                           # number of grid points
L = 2π                            # domain size       
stepper = "ETDRK4"                # timestepper
ν, nν = 1e-14, 4                  # hyperviscosity coefficient and hyperviscosity order, 256: 1e-14
νc, nνc = ν, nν                   # hyperviscosity coefficient and hyperviscosity order for tracer
μ, nμ = 1e-2, 0                   # linear drag coefficient
dt = 1e-3                         # timestep
nsteps = 10000                    # total number of steps
nsubs = 40                        # number of steps between each plot
forcing_wavenumber = 4.0 * 2π / L # the forcing wavenumber, `k_f`, for a spectrum that is a ring in wavenumber space
forcing_bandwidth = 2.0 * 2π / L  # the width of the forcing spectrum, `δ_f`
ε = 0.1                           # energy input rate by the forcing
γ₀ = 1.0                          # saturation specific humidity gradient
e = 0.1                           # evaporation rate           
τc = .5e-1                         # condensation time scale
small_scale_amp = 0.0             # amplitude of small-scale forcing
small_scale_wn = 4                # wavenumber of small-scale forcing

### Grid
grid = TwoDGrid(dev; nx=n, Lx=L)

### Vorticity forcing
# We force the vorticity equation with stochastic excitation that is delta-correlated in time 
# and while spatially homogeneously and isotropically correlated. The forcing has a spectrum 
# with power in a ring in wavenumber space of radius ``k_f`` (`forcing_wavenumber`) and width 
# ``δ_f`` (`forcing_bandwidth`), and it injects energy per unit area and per unit time 
# equal to ``\varepsilon``. That is, the forcing covariance spectrum is proportional to 
# ``\exp{[-(|\bm{k}| - k_f)^2 / (2 δ_f^2)]}``.
K = @. sqrt(grid.Krsq)             # a 2D array with the total wavenumber
forcing_spectrum = @. exp(-(K - forcing_wavenumber)^2 / (2 * forcing_bandwidth^2))
CUDA.@allowscalar forcing_spectrum[grid.Krsq.==0] .= 0 # ensure forcing has zero domain-average
ε0 = FourierFlows.parsevalsum(forcing_spectrum .* grid.invKrsq / 2, grid) / (grid.Lx * grid.Ly)
@. forcing_spectrum *= ε / ε0        # normalize forcing to inject energy at rate ε

# Next we construct function `calcF!` that computes a forcing realization every timestep.
function calcF!(Fh, sol, t, clock, vars, params, grid)
  Fh .= sqrt.(forcing_spectrum) .* exp.(2π * im * random_uniform(eltype(grid), size(sol))) ./ sqrt(clock.dt)
  return nothing
end

### Saturation specific humidity gradients
function warp_none(grid)
  γx = device_array(dev)(zeros(grid.nx, grid.ny))
  γy = γ₀ * device_array(dev)(ones(grid.nx, grid.ny))
  return γx, γy
end

function warp_sine(grid)
  k = small_scale_wn
  xx, yy = ones(n) * grid.x', grid.y * ones(n)'
  γx = @. γ₀ * (small_scale_amp * cos(2π * k * xx / L) * sin(2π * k * yy / L) + 1)
  γy = @. γ₀ * (small_scale_amp * sin(2π * k * xx / L) * cos(2π * k * yy / L) + cos(2π * yy / L) / 4)
  return device_array(dev)(γx), device_array(dev)(γy)
end

γx, γy = warp_none(grid)

### Problem setup
NSprob = TwoDNavierStokes.Problem(dev; nx=n, Lx=L, ν, nν, μ, nμ, dt, stepper=stepper, calcF=calcF!, stochastic=true)
TwoDNavierStokes.set_ζ!(NSprob, device_array(dev)(zeros(grid.nx, grid.ny)))
ADprob = TracerAdvection.Problem(NSprob; νc=νc, nνc=nνc, e=e, τc=τc, γx=γx, γy=γy, stepper)

# Some shortcuts for the advection-diffusion problem:
sol, clock, vars, params, grid = ADprob.sol, ADprob.clock, ADprob.vars, ADprob.params, ADprob.grid
x, y = grid.x, grid.y

# Set tracer initial conditions
profile(x, y, σ) = 0
amplitude, spread = 1.0, 0.3
c₀ = device_array(dev)([amplitude * profile(x[i], y[j], spread) for j = 1:grid.ny, i = 1:grid.nx])
TracerAdvection.set_c!(ADprob, c₀)

# ### Diagnostics
E = Diagnostic(TwoDNavierStokes.energy, params.base_prob; nsteps) # energy
Z = Diagnostic(TwoDNavierStokes.enstrophy, params.base_prob; nsteps) # enstrophy
diags = [E, Z] # a list of Diagnostics passed to `stepforward!` will  be updated every timestep.

# Makie
c⁻ = Observable(Array(vars.c))
ζ = Observable(Array(params.base_prob.vars.ζ))
title_ζ = Observable("vorticity, μ t=" * @sprintf("%.2f", μ * clock.t))
energy = Observable(Point2f[(μ * E.t[1], E.data[1])])
enstrophy = Observable(Point2f[(μ * Z.t[1], Z.data[1] / forcing_wavenumber^2)])

fig = Figure(resolution=(3200, 1440))
axζ = Axis(fig[1, 1];
  ylabel="y",
  title=title_ζ,
  aspect=1,
  limits=((-L / 2, L / 2), (-L / 2, L / 2)))
axc = Axis(fig[1, 2];
  xlabel="x",
  ylabel="y",
  title="saturation deficit",
  aspect=1,
  limits=((-L / 2, L / 2), (-L / 2, L / 2)))
heatmap!(axζ, x, y, ζ;
  colormap=:balance, colorrange=(-40, 40))
heatmap!(axc, x, y, c⁻;
  colormap=:balance, colorrange=(-5, 5))

# Solution!
startwalltime = time()

frames = 0:round(Int, nsteps / nsubs)
CairoMakie.record(fig, "twodturb_forced.mp4", frames, framerate=25) do j
  # terminal update
  if j % (1000 / nsubs) == 0
    cfl = clock.dt * maximum([maximum(params.base_prob.vars.u) / grid.dx, maximum(params.base_prob.vars.v) / grid.dy])

    log = @sprintf("step: %04d, t: %d, cfl: %.2f, E: %.4f, Z: %.4f, walltime: %.2f min",
      clock.step, clock.t, cfl, E.data[E.i], Z.data[Z.i], (time() - startwalltime) / 60)
    println(log)
  end

  # Diags
  c⁻[] = vars.c
  ζ[] = params.base_prob.vars.ζ
  energy[] = push!(energy[], Point2f(μ * E.t[E.i], E.data[E.i]))
  enstrophy[] = push!(enstrophy[], Point2f(μ * Z.t[E.i], Z.data[Z.i] / forcing_wavenumber^2))
  title_ζ[] = "vorticity, μ t=" * @sprintf("%.2f", μ * clock.t)

  # Step!
  stepforward!(ADprob, nsubs)
  TracerAdvection.updatevars!(ADprob)
  stepforward!(params.base_prob, diags, nsubs)
  TwoDNavierStokes.updatevars!(params.base_prob)
end
