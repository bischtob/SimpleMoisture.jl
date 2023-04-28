using CUDA
using FourierFlows
using GeophysicalFlows
using Random
using UnicodePlots

### Numerical, domain, and simulation parameters
Random.seed!(10000)
dev = CPU()
n = 64                            # number of grid points
L = 2π                            # domain size       
small_scale_amp = 1.0             # amplitude of small-scale forcing; use 5.0
small_scale_wn = 1.0 * 2π / L     # wavenumber of small-scale forcing; use 6
small_scale_wdth = 1.0 * 2π / L   # bandwidth of small-scale forcing; use 2

### Grid
grid = TwoDGrid(nx=n, Lx=L)

### Warping of saturation specific humidity gradient
arr_K = Array(sqrt.(grid.Krsq))
arr_invKrsq = Array(grid.invKrsq)
spectrum = @. exp(-(arr_K - small_scale_wn)^2 / (2 * small_scale_wdth^2))
CUDA.@allowscalar spectrum[grid.Krsq.==0] .= 0 # ensure zero domain-average

# Random warping of background saturation specific humidity field
arr_k = ones(n,1) * Array(grid.k)'
arr_l = Array(grid.l)' * ones(1,n)
warp = sqrt.(spectrum) .* exp.(2π .* im .* rand.(eltype(grid)))
warp = Array(irfft(warp, grid.nx))
warpx = real(ifft(im .* arr_k .* fft(warp)))
warpy = real(ifft(im .* arr_l .* fft(warp)))
warpx .*= small_scale_amp / maximum(warp)
warpy .*= small_scale_amp / maximum(warp)
warp .*= small_scale_amp / maximum(warp)

UnicodePlots.heatmap(warp, width=64)

#Random.seed!(1234)
#Random.seed!(120)
#Random.seed!(240)
#Random.seed!(500)
#Random.seed!(10000)
