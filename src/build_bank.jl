"""
Build the original scalar modal propagation bank on CPU or CUDA.
Use BoostFractor.jl commit acf4f491fe633ab05c3b8abbafe6d6288e42aba0.
Run from the repository root: julia --project=julia src/build_bank.jl config/3d_bank.toml
Arrays are little-endian Float64/ComplexF64 in column-major order.
"""
using BoostFractor
using LinearAlgebra
using TOML
using SHA
const BF=BoostFractor
const Z0=376.730313668
const OUTPUT_ROOT=abspath(joinpath(@__DIR__,"..","data","3d"))

function guarded_directory(path)
    target=abspath(path)
    (target==OUTPUT_ROOT || startswith(target,OUTPUT_ROOT*"/")) ||
        throw(ArgumentError("bank output must remain under data/3d"))
    mkpath(target)
    return target
end

function context(; nxy=128, extent_m=0.6, mmax=8, lmax=0, disk_radius_m=0.15)
    # Half-open, centered FFT grid: exact zero is sampled for even nxy.
    xy = collect((-nxy÷2):(nxy-nxy÷2-1)) .* (extent_m/nxy)
    coords = SeedCoordinateSystem(X=xy, Y=xy)
    legacy = SeedModes(coords; ThreeDim=true, Mmax=mmax, Lmax=lmax,
                       diskR=disk_radius_m)
    basis = ModeBasis(legacy)
    B = reshape(basis.fields, :, size(basis.fields, 4))
    aperture = reshape(coords.X.^2,:,1) .+ reshape(coords.Y.^2,1,:) .<= disk_radius_m^2
    aperture_area_ratio = sum(aperture)*coords.dx*coords.dy/(pi*disk_radius_m^2)
    return (; coords, basis, B, radius=disk_radius_m, gram=B' * B, aperture_area_ratio)
end

function build_bank(ctx,frequencies_hz,distances_m;backend=CPUBackend())
    fixture=SeedSetupBoundaries(ctx.coords;diskno=0,distance=[0.,0.],
                              epsilon=ComplexF64[NaN,1.0-0.0im])
    grid=calc_propagation_matrix_grid(backend,fixture,ctx.coords,ctx.basis,
         Float64.(distances_m),Float64.(frequencies_hz);disk_radius=ctx.radius)
    return Array(grid[:,:,2,:,:,1,1])
end

function write_array!(manifest,directory,name,a)
    array=Array(a)
    eltype(array) in (Float64,ComplexF64,Int64) || error("unsupported binary element type")
    filename=name*".bin"
    path=joinpath(directory,filename)
    open(path,"w") do io
        write(io,vec(array))
    end
    dtype=eltype(array)==ComplexF64 ? "complex128" : eltype(array)==Float64 ? "float64" : "int64"
    manifest["arrays"][name]=Dict("file"=>filename,"dtype"=>dtype,
       "shape"=>collect(size(array)),"order"=>"F","sha256"=>bytes2hex(sha256(read(path))))
end

function write_bank(directory,ctx,frequencies_hz,distances_m,bank;metadata=Dict())
    dir=guarded_directory(directory)
    manifest=Dict{String,Any}("format_version"=>1,"byte_order"=>"little",
        "arrays"=>Dict{String,Any}(),"metadata"=>metadata,
        "normalization"=>Dict("basis"=>"unit discrete L2 norm per mode; no dx*dy in B",
          "physical_E_scale_Pin1"=>sqrt(2Z0/(ctx.coords.dx*ctx.coords.dy)),
          "disk_radius_m"=>ctx.radius,"dx_m"=>ctx.coords.dx,"dy_m"=>ctx.coords.dy,
          "discrete_aperture_to_piR2_ratio"=>ctx.aperture_area_ratio,
          "air_P_layout"=>"destination_mode,source_mode,frequency,distance",
          "trace_layout"=>"region,direction,mode; right region boundary; direction1 toward mirror",
          "incident"=>"whole infinite Gaussian power normalized before aperture projection",
          "received_boost"=>"bilinear overlap squared times sampled-aperture-area/(pi R^2)",
          "phasor"=>"exp(+i omega t), outgoing exp(-ikd), epsilon imaginary part negative",
          "mie_factors"=>"conjugate research/mie.py delta_e and delta_c before applying to these fields"))
    for (name,a) in ("basis_fields"=>ctx.basis.fields,"kt_per_m"=>ctx.basis.kt,
        "gram"=>ctx.gram,"x_m"=>ctx.coords.X,"y_m"=>ctx.coords.Y,
        "kx_fft_per_m"=>ctx.coords.kX_fft,"ky_fft_per_m"=>ctx.coords.kY_fft,
        "frequencies_hz"=>Float64.(frequencies_hz),"distances_m"=>Float64.(distances_m),"air_P"=>bank)
        write_array!(manifest,dir,name,a)
    end
    aperture = reshape(ctx.coords.X.^2,:,1) .+ reshape(ctx.coords.Y.^2,1,:) .<= ctx.radius^2
    uniform = Float64.(aperture)./sqrt(sum(aperture))
    write_array!(manifest,dir,"uniform_axion_modes",ctx.B'*vec(uniform))
    return manifest,dir
end


function run_bank(cfg)
    backend_name=get(cfg,"backend","cpu")
    backend=backend_name=="cuda" ? CUDABackend(get(cfg,"device",0)) :
            backend_name=="cpu" ? CPUBackend() : error("backend must be cpu or cuda")
    backend_name=="cuda" && !CUDA.functional() && error("CUDA is not functional")
    BLAS.set_num_threads(get(cfg,"blas_threads",1))
    ctx=context(nxy=get(cfg,"nxy",128),extent_m=get(cfg,"extent_m",.6),
        mmax=get(cfg,"mmax",8),lmax=get(cfg,"lmax",7),disk_radius_m=get(cfg,"disk_radius_m",.15))
    frequency=Float64.(cfg["frequencies_hz"])
    all(f -> 20e9<=f<=22e9,frequency) || error("frequency outside 20–22 GHz")
    distance=Float64.(cfg["distances_m"])
    println("Building ",length(frequency)," frequencies, ",size(ctx.gram,1)," modes on ",backend_name)
    bank=build_bank(ctx,frequency,distance;backend)
    metadata=Dict{String,Any}("backend"=>backend_name,"config"=>cfg,
        "git_commit"=>"acf4f491fe633ab05c3b8abbafe6d6288e42aba0",
        "model"=>"scalar diffraction/mode matching; not full vector Maxwell")
    manifest,dir=write_bank(cfg["output_dir"],ctx,frequency,distance,bank;metadata)
    open(joinpath(dir,"manifest.toml"),"w") do io
        TOML.print(io,manifest;sorted=true)
    end
    println("Saved ",dir)
end

if abspath(PROGRAM_FILE)==@__FILE__
    length(ARGS)==1 || error("usage: build_bank.jl config.toml")
    cfg=TOML.parsefile(ARGS[1])
    if get(cfg,"backend","cpu")=="cuda"
        @eval import CUDA
        CUDA.allowscalar(false)
    end
    Base.invokelatest(run_bank,cfg)
end
