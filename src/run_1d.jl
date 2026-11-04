"""Run the existing 1D analysis on SBP transverse integrals.

The notebook/MADBead routines are included directly. The additional geometry
driver shares disk gaps and thread heights across the measurement band.
"""
module SBP1D

using LinearAlgebra, Statistics, Random, DataFrames, Measurements, LsqFit, NPZ
include("madbead/MADBead.jl")
include("existing_1d_functions.jl")

scalar(data, key) = data[key] isa AbstractArray ? only(data[key]) : data[key]

function model_priors(data)
    # Same uncertainties as the 5_ notebook; central geometry is our measured setup.
    gaps = vec(data["gap_widths_m"])
    p = Dict(
        "E_0" => 1.0 ± 1.0,
        "z_m" => Float64(scalar(data, "mirror_z_m")) ± 0.0,
        "σ_m" => Float64(scalar(data, "mirror_conductivity_S_m")) ± 1e7,
        "n_disk" => Float64(length(gaps)) ± 0.0,
    )
    for i in eachindex(gaps)
        p["d_v_$i"] = Float64(gaps[i]) ± 0.1e-3
        p["d_d_$i"] = Float64(scalar(data, "disk_thickness_m")) ± 0.05e-3
        p["ϵ_d_$i"] = Float64(scalar(data, "disk_eps_r")) ± 0.1
        p["tanD_d_$i"] = Float64(scalar(data, "disk_loss_tangent")) ± 1e-6
    end
    p["r_b"] = Float64(scalar(data, "bead_diameter_m"))/2 ± 15e-6/2
    p["ϵ_b"] = Float64(scalar(data, "bead_eps_r")) ± 0.2
    return p
end

function check_geometry_bounds(gaps, z, thickness, bead_radius, gap_bound, z_bound)
    any(gaps .- gap_bound .<= 0) && error("Gap bounds permit nonpositive air gaps.")
    any(z .- z_bound .< bead_radius) && error("Thread bounds permit mirror overlap.")
    edge = 0.0
    for (i, gap) in enumerate(gaps)
        edge += gap
        motion = i*gap_bound + z_bound
        any((z .> edge-motion-bead_radius) .&
            (z .< edge+thickness+motion+bead_radius)) &&
            error("Geometry bounds allow a bead to overlap a disk.")
        edge += thickness
    end
end

function fit_shared_geometry(dataset, frequencies, priors, input)
    fit_gaps = Bool(scalar(input, "fit_gaps"))
    fit_positions = Bool(scalar(input, "fit_positions"))
    gaps0 = Float64.(vec(input["gap_widths_m"]))
    z0 = Float64.(vec(input["z_from_mirror_m"]))
    dg = fit_gaps ? Float64(scalar(input, "gap_bound_m")) : 0.0
    dz = fit_positions ? Float64(scalar(input, "position_bound_m")) : 0.0
    check_geometry_bounds(gaps0, z0, Float64(scalar(input, "disk_thickness_m")),
                          Float64(scalar(input, "bead_diameter_m"))/2, dg, dz)
    ng = fit_gaps ? length(gaps0) : 0
    nz = fit_positions ? length(z0) : 0
    ng+nz == 0 && return gaps0, z0
    initial = vcat(fit_gaps ? gaps0 : Float64[], fit_positions ? z0 : Float64[])*1000
    bounds = vcat(fill(dg, ng), fill(dz, nz))*1000
    penalty = Float64(scalar(input, "geometry_penalty"))
    observed = abs.(dataset.ET)
    signal = sqrt.(sum(abs2.(observed), dims=2))[:]
    active = findall(signal .> eps(Float64))
    isempty(active) && error("No nonzero spider signal constrains geometry.")
    delta = [MADBead.calc_δ_e_mie(value_of(priors["ϵ_b"]), value_of(priors["r_b"]); f=f)
             for f in frequencies]
    deltac = [MADBead.calc_δ_c_mie(value_of(priors["ϵ_b"]), value_of(priors["r_b"]); f=f)
              for f in frequencies]

    function unpack(q)
        gaps = fit_gaps ? q[1:ng]/1000 : gaps0
        z = fit_positions ? q[ng+1:end]/1000 : z0
        return gaps, z
    end

    function residual(q::AbstractVector{T}) where T
        gaps, z = unpack(q)
        p = Dict{String,T}(key => value_of(value) for (key,value) in priors)
        p["E_0"] = one(T)
        for i in eachindex(gaps)
            p["d_v_$i"] = gaps[i]
        end
        z_abs = p["z_m"] .- z
        values = T[]
        for k in active
            model = sqrt.(abs.(MADBead.E_field2_conv_1D_z_param(
                z_abs, p; f=frequencies[k], δ=delta[k], δc=deltac[k])))
            amplitude = dot(model, observed[k,:])/sum(abs2, model)
            append!(values, (amplitude*model-observed[k,:])/(signal[k]*sqrt(length(active))))
        end
        append!(values, sqrt(penalty)*(q-initial)./bounds)
        return values
    end

    target = zeros(length(active)*length(z0)+length(initial))
    fitted = curve_fit((_,q) -> residual(q), target, target, initial;
                       lower=initial-bounds, upper=initial+bounds,
                       autodiff=:forwarddiff, maxIter=300)
    fitted.converged || error("Shared geometry fit did not converge.")
    return unpack(coef(fitted))
end

function analyze(input_file, output_file)
    input = npzread(input_file)
    frequencies = vec(Float64.(input["frequencies_Hz"]))
    all((frequencies .>= 20e9) .& (frequencies .<= 22e9)) ||
        error("Reconstruct the raw spider data using only 20–22 GHz first.")
    issorted(frequencies) || error("Frequencies must be sorted.")
    n_mc = Int(scalar(input, "n_mc"))
    n_mc >= 2 || error("Monte Carlo propagation requires at least two draws.")
    Random.seed!(Int(scalar(input, "seed")))
    priors = model_priors(input)
    z0 = vec(Float64.(input["z_from_mirror_m"]))
    original = build_dataset("spider", input["ET"], z0, frequencies;
        z_mirror=value_of(priors["z_m"]), rel_err=Float64(scalar(input, "relative_error")))
    # Keep the supplied T0–T3 order after the notebook's absolute-z sorting.
    undo = invperm(sortperm(value_of(priors["z_m"]) .- z0))
    original = merge(original, (ET=original.ET[:,undo], ET_meas=original.ET_meas[:,undo],
                               z_rel=z0, z_abs=value_of(priors["z_m"]) .- z0))
    outliers = detect_reduced_ET_outliers(original, frequencies;
        window=Int(scalar(input, "outlier_window")),
        threshold=Float64(scalar(input, "outlier_threshold")))
    keep = outliers.keep_frequency_mask
    any(keep) || error("The existing ET outlier filter rejected every frequency.")
    dataset = merge(original, (ET=original.ET[keep,:], ET_meas=original.ET_meas[keep,:]))
    f = frequencies[keep]
    gaps, z = fit_shared_geometry(dataset, f, priors, input)
    for i in eachindex(gaps)
        priors["d_v_$i"] = gaps[i] ± error_of(priors["d_v_$i"])
    end
    dataset = merge(dataset, (z_abs=value_of(priors["z_m"]) .- z, z_rel=z))

    # Geometry is the SBP amendment. Everything from here uses the existing
    # notebook's amplitude fit, MC propagation, field integration and boost.
    optimized = ["E_0"]
    helpers = ["n_disk"]
    fixed = [setdiff(keys(priors), optimized, helpers)...]
    result = run_boostfactor_analysis(dataset, f, priors, optimized, fixed, helpers,
        n_mc; P_in=Float64(scalar(input, "input_power_W")),
        A=π*Float64(scalar(input, "disk_radius_m"))^2)
    nf, nt = length(f), length(z)
    predicted = zeros(nt,nf)
    field = zeros(ComplexF64,nt,nf)
    model_integral = zeros(ComplexF64,nf)
    for k in eachindex(f)
        p = Dict(key => Float64(value_of(result.p_final_ν[k,key])) for key in keys(priors))
        de = MADBead.calc_δ_e_mie(p["ϵ_b"],p["r_b"];f=f[k])
        dc = MADBead.calc_δ_c_mie(p["ϵ_b"],p["r_b"];f=f[k])
        predicted[:,k] = sqrt.(abs.(MADBead.E_field2_conv_1D_z_param(dataset.z_abs,p;f=f[k],δ=de,δc=dc)))
        field[:,k] = MADBead.E_field_1D_z_param(dataset.z_abs,p;f=f[k])
        model_integral[k] = MADBead.int_dz_E_param(p;f=f[k])
    end
    output = Dict{String,Any}(
        "frequencies_Hz"=>f, "z_from_mirror_m"=>z,
        "z_nominal_m"=>z0, "gap_widths_nominal_m"=>vec(input["gap_widths_m"]),
        "gap_widths_m"=>gaps, "ET_measured"=>permutedims(dataset.ET),
        "amplitude"=>value_of.(result.p_final_ν[!,"E_0"]),
        "ET_fitted_magnitude"=>predicted, "ET_unperturbed"=>field,
        "integral_ET_dz"=>value_of.(result.∫dV_E),
        "integral_ET_dz_error"=>error_of.(result.∫dV_E),
        "integral_ET_dz_model"=>model_integral,
        "integral_ET_dz_mc"=>result.int_dz_E_mc,
        "beta2"=>value_of.(result.boostfactor),
        "beta2_error"=>error_of.(result.boostfactor),
        "residual_rms"=>sqrt.(mean(abs2.(predicted-abs.(permutedims(dataset.ET))),dims=1))[:],
        "input_frequencies_Hz"=>frequencies, "keep_frequency_mask"=>Int.(keep),
        "outlier_point_mask"=>Int.(outliers.point_mask),
        "n_mc"=>n_mc, "relative_error"=>scalar(input,"relative_error"),
    )
    for key in keys(priors)
        # ASCII archive keys avoid ambiguous ZIP filename encodings across languages.
        name = replace(key, "ϵ"=>"eps", "σ"=>"sigma")
        output["parameter__"*name] = value_of.(result.p_final_ν[!,key])
        output["parameter_error__"*name] = error_of.(result.p_final_ν[!,key])
        output["mc_parameter__"*name] = permutedims(reduce(hcat,result.p_all_ν_mc[!,key]))
    end
    npzwrite(output_file,output)
end

if abspath(PROGRAM_FILE) == @__FILE__
    length(ARGS)==2 || error("usage: run_1d.jl input.npz output.npz")
    analyze(ARGS...)
end

end
