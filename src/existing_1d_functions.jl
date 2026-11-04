# Analysis functions taken from 5_Analysis_Boostfactor_Determination_Spider_ET.ipynb.
# Only the notebook's ET-to-boost functions are included; raw SBP preparation is separate.
value_of(x) = hasproperty(x, :val) ? getfield(x, :val) : x
error_of(x) = hasproperty(x, :err) ? getfield(x, :err) : zero(value_of(x))

function assert_finite_vector(name, values)
    @assert ndims(values) == 1 "$(name) must be a vector."
    @assert !isempty(values) "$(name) must not be empty."
    @assert all(isfinite.(values)) "$(name) contains non-finite values."
    return values
end


function orient_ET(name, ET_zf, ν, z_rel)
    @assert ndims(ET_zf) == 2 "$(name) must be a 2D array with shape (z, frequency)."
    @assert size(ET_zf, 1) == length(z_rel) "$(name) z dimension does not match its z array."
    @assert size(ET_zf, 2) == length(ν) "$(name) frequency dimension does not match frequencies."
    finite_mask = isfinite.(real.(ET_zf)) .& isfinite.(imag.(ET_zf))
    bad_count = count(.!finite_mask)
    if bad_count > 0
        @warn "$(name) contains $(bad_count) non-finite ET value(s); cleanup will skip affected reduced frequencies."
    end
    return permutedims(ET_zf, (2, 1))
end


function make_ET_measurements(ET_fz; rel_err=0.05)
    @assert 0 <= rel_err < 1 "rel_err should be a fractional uncertainty."
    real_err = rel_err .* abs.(real.(ET_fz))
    imag_err = rel_err .* abs.(imag.(ET_fz))
    return (real.(ET_fz) .± real_err) .+ 1im .* (imag.(ET_fz) .± imag_err)
end


function build_dataset(label, ET_zf, z_rel, ν; z_mirror, rel_err)
    z_rel = vec(Float64.(z_rel))
    ET_zf = Array{ComplexF64}(ET_zf)
    ET_fz = orient_ET(label, ET_zf, ν, z_rel)
    z_abs = z_mirror .- z_rel
    @assert all(z_abs .< z_mirror) "$(label) z values must lie left of the mirror."

    order = sortperm(z_abs)
    ET_fz = ET_fz[:, order]
    z_rel = z_rel[order]
    z_abs = z_abs[order]

    return (
        label=label,
        z_rel=z_rel,
        z_abs=z_abs,
        ET=ET_fz,
        ET_meas=make_ET_measurements(ET_fz; rel_err=rel_err),
    )
end


function robust_median_mad(values)
    finite_values = values[isfinite.(values)]
    if isempty(finite_values)
        return 0.0, 0.0
    end
    med = median(finite_values)
    scale = 1.4826 * median(abs.(finite_values .- med))
    return med, scale
end


function hampel_component_flags(values; window=51, threshold=12.0)
    @assert isodd(window) "Hampel window must be odd."
    @assert window >= 3 "Hampel window must contain at least 3 points."
    x = Float64.(values)
    n = length(x)
    flags = falses(n)
    scores = zeros(Float64, n)

    global_med, global_scale = robust_median_mad(x)
    half_window = window ÷ 2

    for i in eachindex(x)
        if !isfinite(x[i])
            flags[i] = true
            scores[i] = Inf
            continue
        end

        lo = max(firstindex(x), i - half_window)
        hi = min(lastindex(x), i + half_window)
        neighbor_idx = vcat(collect(lo:i-1), collect(i+1:hi))
        neighbors = x[neighbor_idx]
        finite_neighbors = neighbors[isfinite.(neighbors)]

        local_med, local_scale = isempty(finite_neighbors) ? (global_med, global_scale) : robust_median_mad(finite_neighbors)
        scale = max(local_scale, global_scale, eps(Float64))
        scores[i] = abs(x[i] - local_med) / scale
        flags[i] = scores[i] > threshold
    end

    return flags, scores
end


function detect_reduced_ET_outliers(dataset, ν; window=51, threshold=12.0)
    @assert size(dataset.ET, 1) == length(ν) "Reduced ET frequency dimension must match ν."

    point_mask = falses(size(dataset.ET))
    real_score = zeros(Float64, size(dataset.ET))
    imag_score = zeros(Float64, size(dataset.ET))

    for z_idx in axes(dataset.ET, 2)
        real_flags, real_scores = hampel_component_flags(real.(dataset.ET[:, z_idx]); window=window, threshold=threshold)
        imag_flags, imag_scores = hampel_component_flags(imag.(dataset.ET[:, z_idx]); window=window, threshold=threshold)
        point_mask[:, z_idx] .= real_flags .| imag_flags
        real_score[:, z_idx] .= real_scores
        imag_score[:, z_idx] .= imag_scores
    end

    frequency_skip_mask = vec(any(point_mask, dims=2))
    keep_frequency_mask = .!frequency_skip_mask

    return (
        point_mask=point_mask,
        real_score=real_score,
        imag_score=imag_score,
        frequency_skip_mask=frequency_skip_mask,
        keep_frequency_mask=keep_frequency_mask,
        skipped_frequencies=ν[frequency_skip_mask],
        window=window,
        threshold=threshold,
    )
end


function final_parameter_table(ν, p_all_ν_mc, p0_all)
    p_final_ν = DataFrame("f"=>ν)
    for key in keys(p0_all)
        p_final_ν[!, key] = mean.(p_all_ν_mc[!, key]) .± std.(p_all_ν_mc[!, key])
    end
    return p_final_ν
end


function integrate_mc_fields(ν, p_all_ν_mc, p0_all, N_mc)
    int_dz_E_mc = zeros(ComplexF64, length(ν), N_mc)
    for f in 1:length(ν), i in 1:N_mc
        p_dict = Dict(key=>p_all_ν_mc[f, key][i] for key in keys(p0_all))
        int_dz_E_mc[f, i] = MADBead.int_dz_E_param(p_dict; f=ν[f])
    end
    return int_dz_E_mc
end


function boostfactor_from_int_dz(ν, int_dz_E_mc; P_in, A, J_0=1)
    ∫dV_E = mean(abs.(int_dz_E_mc), dims=2)[:] .± std(abs.(int_dz_E_mc), dims=2)[:]
    P_sig = J_0^2 / (16 * P_in) .* abs2.(∫dV_E)
    P_0 = MADBead.c_const * A * J_0^2 ./ (2 * MADBead.ϵ0 * (2π .* ν).^2)
    boostfactor = P_sig ./ P_0
    return ∫dV_E, boostfactor
end


function ET_dataframe(ν, dataset)
    df_ET = DataFrame("f"=>ν)
    df_ET[!, "ET"] = [dataset.ET_meas[f, :] for f in 1:length(ν)]
    df_ET[!, "z_abs"] = [dataset.z_abs for _ in 1:length(ν)]
    df_ET[!, "z_rel"] = [dataset.z_rel for _ in 1:length(ν)]
    return df_ET
end


function run_boostfactor_analysis(dataset, ν, p0_all, keys_optim, keys_fixed, keys_helper, N_mc; P_in, A)
    println("Running $(dataset.label) fit with $(length(dataset.z_abs)) z points and $(length(ν)) frequencies.")

    p_all_ν_mc = MADBead.fit_E_z_MC(ν, dataset.z_abs, dataset.ET_meas, p0_all, keys_optim, keys_fixed, keys_helper, N_mc)
    p_final_ν = final_parameter_table(ν, p_all_ν_mc, p0_all)
    int_dz_E_mc = integrate_mc_fields(ν, p_all_ν_mc, p0_all, N_mc)
    ∫dV_E, boostfactor = boostfactor_from_int_dz(ν, int_dz_E_mc; P_in=P_in, A=A)

    df_bf_analysis = DataFrame("f"=>ν, "bf"=>boostfactor, "int_dV_E"=>∫dV_E)
    df_bf_analysis = innerjoin(df_bf_analysis, p_final_ν, on="f")

    df_bf_mc_analysis = DataFrame("f"=>ν)
    df_bf_mc_analysis[!, "int_dV_E_mc"] = [int_dz_E_mc[f, :] for f in 1:length(ν)]
    df_bf_mc_analysis = innerjoin(df_bf_mc_analysis, p_all_ν_mc, on="f")

    return (
        label=dataset.label,
        p_all_ν_mc=p_all_ν_mc,
        p_final_ν=p_final_ν,
        int_dz_E_mc=int_dz_E_mc,
        ∫dV_E=∫dV_E,
        boostfactor=boostfactor,
        df_ET=ET_dataframe(ν, dataset),
        df_bf_analysis=df_bf_analysis,
        df_bf_mc_analysis=df_bf_mc_analysis,
    )
end
