function fit_param_to_model(p::Dict{String,T};f=22e9) where T<:Real
    E_in = p["E_0"]
    E_in = Complex{T}(E_in)
#    if length(p)>2
    z_m, σ_m = p["z_m"],p["σ_m"]
    d_v_i = []
    d_d_i = []
    ϵ_d_i = []
    tanD_d_i = []
    for i in 1:Int(p["n_disk"])
        push!(d_v_i,p["d_v_$i"])
        push!(d_d_i,p["d_d_$i"])
        push!(ϵ_d_i,p["ϵ_d_$i"])
        push!(tanD_d_i,p["tanD_d_$i"])
    end
    ϵ0 = 8.85418782e-12
    ϵ_σ = (1+1im*σ_m/(2π*f*ϵ0))
    if Int(p["n_disk"])<1
        d_v_last = z_m
    else
        d_v_last = z_m .- sum(d_v_i.+d_d_i)
    end
    n_region = length(d_v_i)+length(d_d_i)+3
    distance = zeros(T,n_region)
    distance[2:2:end-3] .= d_v_i
    distance[3:2:end-2] .= d_d_i
    distance[end-1] = d_v_last
    epsilon = ones(Complex{T},n_region)
    epsilon[1] = ϵ_σ
    epsilon[3:2:end-2] .= ϵ_d_i.*(1 .+ 1im*tanD_d_i)
    return distance, epsilon, E_in
end






function E_field_1D_z_param(z,p::Dict{String,T}; f=22e9) where T<:Real
    distance, epsilon, E_in = fit_param_to_model(p,f=f)
    E = E_field_1D_z(z,distance,epsilon,E_in;f=f)
    return E
end

function E_field2_conv_1D_z_param(z,p::Dict{String,T}; f=22e9, δ=1.0+0.0im,δc=1.0+0.0im) where T<:Real
    distance, epsilon, E_in = fit_param_to_model(p,f=f)
    E2_avg = E_field2_conv_1D_z(z,distance,epsilon,E_in;f=f,δ=δ,δc=δc)
    return E2_avg
end






function E_field_1D_z(z,distance::Array{T},epsilon::Array{Complex{T}},E_in::Complex{T};f=22e9,scale_region=2) where T<:Real
    k0 = 2*pi*f/c_const
    Nregions = length(distance)
    E = zeros(Complex{T},length(z))
    input_region = 2
    n_r = sqrt(epsilon[1])

    ρ0 = (1-n_r)/(1+n_r)#Fresnel reflection coeff for mirror
    RL = [E_in, ρ0*E_in]
    E_scale = 1
    z_r = sum(distance[input_region:end])
    idx_sub = z.>z_r
    z_sub = z_r .- z[idx_sub]
    E[idx_sub] = (1+ρ0)*RL[1] * exp.(-1im*k0*n_r*z_sub)
    for r in input_region:Nregions-1
        if r==scale_region
            E_scale = RL[1]
        end
        n_r = sqrt(epsilon[r])
        d_r = distance[r]
        idx_sub = z_r.>=z.>z_r-d_r
        z_sub = z_r .- z[idx_sub]
        phase_term = exp.(-1im*n_r*k0*z_sub)
        E_sub = RL[1]*phase_term .+ RL[2]*conj.(phase_term)
        E[idx_sub] = E_sub
        # Propagate to next region
        n_rp1 = sqrt(epsilon[r+1])
        P_r = diagm([exp(-1im*n_r*k0*d_r),exp(+1im*n_r*k0*d_r)])
        G_r = 1/(2*n_rp1)*[(n_rp1+n_r) (n_rp1-n_r) ; (n_rp1-n_r) (n_rp1+n_r)]
        RL = G_r * P_r * RL
        z_r = z_r-d_r
    end
    if Nregions==scale_region
        E_scale = RL[1]
    end
    n_r = sqrt(epsilon[end])
    z_r = 0
    idx_sub = z.<z_r
    z_sub = z_r .- z[idx_sub]
    phase_term = exp.(-1im*n_r*k0*z_sub)
    E_sub = RL[1]*phase_term .+ RL[2]*conj.(phase_term)
    E[idx_sub] = E_sub
    return E_in/E_scale*E
end

function E_field2_conv_1D_z(z,distance::Array{T},epsilon::Array{Complex{T}},E_in::Complex{T};f=22e9,δ=1.0+0.0im,δc=1.0+0.0im) where T<:Real
    k0 = 2*pi*f/c_const
    Nregions = length(distance)
    E2_conv = zeros(Complex{T},length(z))
    input_region = 2
    n_r = sqrt(epsilon[1])

    ρ0 = (1-n_r)/(1+n_r)#Fresnel reflection coeff for mirror
    RL = [E_in, ρ0*E_in]
    z_r = sum(distance[input_region:end])
    idx_sub = z.>z_r
    z_sub = z_r .- z[idx_sub]
    E2_conv[idx_sub] = (1+ρ0)*δ*RL[1]^2 * exp.(-2im*k0*n_r*z_sub)
    for r in input_region:Nregions-1
        n_r = sqrt(epsilon[r])
        d_r = distance[r]
        idx_sub = z_r.>=z.>z_r-d_r
        z_sub = z_r .- z[idx_sub]
        phase_term = exp.(-2im*n_r*k0*z_sub)
        E2_sub = δ*(RL[1]^2*phase_term .+ RL[2]^2*conj.(phase_term)) .+ 2*δc*RL[1]*RL[2]
        E2_conv[idx_sub] = E2_sub
        # Propagate to next region
        n_rp1 = sqrt(epsilon[r+1])
        P_r = diagm([exp(-1im*n_r*k0*d_r),exp(+1im*n_r*k0*d_r)])
        G_r = 1/(2*n_rp1)*[(n_rp1+n_r) (n_rp1-n_r) ; (n_rp1-n_r) (n_rp1+n_r)]
        RL = G_r * P_r * RL
        z_r = z_r-d_r
    end
    n_r = sqrt(epsilon[end])
    z_r = 0
    idx_sub = z.<z_r
    z_sub = z_r .- z[idx_sub]
    phase_term = exp.(-2im*n_r*k0*z_sub)
    E2_sub = δ*(RL[1]^2*phase_term .+ RL[2]^2*conj.(phase_term)) .+ 2*δc*RL[1]*RL[2]
    E2_conv[idx_sub] = E2_sub
    return E2_conv
end

function field_regions(distance::Array{T},epsilon::Array{Complex{T}},E_in::Complex{T};f=22e9) where T<:Real
    k0 = 2*pi*f/c_const
    Nregions = length(distance)
    field_regions = zeros(Complex{T},length(distance),2)
    input_region = 2
    n_r = sqrt(epsilon[1])

    ρ0 = (1-n_r)/(1+n_r)#Fresnel reflection coeff for mirror
    RL = [E_in, ρ0*E_in]
    field_regions[input_region,:] = RL
    field_regions[1,:] = [(1+ρ0)*RL[1],0*RL[1]]
    for r in input_region:Nregions-1
        n_r = sqrt(epsilon[r])
        d_r = distance[r]
        n_rp1 = sqrt(epsilon[r+1])
        P_r = diagm([exp(-1im*n_r*k0*d_r),exp(+1im*n_r*k0*d_r)])
        G_r = 1/(2*n_rp1)*[(n_rp1+n_r) (n_rp1-n_r) ; (n_rp1-n_r) (n_rp1+n_r)]
        RL = G_r * P_r * RL
        field_regions[r+1,:] = RL
    end
    return field_regions
end


function int_dz_E_param(p;f=22e9)
    d_r, epsilon, E_in = fit_param_to_model(p,f=f)
    n_r = sqrt.(epsilon)
    RL = field_regions(d_r,epsilon,E_in;f=f)
    R_r = RL[:,1]
    L_r = RL[:,2]
    k0 = 2*pi*f/c_const
    intE_0 = (R_r[end] - L_r[end])/n_r[end]
    phase_term = exp.(-1im*k0*n_r[2:end-1].*d_r[2:end-1])
    intE_r = -(R_r[2:end-1].*(phase_term.-1) .- L_r[2:end-1].*(conj.(phase_term).-1))./n_r[2:end-1]
    intE_m = -(R_r[1] - L_r[1])/n_r[1]

    intE = (intE_0 + sum(intE_r) + intE_m)/k0
    return intE
end


function fit_E_z_MC(ν,z,∫dA_E,p0_all,keys_optim,keys_fixed,keys_helper,N_mc)
    keys_all = [keys_optim...,keys_fixed...]
    nparam_optim = length(keys_optim)
    nparam_all = length(p0_all)
    p_all_ν_mc = DataFrame("f"=>ν)
    p0_rand = DataFrame((i=>(getfield(p0_all[i],:val).+getfield(p0_all[i],:err)*randn(N_mc)) for i in keys(p0_all))...)
    [p_all_ν_mc[!,key].=[p0_rand[!,key]] for key in keys(p0_all)];

    r_b = p0_all["r_b"]
    V_b = 4/3*π*r_b^3
    ϵ_b = p0_all["ϵ_b"]
    α_0 = 3*(ϵ_b-1)*V_b/(ϵ_b+2)

    r_b_rand = p0_rand[!,"r_b"]
    V_b_rand = 4/3*π*r_b_rand.^3
    ϵ_b_rand = p0_rand[!,"ϵ_b"]
    α_0_rand = @. 3*(ϵ_b_rand-1)*V_b_rand/(ϵ_b_rand+2)


    δ_mie_rand = [MADBead.calc_δ_e_mie(ϵ_b_rand[i],r_b_rand[i];f=f) for i in 1:N_mc, f in ν]
    δc_mie_rand = [MADBead.calc_δ_c_mie(ϵ_b_rand[i],r_b_rand[i];f=f) for i in 1:N_mc, f in ν];

    p_res = zeros(length(keys_optim),length(ν),N_mc)
    lower = fill(-Inf, nparam_optim)
    upper = fill(Inf, nparam_optim)
    for (k, key) in enumerate(keys_optim)
        if startswith(key, "d_v_") || startswith(key, "d_d_") || key == "r_b"
            lower[k] = 0.0
        end
    end
    Threads.@threads for f in 1:length(ν)
        ref = getfield.(abs.(∫dA_E[f,:]),:val)
        ref_err = abs.(getfield.(abs.(∫dA_E[f,:]),:err))
        ref_err[.!isfinite.(ref_err)] .= 0.0
        for i in 1:N_mc
            ref_rand = ref .+ ref_err .* randn(length(ref))

            scale = α_0.val/α_0_rand[i]
            if !(isfinite(scale) && scale > 0)
                scale = 1.0
            end
            ref_rand = ref_rand .* sqrt(scale)

            finite_mask = isfinite.(z) .& isfinite.(ref_rand)
            z_fit = z[finite_mask]
            ref_fit = ref_rand[finite_mask]

            p0_optim = [p0_rand[i,key] for key in keys_optim]
            p0_optim[1] = mean(ref_fit)/2
            function model(z,p::Array{T}) where T<:Real
                p1 = Dict{String,T}()
                [p1[keys_optim[j]]=p[j] for j in 1:length(p)]
                [p1[key]=p0_rand[i,key] for key in keys_fixed]
                [p1[key]=p0_all[key].val for key in keys_helper]
                E_abs = sqrt.(abs.(MADBead.E_field2_conv_1D_z_param(z,p1;
                            f=ν[f],δ=δ_mie_rand[i,f],δc=δc_mie_rand[i,f])))
                if any(!isfinite, E_abs)
                    return fill(1e12, length(z))
                end
                return E_abs
            end
            try
                fitlsq = curve_fit(model,z_fit,ref_fit,p0_optim;autodiff=:forwarddiff,lower=lower,upper=upper)
                p_fit = coef(fitlsq)
                if all(isfinite, p_fit)
                    p_res[:,f,i] = p_fit
                else
                    p_res[:,f,i] = p0_optim
                end
            catch
                p_res[:,f,i] = p0_optim
            end
        end
    end
    for f in 1:length(ν),k in 1:length(keys_optim)
            p_all_ν_mc[f,keys_optim[k]] = p_res[k,f,:]
    end
    return  p_all_ν_mc
end
