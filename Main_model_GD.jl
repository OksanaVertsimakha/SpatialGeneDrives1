using Pkg
#= Pkg.add("DataFrames")
Pkg.add("BSON")
Pkg.add("SplitApplyCombine")
Pkg.add("CSV")
Pkg.add("PrettyTables")
Pkg.add("Random")
Pkg.add("Distributions")
Pkg.add("Statistics")
Pkg.add("StatsBase")
Pkg.add("CategoricalArrays")
Pkg.add("Distances")
Pkg.add("Hexagons")
Pkg.add("GeometryBasics")
Pkg.add("ProgressMeter")
Pkg.add("LinearAlgebra")
Pkg.add("StatsBase")
Pkg.add("CairoMakie")
Pkg.add("JLD2")
Pkg.add("NamedArrays")
Pkg.add("StaticArrays") 
Pkg.add("SparseArrays")
Pkg.add("DoubleFloats")
Pkg.add("Distributed")
Pkg.add("FilePathsBase")
Pkg.add("GLM") =#
using BSON
using CategoricalArrays
using Random
using Distributions
using DataFrames
using Distances
using StatsBase
using GLM
using CairoMakie
using StaticArrays
using GeometryBasics
using CSV
using ProgressMeter
using LinearAlgebra
using PrettyTables
using JLD2
using NamedArrays
using Integrals
using DoubleFloats
using LinearAlgebra
using SparseArrays
using Base.Threads
using FilePathsBase
using Distributed



# Core model elements
function Mating(; Nf, Nm, E, ff, r_sex, beta=0, M2=10, M=4, tp=Int64,wa=1)
    N_pairs_A = zeros(tp, M2, M2)
    N_pairs_Af = zeros(tp, M2, M2)
    N_exp_f = zeros(tp, M2)
    N_exp_m = zeros(tp, M2)
    
    Nm_sum = sum(Nm.*wa)
    if Nm_sum == 0
        return N_exp_f, N_exp_m  # Early return if no males
    end
    pAm = Nm.*wa ./ Nm_sum
    p_beta = Nm_sum / (beta + Nm_sum)

    # Vectorized mating pair sampling
    @inbounds for i in 1:M2
        if Nf[i] > 0
            n_pairs = round(tp, Nf[i] * p_beta)
            if n_pairs > 0
                N_pairs_A[i, :] .= rand(Multinomial(n_pairs, pAm))
            end
        end
    end

    # Vectorized offspring production
    @inbounds for j in 1:M2, i in 1:M2
        if N_pairs_A[i, j] > 0
            N_pairs_Af[i, j] = rand(Poisson(N_pairs_A[i, j] * ff[i, j]))
            
            if N_pairs_Af[i, j] > 0
                N_exp = rand(Multinomial(N_pairs_Af[i, j], E[i][j]))
                # Sex allocation 
                N_sx = rand.(Binomial.(N_exp, 1 .- r_sex[j, i]))
                
                N_exp_f .+= N_sx
                N_exp_m .+= N_exp .- N_sx
            end
        end
    end

    return N_exp_f, N_exp_m
end

function Fitness_change(; Nf, Nm, wf, wm, M2=10, tp=Int64)
    Nf_new = Vector{tp}(undef, M2)
    Nm_new = Vector{tp}(undef, M2)
    
    @inbounds for i in 1:M2
        Nf_new[i] = rand(Binomial(Nf[i], wf[i]))
        Nm_new[i] = rand(Binomial(Nm[i], wm[i]))
    end
    
    return Nf_new, Nm_new
end

#Density-dependent mortality
function Alpha(;N, f,phi1=1, phi2=1,tp=Int64)
    if phi2==1
        a = (N*f/2)/(phi1*f/2-1)
    else 
        a= (N*f/2)/(phi1*f/2-1)^(1/phi2)
    end
    return a
end

function Ddm_change(; Nf, Nm, Nh, alpha, M2=10, tp=Int64,phi1=1,phi2=1)
    p = phi1 / ((Nh/alpha)^phi2 + 1)
    
    Nf_new = Vector{tp}(undef, M2)
    Nm_new = Vector{tp}(undef, M2)
    
    @inbounds for i in 1:M2
        Nf_new[i] = rand(Binomial(Nf[i], p))
        Nm_new[i] = rand(Binomial(Nm[i], p))
    end
    
    return Nf_new, Nm_new
end
function Migration_change(; Nf_s, Nm_s, A, w, Nsites, mu_mig=1.0, M2=10, tp=Int64)
    Nm_new = zeros(tp, M2, Nsites)
    Nf_new = zeros(tp, M2, Nsites)

    # Migration scaling
    w_scaled = mu_mig .* w
    
    p = collect(eachcol(A)) 
    local_storage = [zeros(tp, Nsites) for _ in 1:Threads.nthreads()]

    @threads for i in 1:M2
        tid = Threads.threadid()
        local_buf = local_storage[tid]

        #### --- Males --- ####
        fill!(local_buf, zero(tp))
        for site in 1:Nsites
            n_m = round(tp, Nm_s[i, site] * w_scaled[site])
            if n_m > 0
                @inbounds local_buf .+= rand(Multinomial(n_m, p[site]))
            end
        end
        @inbounds @simd for site in 1:Nsites
            Nm_new[i, site] = local_buf[site]
        end

        #### --- Females --- ####
        fill!(local_buf, zero(tp))
        for site in 1:Nsites
            n_f = round(tp, Nf_s[i, site] * w_scaled[site])
            if n_f > 0
                @inbounds local_buf .+= rand(Multinomial(n_f, p[site]))
            end
        end
        @inbounds @simd for site in 1:Nsites
            Nf_new[i, site] = local_buf[site]
        end
    end

    return Nf_new, Nm_new
end

### Gene drive models 
# Pupal mortality
Model2 = function (; Nf, Nm, E, wm, wf, alpha=10^6, ff, r_sex=0.5, beta=1, M2=10, M=4,tp=Int64,phi1=1,phi2=1,wa=1)
    Nf, Nm = Mating(; Nf=Nf, Nm=Nm, E=E, ff=ff, r_sex=r_sex, beta=beta, M2=M2, M=M,tp=tp,wa=wa)
    Nh = sum(Nf .+ Nm)
    Nf_1, Nm_1 = Ddm_change(; Nf=Nf, Nm=Nm, Nh=Nh, alpha=alpha, M2=M2,phi1=phi1,phi2=phi2)[1:2]
    Nf_2, Nm_2 = Fitness_change(; Nf=Nf_1, Nm=Nm_1, wf=wf, wm=wm, M2=M2)
    return [Nf_2, Nm_2]#Ntf, Ntm, Ntfi, Ntmi
end

# Embrionic mortality
Model1 = function (; Nf, Nm, E, wm, wf, alpha, ff, r_sex=0.5, beta=1, M2=10, M=4,tp=Int64,phi1=1,phi2=1,wa=1)
    Nf, Nm = Mating(; Nf=Nf, Nm=Nm, E=E, ff=ff, r_sex=r_sex, beta=beta, M2=M2, M=M,tp=tp,wa=wa)
    Nf_1, Nm_1 = Fitness_change(; Nf=Nf, Nm=Nm, wf=wf, wm=wm, M2=M2)
    Nh = sum(Nf_1 .+ Nm_1)
    Nf_2, Nm_2 = Ddm_change(; Nf=Nf_1, Nm=Nm_1, Nh=Nh, alpha=alpha, M2=M2,phi1=phi1,phi2=phi2)[1:2]
    return [Nf_2, Nm_2]
end

Input_nosp = function (; M=4, Nmean=10^5, pm=0.01,tp=Int64)
    M2 = Int(M * (M + 1) / 2)
    Ntf_init = zeros(tp,M2)
    Ntf_init[1] = tp(round(Nmean / 2))
    Ntm_init = zeros(tp,M2)
    Ntm_init[1] = tp(round(Nmean / 2))
    Ntm_init[2] = ceil(tp, Ntm_init[1] * pm)
    Ntm_init[1] = ceil(tp, Ntm_init[1] * (1 - pm))
    return Ntf_init, Ntm_init
end

# Tracking the population dynamics by genotype
function Model_nosp_T(; N=10^15, pm=0.01, wmf, e, mD_sex, beta=0, alpha=0, phi1=1,phi2=1,maxgen=10, Model=2,M=4,tp=Int64,b2=0,wa=1)
    Ntf_init, Ntm_init = Input_nosp(; M=M, Nmean=N, pm=pm,tp=tp)
   
    M2 = Int(M * (M + 1) / 2)
    f=wmf[3][1,1]
    alpha = alpha == 0 ? Alpha(phi1=phi1,phi2=phi2,f=f,N=N) : alpha
    r_sex = 0.5 .* ones(M2, M2)
    if M == 4
        for i in [2, 5, 6, 7]
            r_sex[i, :] .= mD_sex
        end
    end
    if M == 5
        for i in [2, 6, 7, 8, 9]
            r_sex[i, :] .= mD_sex
        end
    end
    if b2>0     
        if M==4 r_sex[7,:] .=0.5;end
        if M==5 r_sex[9,:] .=0.5 ;end
    end
    model = Model == 2 ? Model2 : Model1
    Ntf, Ntm = Array{Vector{tp}}(undef, maxgen),Array{Vector{tp}}(undef, maxgen)
    Ntf[1],Ntm[1]=copy(Ntf_init), copy(Ntm_init)
    NCountF=sum(Ntf_init)
    NCountM=sum(Ntm_init)

    for gen in 1:(maxgen - 1)
        # Reproduction change
        if ((NCountM>1) * (NCountF>1)) > 0
            beta = NCountM / (NCountM + beta)
            m = model(; Nf=Ntf[gen], Nm=Ntm[gen], E=e, alpha=alpha, ff=wmf[3], wm=wmf[1], wf=wmf[2], r_sex=r_sex,phi1=phi1,phi2=phi2, beta=beta, M2=M2, M=M,tp=tp,wa=wa)
            Ntf[gen+1] = m[1]
            Ntm[gen+1] = m[2]
            
        else
            Ntf[gen+1], Ntm[gen+1] = zeros(tp, M2), zeros(tp, M2)
        end
        NCountM = sum(Ntm[gen+1])
        NCountF = sum(Ntf[gen+1])
    end
    dc_ = [Ntf, Ntm]
    return dc_
end

#Tracking only cumulative population size by sex (outcomes)
function Model_nosp_NT(; N=10^15, pm=0.01, wmf,e, mD_sex, beta=0, alpha=0,phi1=1,phi2=1, maxgen=10, Model=2, M=4,tp=Int64,b2=0,wa=1)
    Ntf_init, Ntm_init = Input_nosp(; M=M, Nmean=N, pm=pm,tp=tp)
   
    M2 = Int(M * (M + 1) / 2)
    f=wmf[3][1,1]
    alpha = alpha == 0 ? Alpha(phi1=phi1,phi2=phi2, f=f,N=N) : alpha

    r_sex = 0.5 .* ones(M2, M2)
    if M == 4
        for i in [2, 5, 6, 7]
            r_sex[i, :] .= mD_sex
        end
    end
    if M == 5
        for i in [2, 6, 7, 8, 9]
            r_sex[i, :] .= mD_sex
        end
    end
    if b2>0     
        if M==4 r_sex[7,:] .=0.5 ;end
        if M==5 r_sex[9,:] .=0.5 ;end
    end
    model = Model == 2 ? Model2 : Model1

    NCountF=Array{tp}(undef, maxgen)
    NCountM=Array{tp}(undef, maxgen)

    
    Ntf, Ntm = copy(Ntf_init), copy(Ntm_init)
    NCountF[1]=sum(Ntf_init)
    NCountM[1]=sum(Ntm_init)

    for gen in 1:(maxgen - 1)
        # Reproduction change
       
        if (NCountM[gen]>1) * (NCountF[gen]>1)> 0
            beta = NCountM[gen] / (NCountM[gen] + beta)
            m = model(; Nf=Ntf, Nm=Ntm, E=e, alpha=alpha, ff=wmf[3], wm=wmf[1], wf=wmf[2], r_sex=r_sex,phi1=phi1,phi2=phi2, beta=beta, M2=M2, M=M,tp=tp,wa=wa)
            Ntf = m[1]
            Ntm = m[2]
            
        else
            Ntf, Ntm = zeros(tp, M2), zeros(tp, M2)
        end
        NCountM[gen+1] = sum(Ntm)
        NCountF[gen+1] = sum(Ntf)
    end
    dc_ = [NCountF, NCountM]
    return dc_
end



# Mulitplexing M1 (no deletions) panmictic model
function Model_nospM(; N=10^15, pm=0.01, wmf, e, mD_sex,wa=1, beta=0, alpha=0, phi1=1,phi2=1,maxgen=10, Model=2, M=4,tp=Int64,track=false)
    Ntf_init, Ntm_init = InputMulti(N; M=M,pm=pm,tp=tp)
   
    M=M-1
    M1= Int(M * (M + 1) ÷ 2)+1
    M2= Int(M1 * (M1 + 1) ÷ 2)
    
    f=wmf[3][1]

    alpha = alpha == 0 ? Alpha(;N=N,f=f, phi1=phi1,phi2=phi2) : alpha

    r_sex = 0.5 .* ones(M2, M2)

    D_Ind=[( M2-M1+1):M2;]

    for i in D_Ind r_sex[i,:].=mD_sex; end
    
    model=Model2

    NCountF=Array{tp}(undef, maxgen)
    NCountM=Array{tp}(undef, maxgen)

    
    Ntf, Ntm = copy(Ntf_init), copy(Ntm_init)
    
    NCountF[1]=sum(Ntf_init)
    NCountM[1]=sum(Ntm_init)

    for gen in 1:(maxgen - 1)
        # Reproduction change
       
        if (NCountM[gen]>1) * (NCountF[gen]>1)> 0
            beta = NCountM[gen] / (NCountM[gen] + beta)
            m = model(; Nf=Ntf, Nm=Ntm, E=e, alpha=alpha,ff=wmf[3], wm=wmf[1], wf=wmf[2], r_sex=r_sex, beta=beta, M2=M2, M=M1,tp=tp,phi1=phi1,phi2=phi2,wa=wa)
            Ntf = m[1]
            Ntm = m[2]
            
        else
            Ntf, Ntm = zeros(tp, M2), zeros(tp, M2)
        end
        if !track 
        NCountM[gen+1] = sum(Ntm)
        NCountF[gen+1] = sum(Ntf)
        end 
    end
    dc_ = [NCountF, NCountM]
    return dc_
end


# Spatial model: males and females migrating at rate mig
# mD_sex = sex ratio
function Model_sp(; Ntf_init, Ntm_init, Adj, Release_sites, wmf, E, mD_sex=0.5,wa=1,
    alpha=0,phi1=1,phi2=1, maxgen=100, Model=2, bound=1, 
    mu_mig=1, maxdist=1, M=4, beta=0, tp=Int64,
    track=false, stats_only=true)

    M2 = Int(div(M * (M + 1), 2))
    Nsites = size(Adj[1], 1)
    f = wmf[3][1, 1]

   if alpha == 0
        Npop=sum(Ntf_init.+Ntm_init)
        alpha = [Alpha(;N=Npop[s],f=f, phi1=phi1,phi2=phi2) for s in 1:Nsites]
    end

    model = Model == 2 ? Model2 : Model1

    # Precompute r_sex (for sex distortion strategies)
    if length(mD_sex)<2
        r_sex = fill(0.5, M2, M2)
        if M == 4
            for i in (2, 5, 6, 7)
                r_sex[i, :] .= mD_sex
            end
        elseif M == 5
            for i in (2, 6, 7, 8, 9)
                r_sex[i, :] .= mD_sex
            end
        end
    else r_sex=mD_sex
    end

    # Preallocate population matrices
    Ntf_current = copy(Ntf_init)
    Ntm_current = copy(Ntm_init)
    Ntf_next = zeros(tp, M2, Nsites)
    Ntm_next = zeros(tp, M2, Nsites)

    Am, w = Adj[1], Adj[2]
    
    nsf=[sum(Ntf_init[:,site]) for site in 1:Nsites]

    # Stat1 - female population relative to the initial size
    if stats_only
        Stat1 = zeros(Float64, maxgen, Nsites)
    else
        if !track
            Nt_mat = Array{tp}(undef, 2, maxgen, Nsites)
        else
            Nt_mat = Array{tp}(undef, 2, maxgen, Nsites, M2)
        end
    end

    for gen in 1:maxgen
        @inbounds for site in 1:Nsites
            sum_Ntm = sum(Ntm_current[:, site])
            sum_Ntf = sum(Ntf_current[:, site])

            if sum_Ntm > 0 && sum_Ntf > 0
                beta0 = sum_Ntm / (sum_Ntm + beta)
                m = model(; Nf=view(Ntf_current, :, site),
                           Nm=view(Ntm_current, :, site),
                           E=E, alpha=alpha[site], ff=wmf[3],
                           wm=wmf[1], wf=wmf[2], r_sex=r_sex,phi1=phi1,phi2=phi2,wa=wa,
                           beta=beta0, M2=M2, M=M, tp=tp)
                Ntf_next[:, site] .= m[1]
                Ntm_next[:, site] .= m[2]
            else
                Ntf_next[:, site] .= 0
                Ntm_next[:, site] .= 0
            end

            if stats_only
                Stat1[gen, site] =  sum_Ntf /nsf[site]
            else
                if !track
                    Nt_mat[1, gen, site] = sum(Ntf_next[:, site])
                    Nt_mat[2, gen, site] = sum(Ntm_next[:, site])
                else
                    Nt_mat[1, gen, site, :] = Ntf_next[:, site]
                    Nt_mat[2, gen, site, :] = Ntm_next[:, site]
                end
            end
        end

        # Migration phase 
        if gen < maxgen
            Ntf_next, Ntm_next = Migration_change(; Nf_s=Ntf_next,
                                                  Nm_s=Ntm_next,
                                                  A=Am, w=w,
                                                  Nsites=Nsites, M2=M2, tp=tp)
            Ntf_current .= Ntf_next
            Ntm_current .= Ntm_next
        end
    end

    return stats_only ? Stat1 : Nt_mat
end


# Migration by gravid females only
Mating_phase1 = function (; Nf, Nm,beta=0,ff, M2=10,tp=Int64,wa=1)
    N_pairs_A = zeros(tp, M2, M2)
    N_pairs_Af = zeros(tp, M2, M2)
    
    Nm_sum = sum(Nm.*wa)
    pAm = ifelse(Nm_sum > 0, Nm.*wa / Nm_sum, fill(1 / M2, M2))  # Avoid division by zero
    p_beta = Nm_sum / (beta + Nm_sum)

    N_exp_f = zeros(tp, M2)
    N_exp_m = zeros(tp, M2)

    multinomial_dist = [Multinomial(round(tp, Nf[i] * p_beta), pAm) for i in 1:M2]

    # Sample number of mating pairs for each genotype combination
    for i in 1:M2
        rand!(multinomial_dist[i], view(N_pairs_A, i, :))
    end

    poisson_samples = Vector{Poisson}(undef, M2)
    for j in 1:M2
        for i in 1:M2
            poisson_samples[i] = Poisson(N_pairs_A[i, j] * ff[i,j])
            N_pairs_Af[i, j] = rand(poisson_samples[i])  # Sample once, reusing distribution
        end
    end
    return N_pairs_Af
end
Mating_phase2=function(;N_pairs_Af,E, M2,r_sex,tp=Int64)
    N_exp_f = zeros(tp, M2)
    N_exp_m = zeros(tp, M2)
    for j in 1:M2
        for i in 1:M2
            if N_pairs_Af[i, j] > 0
                N_exp = rand(Multinomial(N_pairs_Af[i, j], E[i][j]))  # Offspring count
                N_sx = rand.(Binomial.(N_exp, 1 .- r_sex[j, i]))  # Sex allocation

                @inbounds N_exp_f .+= N_sx
                @inbounds N_exp_m .+= (N_exp .- N_sx)
            end
        end
    end

    return N_exp_f, N_exp_m
end


function Migration_inter(; Nf_pairs, A, w, Nsites, mu_mig=1, M2=10,tp=Int64)
    Nf_new = zeros(tp, M2, M2, Nsites)

    pairs = collect(Iterators.product(1:M2, 1:M2))

    @threads for idx in eachindex(pairs)
        (i, j) = pairs[idx]

        @inbounds for site in 1:Nsites
            nz_inds, nz_vals = findnz(A[:, site])
            if isempty(nz_inds)
                continue
            end

            probvec = nz_vals ./ sum(nz_vals)
            count = round(tp, mu_mig * Nf_pairs[i, j, site] * w[site])

            if count > 0
                sample = rand(Multinomial(count, probvec))
                @inbounds @simd for k in eachindex(nz_inds)
                    dst = nz_inds[k]
                    Nf_new[i, j, dst] += sample[k]  
                end
            end
        end
    end

    return Nf_new
end


# Spatial model for female-only migration
function Model_sp_inter(; Ntf_init, Ntm_init, Adj, Release_sites, wmf, E,
    mD_sex=0.5, alpha=0,phi1=1,phi2=1, maxgen=100, bound=1,Model=2,
    mu_mig=1, maxdist=1, M=4, M2=0, beta=0,fs=[],wa=1,
    tp=Int64, track=false, stats_only=true)

    if M2<1 M2 = div(M * (M + 1), 2); end
    Nsites = size(Adj[1], 1)

    f = wmf[3][1, 1]
    if alpha == 0
        Npop=sum(Ntf_init.+Ntm_init)
        alpha = [Alpha(;N=Npop[s],f=f, phi1=phi1,phi2=phi2) for s in 1:Nsites]
    end

    if length(mD_sex)<2
        r_sex = fill(0.5, M2, M2)
        if M == 4
            for i in (2, 5, 6, 7)
                r_sex[i, :] .= mD_sex
            end
        elseif M == 5
            for i in (2, 6, 7, 8, 9)
                r_sex[i, :] .= mD_sex
            end
        end
    else r_sex=mD_sex
    end
    nsf=[sum(Ntf_init[:,site]) for site in 1:Nsites]

    Ntf_current = copy(Ntf_init)
    Ntm_current = copy(Ntm_init)
    Ntf_next = zeros(tp, M2, Nsites)
    Ntm_next = zeros(tp, M2, Nsites)

    Nf_pairs1Temp = zeros(tp, M2, M2, Nsites)
    Nf_pairs2Temp = zeros(tp, M2, M2, Nsites)

    Am = Adj[1]; w = Adj[2]

    if stats_only
        Stat1 = zeros(Float64, maxgen, Nsites)
    else
        if !track
            Nt_mat = Array{tp}(undef, 2, maxgen, Nsites)
        else
            Nt_mat = Array{tp}(undef, 2, maxgen, Nsites, M2)
        end
    end

    if Model==2
        for gen in 1:maxgen
            @inbounds for site in 1:Nsites
                sum_Ntm = sum(Ntm_current[:, site])
                sum_Ntf = sum(Ntf_current[:, site])

                if sum_Ntm > 0 && sum_Ntf > 0
                    beta0 = sum_Ntm / (sum_Ntm + beta)
                    Nf_pairs1Temp[:, :, site] =Mating_phase1(; Nf=view(Ntf_current, :, site),
                                    Nm=view(Ntm_current, :, site),beta=beta0, ff=wmf[3], M2=M2, tp=tp)
                else
                    Nf_pairs1Temp[:, :, site] .= 0
                end
            end

            Nf_pairs2Temp = Migration_inter(; Nf_pairs=Nf_pairs1Temp,A=Am, w=w, Nsites=Nsites,mu_mig=1, M2=M2, tp=tp)

            @inbounds for site in 1:Nsites
                if sum(Nf_pairs2Temp[:, :, site]) > 0
                    Nf, Nm = Mating_phase2(; N_pairs_Af=Nf_pairs2Temp[:, :, site],E=E, M2=M2, r_sex=r_sex, tp=tp)
                    Nh = sum(Nf .+ Nm)
                    Nf_1, Nm_1 = Ddm_change(; Nf=Nf, Nm=Nm, Nh=Nh,alpha=alpha[site], M2=M2,phi1=phi1,phi2=phi2)[1:2]
                    Nf_2, Nm_2 = Fitness_change(; Nf=Nf_1, Nm=Nm_1,wf=wmf[2], wm=wmf[1], M2=M2)

                    Ntf_next[:, site] .= Nf_2
                    Ntm_next[:, site] .= Nm_2
                else
                    Ntf_next[:, site] .= 0
                    Ntm_next[:, site] .= 0
                end

                if stats_only

                    Stat1[gen, site] = sum(Ntf_next[:, site])/nsf[site]

                else
                    if !track
                        Nt_mat[1, gen, site] = sum(Ntf_next[:, site])
                        Nt_mat[2, gen, site] = sum(Ntm_next[:, site])
                    else
                        Nt_mat[1, gen, site, :] = Ntf_next[:, site]
                        Nt_mat[2, gen, site, :] = Ntm_next[:, site]
                    end
                end
            end

            if gen < maxgen
                copy!(Ntf_current, Ntf_next)
                copy!(Ntm_current, Ntm_next)
            end
        end
    else 
        for gen in 1:maxgen
        @inbounds for site in 1:Nsites
            sum_Ntm = sum(Ntm_current[:, site])
            sum_Ntf = sum(Ntf_current[:, site])

            if sum_Ntm > 0 && sum_Ntf > 0
                beta0 = sum_Ntm / (sum_Ntm + beta)
                Nf_pairs1Temp[:, :, site] = Mating_phase1(; Nf=view(Ntf_current, :, site),Nm=view(Ntm_current, :, site),
                                   beta=beta0, ff=wmf[3], M2=M2, tp=tp,wa=wa)
            else
                Nf_pairs1Temp[:, :, site] .= 0
            end
        end

        Nf_pairs2Temp = Migration_inter(; Nf_pairs=Nf_pairs1Temp,A=Am, w=w, Nsites=Nsites,mu_mig=1, M2=M2, tp=tp)

        @inbounds for site in 1:Nsites
            if sum(Nf_pairs2Temp[:, :, site]) > 0
                Nf, Nm = Mating_phase2(; N_pairs_Af=Nf_pairs2Temp[:, :, site],E=E, M2=M2, r_sex=r_sex, tp=tp)
                Nf_1, Nm_1 = Fitness_change(; Nf=Nf, Nm=Nm,wf=wmf[2], wm=wmf[1], M2=M2)
                Nh = sum(Nf_1 .+ Nm_1)
                Nf_2, Nm_2 = Ddm_change(; Nf=Nf, Nm=Nm, Nh=Nh,alpha=alpha[site], M2=M2,phi1=phi1,phi2=phi2)[1:2]

                Ntf_next[:, site] .= Nf_2
                Ntm_next[:, site] .= Nm_2
            else
                Ntf_next[:, site] .= 0
                Ntm_next[:, site] .= 0
            end

            if stats_only

                Stat1[gen, site] = sum(Ntf_next[:, site])/nsf[site]
            else
                if !track
                    Nt_mat[1, gen, site] = sum(Ntf_next[:, site])
                    Nt_mat[2, gen, site] = sum(Ntm_next[:, site])
                else
                    Nt_mat[1, gen, site, :] = Ntf_next[:, site]
                    Nt_mat[2, gen, site, :] = Ntm_next[:, site]
                end
            end
        end

        if gen < maxgen
            copy!(Ntf_current, Ntf_next)
            copy!(Ntm_current, Ntm_next)
        end
    end
end
    return stats_only ? Stat1 : Nt_mat
end


function Model_spO(; Ntf_init, Ntm_init, Adj, Release_sites, wmf, E, mD_sex=0.5,wa=1,
    alpha=0,phi1=1,phi2=1, maxgen=100, Model=2, bound=1, R1Ind,sex=:b,freq=false,
    mu_mig=1, maxdist=1, M=4, beta=0, tp=Int64,
    track=false, stats_only=true)

    M2 = Int(div(M * (M + 1), 2))

    Nsites = size(Adj[1], 1)

    f = wmf[3][1, 1]

   if alpha == 0
        Npop=sum(Ntf_init.+Ntm_init)
        alpha = [Alpha(;N=Npop[s],f=f, phi1=phi1,phi2=phi2) for s in 1:Nsites]
    end

    model = Model == 2 ? Model2 : Model1

    if sex==:b k1=1; k2=2;
    elseif sex==:f k1=1; k2=0
    else k1=0;k2=1
    end
    # Precompute r_sex
    if length(mD_sex)<2
        r_sex = fill(0.5, M2, M2)
        if M == 4
            for i in (2, 5, 6, 7)
                r_sex[i, :] .= mD_sex
            end
        elseif M == 5
            for i in (2, 6, 7, 8, 9)
                r_sex[i, :] .= mD_sex
            end
        end
    else r_sex=mD_sex
    end

    # Preallocate population matrices 
    Ntf_current = copy(Ntf_init)
    Ntm_current = copy(Ntm_init)
    Ntf_next = zeros(tp, M2, Nsites)
    Ntm_next = zeros(tp, M2, Nsites)

    Am = Adj[1]; w = Adj[2]
    
    nsf=[sum(Ntf_init[:,site]) for site in 1:Nsites]

    # Allocate outputs

    Stat1 = zeros(Float64, maxgen, Nsites)

    for gen in 1:maxgen
        @inbounds for site in 1:Nsites
            sum_Ntm = sum(Ntm_current[:, site])
            sum_Ntf = sum(Ntf_current[:, site])

            if sum_Ntm > 0 && sum_Ntf > 0
                beta0 = sum_Ntm / (sum_Ntm + beta)
                m = model(; Nf=view(Ntf_current, :, site),
                           Nm=view(Ntm_current, :, site),
                           E=E, alpha=alpha[site], ff=wmf[3],
                           wm=wmf[1], wf=wmf[2], r_sex=r_sex,phi1=phi1,phi2=phi2,wa=wa,
                           beta=beta0, M2=M2, M=M, tp=tp)
                Ntf_next[:, site] .= m[1]
                Ntm_next[:, site] .= m[2]
            else
                Ntf_next[:, site] .= 0
                Ntm_next[:, site] .= 0
            end

            if !freq Stat1[gen, site] =  (sum(Ntf_next[R1Ind, site])*k1 + sum(Ntm_next[R1Ind, site])*k2) /(nsf[site]*(k1+k2))
            else Stat1[gen, site] =  (sum(Ntf_next[R1Ind, site])*k1 + sum(Ntm_next[R1Ind, site])*k2) /(sum(Ntf_next[:, site])*k1 + sum(Ntm_next[:, site])*k2)
            end
        end

        # Migration phase 
        if gen < maxgen
            Ntf_next, Ntm_next = Migration_change(; Nf_s=Ntf_next,
                                                  Nm_s=Ntm_next,
                                                  A=Am, w=w,
                                                  Nsites=Nsites, M2=M2, tp=tp)
            Ntf_current .= Ntf_next
            Ntm_current .= Ntm_next
        end
    end

    return  Stat1
end



#####
function Model_spO_cutM1(; Ntf_init, Ntm_init, Adj, Release_sites, wmf, E, mD_sex=0.5,wa=1,
    alpha=0,phi1=1,phi2=1, maxgen=100, Model=2, bound=1, R1Ind,sex=:b,
    mu_mig=1, maxdist=1, M=4,M2, beta=0, tp=Int64,
    track=false, stats_only=true, to_first=true, threshold=1,p=0.33)

    M=M-1             
    M1= Int(M * (M + 1) ÷ 2)+1
    M2= Int(M1 * (M1 + 1) ÷ 2)

    Nsites = size(Adj[1], 1)

    f = wmf[3][1, 1]

   if alpha == 0
        Npop=sum(Ntf_init.+Ntm_init)
        alpha = [Alpha(;N=Npop[s],f=f, phi1=phi1,phi2=phi2) for s in 1:Nsites]
    end
    model = Model == 2 ? Model2 : Model1

    if sex==:b k1=1; k2=2;
    elseif sex==:f k1=1; k2=0
    else k1=0;k2=1
    end
    # Precompute r_sex
    r_sex = fill(0.5, M2, M2)
    for i in (M2-M1+1):M2
        r_sex[i, :] .= mD_sex
    end

    Ntf_current = copy(Ntf_init)
    Ntm_current = copy(Ntm_init)
    Ntf_next = zeros(tp, M2, Nsites)
    Ntm_next = zeros(tp, M2, Nsites)

    Am = Adj[1]; w = Adj[2]
    
    nsf=[sum(Ntf_init[:,site]) for site in 1:Nsites]

    Stat1 = zeros(Float64, maxgen, Nsites)
    Stat2 = zeros(Float64, maxgen, Nsites)

    found_gen = nothing
    found_site = nothing

    found_gen2 = zeros(Nsites)
    found_site2= zeros(Nsites)

    denom = nsf .* (k1 + k2)

    for gen in 1:maxgen
        @inbounds for site in 1:Nsites

            Ntf_site = view(Ntf_current, :, site)
            Ntm_site = view(Ntm_current, :, site)

            sum_Ntm = sum(Ntm_site)
            sum_Ntf = sum(Ntf_site)

            if sum_Ntm > 0 && sum_Ntf > 0
                beta0 = sum_Ntm / (sum_Ntm + beta)

                m = model(; Nf=Ntf_site,
                        Nm=Ntm_site,
                        E=E, alpha=alpha[site], ff=wmf[3],
                        wm=wmf[1], wf=wmf[2],
                        r_sex=r_sex,phi1=phi1,phi2=phi2,wa=wa,
                        beta=beta0, M2=M2, M=M, tp=tp)

                Ntf_next[:, site] .= m[1]
                Ntm_next[:, site] .= m[2]
            else
                Ntf_next[:, site] .= 0
                Ntm_next[:, site] .= 0
            end
    
            Stat2_val = sum(Ntf_next[:, site]) / nsf[site]
            if Stat2_val <= p
                found_gen2[site] = gen
                found_site2[site] = nsf[site]
            end
            stat_val =
                (sum(view(Ntf_next, R1Ind, site))*k1 +
                sum(view(Ntm_next, R1Ind, site))*k2) / denom[site]

            Stat1[gen, site] = stat_val
            # Stop early if threshold reached
            if to_first && stat_val >= threshold#[site]
                found_gen = gen
                found_site = site
                return found_gen, found_site,found_gen2, found_site2
            end

        end

        # Migration
        if gen < maxgen
            Ntf_next, Ntm_next = Migration_change(; Nf_s=Ntf_next,
                                                Nm_s=Ntm_next,
                                                A=Am, w=w,
                                                Nsites=Nsites, M2=M2, tp=tp)

            Ntf_current .= Ntf_next
            Ntm_current .= Ntm_next
        end
        end

    return  found_gen, found_site,found_gen2, found_site2#[found_site2.>0]
end




# Origin of resistance model with time record at the moment of 
function Model_spO_cut_flagged(; Ntf_init, Ntm_init, Adj, Release_sites, wmf, E, mD_sex=0.5, wa=1,
    alpha=0, phi1=1, phi2=1, maxgen=100, Model=2, bound=1, R1Ind, sex=:b,
    mu_mig=1, maxdist=1, M=4, beta=0, tp=Int64,
    track=false, stats_only=true, to_first=true, threshold=1, p=0.33)

    M2 = M * (M + 1) ÷ 2
    Nsites = size(Adj[1], 1)
    f = wmf[3][1, 1]

    if alpha == 0
        Npop = sum(Ntf_init .+ Ntm_init)
        alpha = [Alpha(; N=Npop[s], f=f, phi1=phi1, phi2=phi2) for s in 1:Nsites]
    end

    model = Model == 2 ? Model2 : Model1

    # Sex selection (:b = both, :f = females only, :m = males only)
    k1, k2 = if sex == :b
        1, 2
    elseif sex == :f
        1, 0
    else
        0, 1
    end

    # Pre-compute r_sex
    if length(mD_sex) < 2
        r_sex = fill(0.5, M2, M2)
        if M == 4
            r_sex[[2, 5, 6, 7], :] .= mD_sex
        elseif M == 5
            r_sex[[2, 6, 7, 8, 9], :] .= mD_sex
        end
    else
        r_sex = mD_sex
    end

    Ntf_current = copy(Ntf_init)
    Ntm_current = copy(Ntm_init)
    Ntf_next = zeros(tp, M2, Nsites)
    Ntm_next = zeros(tp, M2, Nsites)

    Am, w = Adj[1], Adj[2]
    nsf = [sum(Ntf_init[:, site]) for site in 1:Nsites]

    Stat1 = zeros(Float64, maxgen, Nsites)
    Stat2 = zeros(Float64, maxgen, Nsites)

    found_gen = nothing
    found_site = nothing
    found_gen2 = zeros(Nsites)
    found_site2 = zeros(Nsites)

    denom = nsf .* (k1 + k2)

    # Flagging system
    flagged_sites = Set{Int}()
    flag_gen = Dict{Int, Int}()
    for gen in 1:maxgen
        @inbounds for site in 1:Nsites
            Ntf_site = view(Ntf_current, :, site)
            Ntm_site = view(Ntm_current, :, site)

            sum_Ntm = sum(Ntm_site)
            sum_Ntf = sum(Ntf_site)

            if sum_Ntm > 0 && sum_Ntf > 0
                beta0 = sum_Ntm / (sum_Ntm + beta)

                m = model(; Nf=Ntf_site, Nm=Ntm_site, E=E, alpha=alpha[site], 
                         ff=wmf[3], wm=wmf[1], wf=wmf[2], r_sex=r_sex, 
                         phi1=phi1, phi2=phi2, wa=wa, beta=beta0, M2=M2, M=M, tp=tp)

                Ntf_next[:, site] = m[1]
                Ntm_next[:, site] = m[2]
            else
                Ntf_next[:, site] .= 0
                Ntm_next[:, site] .= 0
            end

            # Stat2  - relative female population below threshold p record
            Stat2_val = sum(Ntf_next[:, site]) / nsf[site]
            if Stat2_val <= p
                found_gen2[site] = gen
                found_site2[site] = nsf[site]
            end

            # Stat1 - filtered population size relative to the initial conditions
            stat_val = (sum(view(Ntf_next, R1Ind, site)) * k1 + 
                       sum(view(Ntm_next, R1Ind, site)) * k2) / denom[site]
            
            Stat1[gen, site] = stat_val
            # 1) Flag site if Stat1 > 0 (first time or re-flagging after going to 0)

            if stat_val > 0
                if !(site in flagged_sites)
                    push!(flagged_sites, site)  
                    flag_gen[site] = gen  # Record when it was flagged
                end
                
                # 2) Check if this flagged site now exceeds threshold
                if to_first && stat_val >= threshold
                    found_gen = flag_gen[site]  # Return when it was FLAGGED
                    found_site = site
                    return found_gen, found_site, found_gen2, found_site2
                end
            else
                # 3) Remove from flagged sites if Stat1 <= 0
                if site in flagged_sites
                    delete!(flagged_sites, site)  
                    delete!(flag_gen, site)  # Remove its flag generation record
                end
            end
        end

        # Migration
        if gen < maxgen
            Ntf_next, Ntm_next = Migration_change(; Nf_s=Ntf_next, Nm_s=Ntm_next,
                                                  A=Am, w=w, Nsites=Nsites, M2=M2, tp=tp)

            Ntf_current .= Ntf_next
            Ntm_current .= Ntm_next
        end
    end

    return found_gen, found_site, found_gen2, found_site2
end




#### Supplementary functions

# Genotypes from gametes pairs formation
Genot=function(;A,M2,M)
    v = Array{Float64}(undef,M2)
    k=0
    for i in 1:M
        v[k+=1] =A[i,i]
        for j in (i+1):M
            v[k+=1]=A[i,j]+A[j,i]
        end
    end
    return v
end

# Gametes produced probability (genotype)
E_Make_1=function(;c=0.95,j=0.03,a=0.01,b=0.001,show=false,M=4,title="Gametes produces")
    if M==5
        E5=vcat(
            [1 0 0 0 0],
            [(1-c)/2 (1/2+c*(1-j)*(1-a)/2) c*j*(1-b)/2 c*j*b/2 c*(1-j)*a/2],
            [1/2 0 1/2 0 0],
            [1/2 0 0 1/2 0],
            [(1-c)/2 0 c*j*(1-b)/2 c*j*b/2 (1/2+c*(1-j)/2)],
            [0 1 0 0 0],
            [0 1/2 1/2 0 0],
            [0 1/2 0 1/2 0],
            [0 1/2 0 0 1/2],
            [0 0 1 0 0],
            [0 0 1/2 1/2 0],
            [0 0 1/2 0 1/2],
            [0 0 0 1 0],
            [0 0 0 1/2 1/2],
            [0 0 0 0 1]
            )
        if show 
            namesG=["WW","WD","WN","WR","WO","DD","DN","DR","DO","NN","NR","NO", "RR","RO","OO"]
            namesA=["W","D","N","R","O"]
            hl_odd = Highlighter( f   = (data,i,j) -> i in [2,6,7,8,9],
                             crayon = Crayon(background = :light_blue))
            pretty_table(E5,column_labels=namesA,table_format = TextTableFormat(borders = text_table_borders__borderless),row_labels=namesG,highlighters=hl_odd,title=title)
        end
    end
    if M==4
        E5=vcat(
            [1 0 0 0 ],
            [(1-c)/2 (1/2+c*(1-j)/2) c*j*(1-b)/2 c*j*b/2 ],
            [1/2 0 1/2 0 ],
            [1/2 0 0 1/2 ],
            [0 1 0 0 ],
            [0 1/2 1/2 0 ],
            [0 1/2 0 1/2 ],
            [0 0 1 0 ],
            [0 0 1/2 1/2 ],
            [0 0 0 1 ]
            )
        if show 
            namesG=["WW","WD","WN","WR","DD","DN","DR","NN","NR", "RR"]
            namesA=["W","D","N","R"]
            hl_odd = Highlighter( f   = (data,i,j) -> i in [2,5,6,7],
                                  crayon = Crayon(background = :light_blue))
            pretty_table(E5,column_labels=namesA,table_format = TextTableFormat(borders = text_table_borders__borderless),row_labels=namesG,highlighters=hl_odd,title=title)

        end
    end    
    return transpose(E5)
end
# Gametes produced probability (sex-specific homing)
E_Make = function (; M=4, homing=:b, c_m=0.8, j_m=0.2, a_m=0.1, b_m=0.1, c_f=0.8, j_f=0.2, a_f=0.1, b_f=0.1, show=false)
    M2 =Int(M*(M+1)/2)
    if homing == :b
        if show
            println("Homing in both sexes")
        end
        Em = E_Make_1(; c=c_m, j=j_m, a=a_m, b=b_m, show=show, M=M, title="Gametes produced (males)")
        Ef = E_Make_1(; c=c_f, j=j_f, a=a_f, b=b_f, show=show, M=M, title="Gametes produced (females)")
    elseif homing == :m
        if show
            println("Homing in males")
        end
        Em = E_Make_1(; c=c_m, j=j_m, a=a_m, b=b_m, show=show, M=M, title="Gametes produced (males)")
        Ef = E_Make_1(; c=0, j=0, a=0, b=0, show=false, M=M, title="Gametes produced (females)")
    elseif homing == :f
        if show
            println("Homing in females")
        end
        Em = E_Make_1(; c=0, j=0, a=0, b=0, show=false, M=M, title="Gametes produced (males)")
        Ef = E_Make_1(; c=c_f, j=j_f, a=a_f, b=b_f, show=show, M=M, title="Gametes produced (females)")
    end
    e = E_mat(; Ef=Ef, Em=Em, M2=M2, M=M)
    return e
end
# f x m gametes pairs produced by each genotype pair
E_mat=function(;Ef,Em,M2,M)
    E_temp=[[Ef[:,i]*transpose(Em[:,j]) for i in 1:M2] for j in 1:M2]
    e=[[Genot(;A=E_temp[i][j],M=M,M2=M2) for i in 1:M2] for j in 1:M2]
    return e
end

Fitness_input1=function(;sigma=0.02,s=1,h=1,hN=0.03,hDR=0.03,show=true,M=4,hm=0,f=12,inter=true,wf_nb=[],non_func=["D","N","O"])
    M2=Int(M*(M+1)/2)
    wm=ones(M2)
    if M==5
        wf0=[1,
        1-h*s,
        1-hN*s,
        1-sigma,
        1-h*s,
        1-s,
        1-s,
        (1-sigma)*(1-hDR*s),
        1-s,
        1-s,
        (1-sigma)*(1-hN*s),
        1-s,
        (1-sigma)^2,
        (1-sigma)*(1-hDR*s),
        1-s
        ]
        wfth=["1",
        "1-hs",
        "1-hN*s",
        "1-sigma",
        "1-h*s",
        "1-s",
        "1-s",
        "(1- sigma)(1-hDRs)",
        "1-s",
        "1-s",
        "(1-sigma)(1-hN*s)",
        "1-s",
        "(1-sigma)^2",
        "(1-sigma)(1-hDR*s)",
        "1-s"
        ]
        namesG=["WW","WD","WN","WR","WO","DD","DN","DR","DO","NN","NR","NO", "RR","RO","OO"]
        
 
        #hl_odd = Highlighter( f   = (data,i,j) -> i in [2,6,7,8,9], crayon = Crayon(background = :light_blue))
    end
    if M==4
        wf0=[1,
        1-h*s,
        1-hN*s,
        1-sigma,
        1-s,
        1-s,
        (1-sigma)*(1-hDR*s),
        1-s,
        (1-sigma)*(1-hN*s),
        (1-sigma)^2,
        ]
        wfth=["1",
        "1-h*s",
        "1-h_N*s",
        "1-sigma",
        "1-s",
        "1-s",
        "(1-sigma)(1-hDR*s)",
        "1-s",
        "(1-sigma)(1-hN*sN)",
        "(1-sigma)^2",
        ]
        namesG=["WW","WD","WN","WR","DD","DN","DR","NN","NR","RR"]
        
        #hl_odd = Highlighter( f   = (data,i,j) -> i in [2,6,7,8,9], crayon = Crayon(background = :light_blue))
        
    end
    namesF=["Males","Females","Formula (female)"]
    
    if inter
        wf=ones(M2)
        wf[wf0.<=0] .=0
    else wf=wf_nb
    end
    di=occursin.("D",namesG);wm[di].=1-hm
    
    ff=ones(M2,M2).*f
    #wftype=repeat(["Full fitness "],M2);wftype[wf0.<=0].="Full sterility";wftype[1]="Wild-type";wftype[(wf0.>0).&(wf0.<1)].="Full fitness"
    for j in 1:M2 for i in 1:M2 ff[i,j]=wf0[i]*wm[j]*f;end;end
    if show 
        pretty_table([wm wf0 wfth ],column_labels=namesF,table_format = TextTableFormat(borders = text_table_borders__borderless),row_labels=namesG,alignment=:l)
    end
    return wm,wf,ff
end

#Initializing population size 
Initial = function (; NN,Nrelease,M2=10,ReleaseGT=[2],tp=Int64)
    Nf0=zeros(tp,M2)
    Nm0=zeros(tp,M2)
    Nf0[1] = round(tp, NN *  0.5)#, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    Nm0[1] = round(tp, NN *  0.5)#, 0, 0, 0, Nrelease, 0, 0, 0, 0, 0]
    Nm0[ReleaseGT] .=  Nrelease
    return Nf0, Nm0
end
# Initital population size for the spatial model
Initial_vec = function (; N, Nsites, Release_sites, pm=0.1, Rerelease=false, M=4,tp=Int64)
    M2=Int(M*(M+1)/2)

    if length(N)<Nsites
        NN=repeat([round(Int,N/Nsites)],Nsites)
    else 
        NN=N
    end
    Rmv1 = zeros(Nsites)
    Nrelease=round.(tp,NN.*pm.*0.5) #/length(Release_sites))  #Releases pm%, divided amongst release sites
    Rmv1[Release_sites] = Nrelease[Release_sites]
    Nmv0 = [Initial(;NN=NN[i],Nrelease=Rmv1[i],M2=M2,ReleaseGT=M+1,tp=tp)[2] for i in 1:Nsites] 

    Nfv0=[Initial(;NN=NN[i],Nrelease=0,M2=M2,ReleaseGT=M+1,tp=tp)[1] for i in 1:Nsites] 
    Nmv=reshape(reduce(vcat,Nmv0),M2,Nsites)
    Nfv=reshape(reduce(vcat,Nfv0),M2,Nsites)
    return Nfv, Nmv
end
#Initial vestor for the fixed number of mosquitoes released (divided by the number of release sites)
Initial_vec_fixed = function (; N, Nsites, Release_sites, pm=0.1,NumRelease=0, ReleaseGT=[2],M2=10,tp=Int64)
    if length(N)<Nsites
        NN=repeat([round(tp,N/Nsites)],Nsites)
    else 
        NN=N
    end
    Rmv1 = zeros(Nsites)
    if NumRelease<=0 Nrelease=round.(tp,NN.*pm.*0.5) #/length(Release_sites))  #Releases pm%, divided amongst release sites
    else Nrelease=[round(tp,NumRelease/length(Release_sites)) for n in 1:Nsites]
    end
    Rmv1[Release_sites] = Nrelease[Release_sites]
    Nmv0 = [Initial(;NN=NN[i],Nrelease=Rmv1[i],M2=M2,ReleaseGT=ReleaseGT,tp=tp)[2] for i in 1:Nsites] 
    Nfv0=[Initial(;NN=NN[i],Nrelease=0,M2=M2,ReleaseGT=ReleaseGT,tp=tp)[1] for i in 1:Nsites] 
    Nmv=reshape(reduce(vcat,Nmv0),M2,Nsites)
    Nfv=reshape(reduce(vcat,Nfv0),M2,Nsites)
    return Nfv, Nmv
end

# Spatial


function NSites(N)
    y=3*N^2-3*N+1
    return y
end

function NSites_reverse(n)
    d= (sqrt(9+12*(n-1))+3)/6
    return round(Int,d)
end


Test_input = function (; num=6, Nreleases=1, Npop=10^8, p_in=0.9, maxdist=1, rebound=true, rand_release=false,tp=Int64)
    num=num-1
    h_list, v, Nsites, d = Hexagon1(; n=num)
    A = Hexagon_to_adj_new(; Nsites=Nsites, d=d, p_in=p_in, dia=num, maxdist=maxdist)
    if rebound
        A = [A[1], repeat([1.0], Nsites)]
    end
    #for i in 1:Nsites
    #    A[1][i,:]=A[1][i,:]./sum(A[1][i,:])
    #end
    return A
end


SI_inner = function (; Nsites, dia, d, n=0)
    dv = [Matrix(d)[i, :] for i in 1:Nsites]
    h = Hexagon1(; n=n)
    hd = [Matrix(h[4])[i, :] for i in 1:h[3]]
    lvec = [[dv[j] in (hd .+ [dv[k]]) for j in 1:Nsites] for k in 1:Nsites]
    return lvec
end

# Bandwidth for distance calculation (h=width, delta = shift at the centre deme)
Hex_band=function(;dia=25, h=2,delta=1)
    k2=vcat([0],([delta:h:(dia)*2+delta;]).^2)
    K=length(k2)-1
    hh=Hexagon11(;n=dia-1)[4]
    Nsites=size(hh)[1]
    bands=[[i for i in 1:Nsites if k2[ii]<=(hh[i,1])^2+(hh[i,2])^2<k2[ii+1]] for ii in 1:K]
    #    Nr=[ss[pp][length.(ss[pp]).>0]   for pp in eachindex(ss)];
    Nr=bands[length.(bands).>0] 
    return Nr
end

# Find distance for a given site with radius dia of the area
Site_loc=function(dat;dia=25,H=2)
    Nr_=Hex_band(;dia=dia,h=H) ;
    loc=[[r for r in eachindex(Nr_) if (dat[j] in Nr_[r])][1] for j in eachindex(dat)]
    return loc
end
Second(x) = [x[j][2] for j in eachindex(x)]##

#Adjacency matrix with rebound 
function Hexagon_to_adj_new(; maxdist=1, Nsites, dia, d, p_in=0.99)
    n = maxdist + 1

    # Precompute all SI_inner levels
    si = [SI_inner(; Nsites=Nsites, dia=dia, d=d, n=nn) for nn in 0:n]

    # Compute decay difference
    si_diff = si[n+1] .* n .- sum(@view si[1:n])
    
    # Compute di as a matrix directly
    di = zeros(Nsites, Nsites)
    for i in 1:Nsites
        @inbounds for j in 1:Nsites
            if si[n][i][j] > 0
                di[i, j] = exp(-si_diff[i][j])
            end
        end
    end

    # Normalize p_out
    w = sum(@view di[dia+1, :])
    p_out = (1 - p_in) / (w - 1)

    # Construct sparse matrix A
    rows = Int[]
    cols = Int[]
    vals = Float64[]

    for i in 1:Nsites
        total = 0.0
        for j in 1:Nsites
            if i != j && di[i, j] > 0
                val = p_out * di[i, j]
                push!(rows, i); push!(cols, j); push!(vals, val)
                total += val
            end
        end
        # Diagonal element
        push!(rows, i); push!(cols, i); push!(vals, 1.0 - total)
    end

    A = sparse(rows, cols, vals, Nsites, Nsites)
    #for i in 1:Nsites
    #    A[i,:].=A[i,:]./sum(A[i,:])
    #end
    weights = p_in ./ diag(A)
    return A, weights
end


# M1 multiplexing functions:
Transition1=function(;c=0.95,j=0.03,b=0.001, show=false, M=3)
    M0=Int(M*(M+1)/2) # non-drive haplotypes
    M1=M0+1 #all haplotypes

    M2=Int(M1*(M1+1)/2) # all pairs of haplotypes
    MM_Ind=collect(zip(reduce(vcat,[1:i for i in 1:M0]),reduce(vcat,[repeat([i],i) for i in 1:M0]))) # pair of haplotypes to gametes haplotype

    Tr=Array{Array{Float64}}(undef, M2)
    for ii in eachindex(MM_Ind)
        Tr[ii]=zeros(M1)
        Tr[ii][MM_Ind[ii][1]]+=1/2
        Tr[ii][MM_Ind[ii][2]]+=1/2
    end
    Genot=function(A;M,M2)
        v = Array{Float64}(undef,M2)
        k=0
        for i in 1:M
            v[k+=1] =A[i,i]
            for j in (i+1):M
                v[k+=1]=A[i,j]+A[j,i]
            end
        end
        return v
    end

    
    M_Ind = [(i,j) for i in 1:M for j in i:M]
    pnd=ones(M1)./2;pnd[1:M]=pnd[1:M]*(1-c*(1-j));pnd[1]=pnd[1]*(1-c*(1-j))    # prob of no drive
    for ll in 1:M0
        v=[[(1-c)/2 , c*j*(1-b)/2, c*j*b/2]./sum([(1-c)/2 , c*j*(1-b)/2, c*j*b/2]),[0,1,0],[0,0,1]] #gametes formation with D (reweighted)
        vv=deepcopy(Genot(v[first(M_Ind[ll])]*transpose(v[last(M_Ind[ll])]),M=M,M2=M0))
        Tr[length(MM_Ind)+ll]=deepcopy(vcat(pnd[ll]*vv, 1-pnd[ll]))
    end
    Tr[M2] = zeros(M1);Tr[M2][M1]=1
    return Tr 
end

# M1 gametes production
E_Make_Multi = function (; M=3, homing=:b, c_m=0.95, j_m=0.03, b_m=0.001, c_f=0.95, j_f=0.03,  b_f=0.001, show=false)
    M1 =Int(M*(M+1)/2)+1
    M2=Int(M1*(M1+1)/2)
    MM_Ind=collect(zip(reduce(vcat,[1:i for i in 1:M1]),reduce(vcat,[repeat([i],i) for i in 1:M1])))

    if homing == :b
        if show
            println("Homing in both sexes")
        end
        Em = Transition1(; c=c_m, j=j_m, b=b_m, show=show, M=M)
        Ef = Transition1(; c=c_f, j=j_f, b=b_f, show=show, M=M)
    elseif homing == :m
        if show
            println("Homing in males")
        end
        Em = Transition1(; c=c_m, j=j_m, b=b_m, show=show, M=M)
        Ef = Transition1(; c=0, j=0,  b=0, show=false, M=M)
    elseif homing == :f
        if show
            println("Homing in females")
        end
        Em = Transition1(; c=0, j=0, b=0, show=false, M=M)
        Ef = Transition1(; c=c_f, j=j_f, b=b_f, show=show, M=M)
    end
    
    E_temp = [[Ef[i]*transpose(Em[j]) for j in 1:M2] for i in 1:M2]
    for j in 1:M2 
        for i in 1:M2
            for d in 1:M1 E_temp[i][j][d,d]=E_temp[i][j][d,d]/2; end
        end
    end
    
    e=[[[E_temp[i][j][MM_Ind[ii][1],MM_Ind[ii][2]]+E_temp[i][j][MM_Ind[ii][2],MM_Ind[ii][1]] for ii in eachindex(MM_Ind)]   for i in 1:M2] for j in 1:M2] 
    return e
end

InputMulti=function(Npop;M=3, pm=0.1,tp=Int64)
    M1=Int(M*(M+1)/2)+1
    M2=Int(M1*(M1+1)/2)
    Nvec_f=zeros(tp, M2);Nvec_f[1]=round(tp,Npop/2)
    Nvec_m=zeros(tp, M2);Nvec_m[1]=round(tp,(1-pm)*Npop/2)
    Nvec_m[M2]=round(tp,(pm)*Npop/2)
    return Nvec_f,Nvec_m
end



# Other functions for results analysis

TimePerc=function(n;maxt=300,perc=0.5)
    n0=n[1]
    tt=filter(x->n[x]<=n0*perc,1:maxt)
    if length(tt)==0
        tt=[1,1]
    end
    return first(tt),last(tt),last(tt)-first(tt)
end

Prot_minimum_per_site=function(model;maxgen=500)
    if length(size(model))>2
    nsites=length(model[1,1,:])
    temp=[findmin([model[1,t,s]/model[1,1,s] for t in 1:maxgen]) for s in 1:nsites]
    else 
        nsites=length(model[1,:])
        temp=[findmin([model[t,s]/model[1,s] for t in 1:maxgen]) for s in 1:nsites]
    end
    temp1=first.(temp)
    temp2=last.(temp) 
    return temp1,temp2
end

Prot_avarage=function(model;diam=14,r,maxgen=500,perc=0.33)
    #r=Nsize_r0(;diam=diam,maxd=diam);
    tt=Protection_per_site(model;p=perc,maxgen=maxgen)
    res=zeros(length(r))
    for rr in eachindex(r)
        res[rr] = mean(tt[r[rr]])
    end
    return res
end

WaveWidth=function(model; maxgen=500, r,p=0.33)
    if length(size(model))<3
        nsites=length(model[1,:])
        temp=[[t for t in 1:maxgen if model[t,s]<=p] for s in 1:nsites]

    else 
        nsites=length(model[1,1,:])
        temp=[[t for t in 1:maxgen if model[1,t,s]/model[1,1,s]<=p] for s in 1:nsites]
    end
    ss=[s for s in 1:length(temp) if length(temp[s])>0]
    t1=[mean(minimum.(temp,init=1000)[intersect(r[rr], ss)]) for rr in eachindex(r)]
    t2=[mean(maximum.(temp,init=-1000)[intersect(r[rr], ss)])  for rr in eachindex(r)]

    return t1, t2
end
