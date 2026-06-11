
#Multiplexing M1 modelling (running simulations)
NospM_run_NT=function(;SexDistortion=0.5,B=0.01, C=0.95,J=0.05,sigma=0.01,s=1,h=0.3,hDR=0.02,hm=0,hN=0.02,sN=1,wa=1,F=12,phi1=1,phi2=1,Npop=10^8,rep=500,maxgen=500,Release=0.1,M=3,filename="Multi",homing=:b,beta=0,showprogress=false,dir="",tp=Int128)
    M1=Int(M*(M+1)/2)+1

    M2=Int(M1*(M1+1)/2)
    R1_temp=[]
    R1_outcome=[]
    R1_int=[]

    fn=string(dir,filename, "_Npop_",log10.(Npop),"_B_",B,"_J_",J,"_C_",C,"h_",h,"_m_",SexDistortion,".", "jld2")
    fn2=string(dir,"Outcome",filename, "_Npop_",log10.(Npop),"_B_",B,"_J_",J,"_C_",C,"h_",h,"_m_",SexDistortion,".", "jld2")
    fn3=string(dir,"Integral",filename, "_Npop_",log10.(Npop),"_B_",B,"_J_",J,"_C_",C,"h_",h,"_m_",SexDistortion,".", "jld2")

    e = E_Make_Multi(;c_m=C,c_f=C,j_m=J,j_f=J,b_m=B,b_f=B,M=M,homing=homing)
    wmf=FitnessMulti2(;sigma=sigma,s=s,h=h,hN=hN,sN=sN,F=F)


    if !showprogress 
        for rp in 1:rep
            mod=Model_nospM(;N=Npop,wmf=wmf,e=e,mD_sex=SexDistortion,M=M,maxgen=maxgen,Model=2,wa=wa, phi1=phi1,phi2=phi2,pm=Release,beta=beta,tp=tp)
            push!(R1_temp,mod)
            push!(R1_outcome,last.(mod))
            push!(R1_int,IntegrateSlope(mod[1][1:maxgen];to_gen=maxgen,to_min=false)[1])
        end
    else 
        @showprogress 1 for rp in 1:rep
            sleep(0.1)
            mod=Model_nospM(;N=Npop,wmf=wmf,e=e,mD_sex=SexDistortion,M=M,maxgen=maxgen,Model=2v, phi1=phi1,phi2=phi2,pm=Release,beta=beta,tp=tp)
            push!(R1_temp,mod)
            push!(R1_int,IntegrateSlope(mod[1][1:maxgen];to_gen=maxgen,to_min=false)[1])
        end
    end
    jldsave(fn, true; models1=R1_temp)
    jldsave(fn2, true; res=R1_outcome)
    jldsave(fn3, true; res=R1_int)
end



function Sp_runDistr(; wmf, SexDistortion=0.5, B=0.001, C=0.95, J=0.05, A=0,
                 Npop=10^6, rep=500, maxgen=500, dia=10, mig=0.01,Model=2,
                 Release=0.1, Release_sites=[], NumRelease=10^3,alpha=0,phi1=1,phi2=1,wa=1,
                 M=4, filename="Sp", homing=:b, beta=100, track=false,i0=1,stats_only=true,initN=false,Nfv0=[], Nmv0=[],Nfv=[],Nmv=[])
    M2 = Int(M * (M + 1) ÷ 2)
    if isempty(Release_sites) Release_sites = [dia];end

    f=wmf[3][1,1]
    # Precompute shared data 
    e = E_Make(; c_m=C, j_m=J, a_m=A, b_m=B,c_f=C, j_f=J, a_f=A, b_f=B, M=M, homing=homing)
    p_in = 1 .- mig
    Nsites=NSites(dia)
    Adj=Test_input(; num=dia, Nreleases=1, Npop=Npop,p_in=p_in, maxdist=1, rebound=true)
    tr = track ? "_T" : ""
    fn_prefix = string(filename, tr, "_radius_", dia, "_B_", B, "_J_", J, "_C_", C,"_sd_", SexDistortion, "_mig_", mig)

    if !initN 
        Nfv, Nmv = Initial_vec_fixed(; N=Npop, Nsites=Nsites,Release_sites=Release_sites, pm=Release, NumRelease=NumRelease, M2=M2)
        Nfv0, Nmv0 = Initial_vec_fixed(; N=Npop, Nsites=Nsites,Release_sites=[], pm=0, NumRelease=0, M2=M2)
        fn_prefix = string(filename, tr, "_radius_", dia, "_N_", log10(Npop), "_B_", B, "_J_", J, "_C_", C,"_sd_", SexDistortion, "_mig_", mig)

    end
    if alpha==0
        alpha =[ Alpha(;phi1=phi1,phi2=phi2,f=f,N=sum(Nfv0.+Nmv0;dims=1)[s]) for s in 1:Nsites]
    end
    # Run replicates in parallel across workers
    files = pmap(i -> run_replicate(i; wmf=wmf, Nmv=Nmv,Nfv=Nfv, Model=Model,SexDistortion=SexDistortion, e=e,Adj=Adj,maxgen=maxgen,alpha=alpha,wa=wa,phi1=phi1,phi2=phi2,
                                    Release=Release, Release_sites=Release_sites,NumRelease=NumRelease, M=M, homing=homing,
                                    beta=beta, track=track, fn_prefix=fn_prefix,stats_only=stats_only),i0:rep+i0-1)
    return files
end





function Sp_run_M1_Distr(; wmf, SexDistortion=0.5, B=0.001, C=0.95, J=0.05, 
                 Npop=10^6, rep=500, maxgen=500, dia=10, mig=0.01,alpha=0,phi1=1,phi2=1,wa=1,
                 Release=0.1, Release_sites=[], NumRelease=10^3,
                 M=4, filename="SpM1", homing=:b, beta=0, track=false,stats_only=true,i0=1,initN=false,Nfv0=[], Nmv0=[],Nfv=[],Nmv=[])

    f = wmf[3][1, 1]
    tr = track ? "_T" : ""
    fn_prefix = string(filename, tr, "_radius_", dia,"_N_", log10(Npop), "_B_", B, "_J_", J, "_C_", C,"_sd_", SexDistortion,"_f_",f, "_mig_", mig)

    M=M-1             
    M1= Int(M * (M + 1) ÷ 2)+1
    M2= Int(M1 * (M1 + 1) ÷ 2)
    if isempty(Release_sites); 
        Release_sites = [dia]; 
    end


    p_in = 1 .- mig
    Nsites=NSites(dia)
    Adj = Test_input(; num=dia, Nreleases=1, Npop=Npop,p_in=p_in, maxdist=1, rebound=true)
    e = E_Make_Multi(; c_m=C, j_m=J, b_m=B,c_f=C, j_f=J, b_f=B, M=M, homing=homing)
    
    if !initN 
        Nfv, Nmv = Initial_vec_fixed(; N=Npop, Nsites=Nsites,Release_sites=Release_sites, pm=Release,ReleaseGT=M2, NumRelease=NumRelease, M2=M2)
        Nfv0, Nmv0 = Initial_vec_fixed(; N=Npop, Nsites=Nsites,Release_sites=[], pm=0, NumRelease=0, ReleaseGT=M2,M2=M2)
        fn_prefix = string(filename, tr, "_radius_", dia, "_N_", log10(Npop), "_B_", B, "_J_", J, "_C_", C,"_sd_", SexDistortion, "_mig_", mig)

    end
    if alpha==0
        alpha =[ Alpha(;phi1=phi1,phi2=phi2,f=f,N=sum(Nfv0.+Nmv0;dims=1)[s]) for s in 1:Nsites]
    end
    r_sex = fill(0.5, M2, M2)
    for i in (M2-M1+1):M2
        r_sex[i, :] .= SexDistortion
    end
    #println(M, M1,M2)
    files = pmap(i -> run_replicateM1(i; wmf=wmf, r_sex=r_sex,E=e,Adj=Adj, Nfv=Nfv,Nmv=Nmv,
                                    maxgen=maxgen,alpha=alpha, phi1=phi1,phi2=phi2,wa=wa,Release=Release, Release_sites=Release_sites,
                                     M2=M2, homing=homing,beta=beta, track=track, fn_prefix=fn_prefix,stats_only=stats_only),i0:rep+i0-1)
    return files
end





# Calculation of population with allele/genotype filter (relative to the initial population size)
function Sp_runDistrO(; wmf, SexDistortion=0.5, B=0.001, C=0.95, J=0.05, A=0,freq=false,
                 Npop=10^6, rep=500, maxgen=500, dia=10, mig=0.01,allele="R",
                 Release=0.1, Release_sites=[], NumRelease=10^3,alpha=0,phi1=1,phi2=1,wa=1,sex=:b,
                 M=4, filename="Sp", homing=:b, beta=100, track=false,i0=1,stats_only=true,initN=false,Nfv0=[], Nmv0=[],Nfv=[],Nmv=[])
    M2 = Int(M * (M + 1) ÷ 2)
    if M==4 
        namesG=["WW","WD","WN","WR","DD","DN","DR","NN","NR","RR"]
    else 
        namesG=["WW","WD","WN","WR","WO","DD","DN","DR","DO","NN","NR","NO", "RR","RO","OO"]
    end

    R1Ind=sum([occursin.(allele[i],namesG) for i in eachindex(allele)]).>0
    
    if isempty(Release_sites) Release_sites = [dia];end

    f=wmf[3][1,1]
    # Precompute shared data
    e = E_Make(; c_m=C, j_m=J, a_m=A, b_m=B,c_f=C, j_f=J, a_f=A, b_f=B, M=M, homing=homing)
    p_in = 1 - mig
    Nsites=NSites(dia)
    Adj=Test_input(; num=dia, Nreleases=1, Npop=Npop,p_in=p_in, maxdist=1, rebound=true)
    tr = track ? "_T" : ""
    fn_prefix = string(filename, tr, "_radius_", dia, "_B_", B, "_J_", J, "_C_", C,"_sd_", SexDistortion, "_mig_", mig)

    if !initN 
        Nfv, Nmv = Initial_vec_fixed(; N=Npop, Nsites=Nsites,Release_sites=Release_sites, pm=Release, NumRelease=NumRelease, M2=M2)
        Nfv0, Nmv0 = Initial_vec_fixed(; N=Npop, Nsites=Nsites,Release_sites=[], pm=0, NumRelease=0, M2=M2)
        fn_prefix = string(filename, tr, "_radius_", dia, "_N_", log10(Npop), "_B_", B, "_J_", J, "_C_", C,"_sd_", SexDistortion, "_mig_", mig)

    end
    if alpha==0
        alpha =[ Alpha(;phi1=phi1,phi2=phi2,f=f,N=sum(Nfv0.+Nmv0;dims=1)[s]) for s in 1:Nsites]
    end

    # Run replicates in parallel across workers
    files = pmap(i -> run_replicateO(i; freq=freq,wmf=wmf, Nmv=Nmv,Nfv=Nfv, SexDistortion=SexDistortion, e=e,Adj=Adj,maxgen=maxgen,alpha=alpha,wa=wa,phi1=phi1,phi2=phi2,
                                    Release=Release, Release_sites=Release_sites,NumRelease=NumRelease, M=M, sex=sex,homing=homing,R1Ind=R1Ind,#R11Ind=R11Ind,
                                    beta=beta, track=track, fn_prefix=fn_prefix,stats_only=stats_only),i0:rep+i0-1)
    return files
end



function Sp_runDistrO_cut(; wmf, SexDistortion=0.5, B=0.001, C=0.95, J=0.05, A=0,
                 Npop=10^6, rep=500, maxgen=500, dia=10, mig=0.01,allele="R",threshold=1,p=0.33,Model=2,
                 Release=0.1, Release_sites=[], NumRelease=10^3,alpha=0,phi1=1,phi2=1,wa=1,sex=:b,
                 M=4, filename="Sp", homing=:b, beta=100, track=false,i0=1,stats_only=true,initN=false,Nfv0=[], Nmv0=[],Nfv=[],Nmv=[])
    M2 = Int(M * (M + 1) ÷ 2)
    if M==4 
        namesG=["WW","WD","WN","WR","DD","DN","DR","NN","NR","RR"]
    else 
        namesG=["WW","WD","WN","WR","WO","DD","DN","DR","DO","NN","NR","NO", "RR","RO","OO"]
    end
    #R1Ind=occursin.(allele,namesG)
    R1Ind=sum([occursin.(allele[i],namesG) for i in eachindex(allele)]).>0
    
    if isempty(Release_sites) Release_sites = [dia];end

    f=wmf[3][1,1]
    # Precompute shared data (these could be broadcast once if very heavy)
    e = E_Make(; c_m=C, j_m=J, a_m=A, b_m=B,c_f=C, j_f=J, a_f=A, b_f=B, M=M, homing=homing)
    p_in = 1 - mig
    Nsites=NSites(dia)
    Adj=Test_input(; num=dia, Nreleases=1, Npop=Npop,p_in=p_in, maxdist=1, rebound=true)
    tr = track ? "_T" : ""
    fn_prefix = string(filename, tr, "_radius_", dia, "_B_", B, "_J_", J, "_C_", C,"_sd_", SexDistortion, "_mig_", mig)

    if !initN 
        Nfv, Nmv = Initial_vec_fixed(; N=Npop, Nsites=Nsites,Release_sites=Release_sites, pm=Release, NumRelease=NumRelease, M2=M2)
        Nfv0, Nmv0 = Initial_vec_fixed(; N=Npop, Nsites=Nsites,Release_sites=[], pm=0, NumRelease=0, M2=M2)
        fn_prefix = string(filename, tr, "_radius_", dia, "_N_", log10(Npop), "_B_", B, "_J_", J, "_C_", C,"_sd_", SexDistortion, "_mig_", mig)

    end
    if alpha==0
        alpha =[ Alpha(;phi1=phi1,phi2=phi2,f=f,N=sum(Nfv0.+Nmv0;dims=1)[s]) for s in 1:Nsites]
    end

    files = pmap(i -> run_replicateO_cut(i; threshold=threshold,p=p,wmf=wmf, Nmv=Nmv,Nfv=Nfv, SexDistortion=SexDistortion, e=e,Adj=Adj,maxgen=maxgen,alpha=alpha,wa=wa,phi1=phi1,phi2=phi2,
                                    Release=Release, Release_sites=Release_sites,NumRelease=NumRelease, M=M, Model=Model,sex=sex,homing=homing,R1Ind=R1Ind,#R11Ind=R11Ind,
                                    beta=beta, track=track, fn_prefix=fn_prefix,stats_only=stats_only),i0:rep+i0-1)
    return files
end

function Model_spO_cut(; Ntf_init, Ntm_init, Adj, Release_sites, wmf, E, mD_sex=0.5,wa=1,
    alpha=0,phi1=1,phi2=1, maxgen=100, Model=2, bound=1, R1Ind,sex=:b,
    mu_mig=1, maxdist=1, M=4, beta=0, tp=Int64,
    track=false, stats_only=true, to_first=true, threshold=1,p=0.33)

    M2 = Int(div(M * (M + 1), 2))

    Nsites = size(Adj[1], 1)

    f = wmf[3][1, 1]

   if alpha == 0
        Npop=sum(Ntf_init.+Ntm_init)
        alpha = [Alpha(;N=Npop[s],f=f, phi1=phi1,phi2=phi2) for s in 1:Nsites]
    end
    #if length(threshold)<Nsites
    #    threshold=Npop.*threshold
    #end
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

    # Preallocate population matrices (just current/next)
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
            # fast Stat1 calculation
            stat_val =
                (sum(view(Ntf_next, R1Ind, site))*k1 +
                sum(view(Ntm_next, R1Ind, site))*k2) / denom[site]

            Stat1[gen, site] = stat_val
            # stop early if threshold reached
            if to_first && stat_val >= threshold#[site]
                found_gen = gen
                found_site = site
                return found_gen, found_site,found_gen2, found_site2
            end

        end

        # migration
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



function Sp_runDistrO_cut_flagged(; wmf, SexDistortion=0.5, B=0.001, C=0.95, J=0.05, A=0,
                 Npop=10^6, rep=500, maxgen=500, dia=10, mig=0.01,allele="R",threshold=1,p=0.33,Model=2,
                 Release=0.1, Release_sites=[], NumRelease=10^3,alpha=0,phi1=1,phi2=1,wa=1,sex=:b,
                 M=4, filename="Sp", homing=:b, beta=100, track=false,i0=1,stats_only=true,initN=false,Nfv0=[], Nmv0=[],Nfv=[],Nmv=[])
    M2 = Int(M * (M + 1) ÷ 2)
    if M==4 
        namesG=["WW","WD","WN","WR","DD","DN","DR","NN","NR","RR"]
    else 
        namesG=["WW","WD","WN","WR","WO","DD","DN","DR","DO","NN","NR","NO", "RR","RO","OO"]
    end
    #R1Ind=occursin.(allele,namesG)
    R1Ind=sum([occursin.(allele[i],namesG) for i in eachindex(allele)]).>0
    
    if isempty(Release_sites) Release_sites = [dia];end

    f=wmf[3][1,1]
    # Precompute shared data (these could be broadcast once if very heavy)
    e = E_Make(; c_m=C, j_m=J, a_m=A, b_m=B,c_f=C, j_f=J, a_f=A, b_f=B, M=M, homing=homing)
    p_in = 1 - mig
    Nsites=NSites(dia)
    Adj=Test_input(; num=dia, Nreleases=1, Npop=Npop,p_in=p_in, maxdist=1, rebound=true)
    tr = track ? "_T" : ""
    fn_prefix = string(filename, tr, "_radius_", dia, "_B_", B, "_J_", J, "_C_", C,"_sd_", SexDistortion, "_mig_", mig)

    if !initN 
        Nfv, Nmv = Initial_vec_fixed(; N=Npop, Nsites=Nsites,Release_sites=Release_sites, pm=Release, NumRelease=NumRelease, M2=M2)
        Nfv0, Nmv0 = Initial_vec_fixed(; N=Npop, Nsites=Nsites,Release_sites=[], pm=0, NumRelease=0, M2=M2)
        fn_prefix = string(filename, tr, "_radius_", dia, "_N_", log10(Npop), "_B_", B, "_J_", J, "_C_", C,"_sd_", SexDistortion, "_mig_", mig)

    end
    if alpha==0
        alpha =[ Alpha(;phi1=phi1,phi2=phi2,f=f,N=sum(Nfv0.+Nmv0;dims=1)[s]) for s in 1:Nsites]
    end

    files = pmap(i -> run_replicateO_cut_flagged(i; threshold=threshold,p=p,wmf=wmf, Nmv=Nmv,Nfv=Nfv, SexDistortion=SexDistortion, e=e,Adj=Adj,maxgen=maxgen,alpha=alpha,wa=wa,phi1=phi1,phi2=phi2,
                                    Release=Release, Release_sites=Release_sites,NumRelease=NumRelease, M=M, Model=Model,sex=sex,homing=homing,R1Ind=R1Ind,#R11Ind=R11Ind,
                                    beta=beta, track=track, fn_prefix=fn_prefix,stats_only=stats_only),i0:rep+i0-1)
    return files
end



# Combining simulations files
function merge_results(fn_prefix, rep)   #works
    all_models = Vector{Any}(undef, rep)
    for i in 1:rep
        fn = string(fn_prefix, "_rep_", i, ".jld2")
        @load fn mod
        all_models[i] = mod
    end
    jldsave(string(fn_prefix,".jld2"), true; models1=all_models)
    return all_models
end
function delete_files(fn_prefix,rep)   #works
    for i in 1:rep
        fn = string(fn_prefix, "_rep_", i, ".jld2")

        try
            rm(fn; force=true)
        catch err
            @warn "Could not delete file $fn" err
        end
    end
end