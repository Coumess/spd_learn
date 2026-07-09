using NPZ
using Eegle
using YAML


# PUT HERE path to the MI folder of the FII corpus on your PC
# MIDir = joinpath(@__DIR__, "MI") 
MIDir = "C:/Users/coumesa/Documents/BCI/data/FII_BCI_Corpus/MI"
classes = ["left_hand", "right_hand"]; # or for example ["left_hand", "right_hand"]; ["feet",]   # Chosen classes

# select MI databases comprising the given 'classes' and minimum number of trials
inclusion = (("tpc", x -> minimum(values(x)) > 24),)

outDir = "C:/Users/coumesa/Documents/BCI/data/FII_BCI_Corpus/Tensor"    # output to save the tensor 

DBs = selectDB(MIDir, :MI; classes, inclusion); # Select all the databases folowing the criteria

# Create and save all data
for DB ∈ DBs
    
    @info "\n Writin data for databse $(DB.dbName)"

    #=  Compute the rate for the resampling if needed
    fs = DB.sr

    if fs == 250 || fs == 256
        rate_rs = 1.0
    else
        rate_rs = 250/fs
    end 
    =# 

    for (d, file) in enumerate(DB.files)
        
        println("writing file $d of $(length(DB.files))")

        o = readNY(file; bandPass = (4, 36), bpDesign = Butterworth(4), upperLimit = 1.2, classes) # Change the rate depends on the database
        
        k = length(o.trials) # number of trials
        n = o.ne # number of channels  
        t = o.wl # number of samples
        
        out = Array{Float32, 3}(undef, k, n, t)

        @simd for i in 1:k
            @inbounds out[i, :, :] = Float32.(transpose(o.trials[i]))
        end

        npzwrite(joinpath(outDir,"$(DB.dbName)_$(d)_signal_eeg.npy"), out)
        npzwrite(joinpath(outDir,"$(DB.dbName)_$(d)_labels_eeg.npy"), o.y)
    end
    
end

@info "Finished!"