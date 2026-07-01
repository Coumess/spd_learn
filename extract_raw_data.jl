using NPZ
using Eegle


# PUT HERE path to the MI folder of the FII corpus on your PC
# MIDir = joinpath(@__DIR__, "MI") 
MIDir = "C:/Users/coumesa/Documents/BCI/data/FII_BCI_Corpus/MI/Zhou2016/subject_01_session_01.npz"
classes = ["left_hand", "right_hand"]; # or for example ["left_hand", "right_hand"]; ["feet",]   # Chosen classes

# select MI databases comprising the given 'classes' and minimum number of trials
inclusion = (("tpc", x -> minimum(values(x)) > 24),)

# DBs = selectDB(MIDir, :MI; classes, inclusion); # Select databases

outDir = "C:/Users/coumesa/Documents/BCI/data/FII_BCI_Corpus/Tensor"    # output to save the tensor 

o = readNY(MIDir; bandPass = (4, 36), upperLimit = 1.2, classes) # read session

k = length(o.trials)# number of trials
n = o.ne# number of channels  
t = o.wl # number of samples  

out = Array{Float32, 3}(undef, k, n, t) 

@simd for i in 1:k
    @inbounds out[i, :, :] = Float32.(transpose(o.trials[i]))
end

labels = o.y

println("Taille du tenseur : ", size(out))
println("Taille des labels : ", size(labels))
println("Labels : ", labels)

@info "Finished!"