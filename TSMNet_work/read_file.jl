using NPZ

file = joinpath(homedir(), "Documents", "BCI", "data", "FII_BCI_Corpus", "MatCov", "Schirrmeister2017_1_covmats.npy")

isfile(file)

o=npzread(file)
