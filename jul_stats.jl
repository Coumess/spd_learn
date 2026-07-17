using Pkg                           # Installer le package
# Pkg.add("PermutationTests")
# Pkg.add("Plots")
using DelimitedFiles                # Charger les packages
using PermutationTests
using Plots
using Random
using Statistics
using Printf


# Colonnes du CSV (ordre des activations dans test_stat.py)
labels = ["ReEig", "CoshPnorm"]


# Formatage p-value façon papier (digits=3). Comme p >= minp, on ne peut
# écrire "p<0.001" que si le test peut réellement descendre sous 0.001.
fmt_p(p) = p < 0.001 ? "p<0.001" : @sprintf("p=%.3f", p)


#%%
# Charger le fichier des résultats (un seul dataset)
file = "results_4activations_Zhou20168_3_coshPnorm.csv"
y = readdlm(file , ',', skipstart=1)                    # skipstart=1 signifie qu'on enlève l'entête


# Nom de la database
println("Resultats utilisés : ", file)


# Vérificatione la taille de y (doit être (folds*seeds) x 2)
println("Taille de y : ", size(y))
println("Première ligne y :")
println(y[1,:])


N = size(y, 1)                                             # nombre de runs = folds*seeds
K = size(y, 2)                                             # nombre de couches (activations)


if K != 2
    error("Ce script est pour K=2 (ReEig vs CoshPnorm). Ici K=$K -> utilise test_stat_ju.jl")
end


#%%
#---------------------------------------
# MEAN & STD (across folds and seeds)
#---------------------------------------
println("\n=== MEAN ± STD (across folds and seeds) | N=$N runs ===")
for k in 1:K
    @printf("%-10s : %.3f ± %.3f\n", labels[k], mean(y[:,k]), std(y[:,k]))
end


#%%
#---------------------------------------
# TEST APPARIE
# K=2 -> pas d'omnibus ni de correction multiple :
# l'ANOVA a mesures repetees se reduit exactement au t apparie,
# et il n'y a qu'une seule comparaison.
#---------------------------------------
d = y[:,1] .- y[:,2]                                       # ReEig - CoshPnorm


# Monte Carlo 20000 permutations : meme reglage que l'ANOVA/post-hoc du script
# a 4 activations (minp = 5e-5, donc "p<0.001" reste atteignable) et rapide.
res = studentTest1S(d; switch2rand=1, nperm=20000)
# Test EXACT (2^N permutations) : plus rigoureux mais tres lent pour N=25
# (2^25 = 33.5M permutations, plusieurs minutes) :
# res = studentTest1S(d)


println("\n=== TEST APPARIE (paired Student t, permutation) ===")
println(res)
@printf("t = %+.3f | %s | minp = %.2e | nperm = %d\n", res.obsstat, fmt_p(res.p), res.minp, res.nperm)
@printf("difference moyenne (%s - %s) = %+.3f\n", labels[1], labels[2], mean(d))


#%% Bar plot moyennes et écarts types


gr()


moyennes = [mean(y[:,k]) for k in 1:K]
ecarts   = [std(y[:,k])  for k in 1:K]


x = 1:length(labels)


p = bar(
    x, moyennes,
    xticks=(x, labels),
    yerr=ecarts,
    legend=false,
    size=(700,400),
    color=:steelblue,
    guidefont=font(12, "Times"),
    tickfont=font(12, "Times")
)


ymin = minimum(moyennes .- ecarts)
ymax = maximum(moyennes .+ ecarts)
margin = 0.05 * (ymax - ymin)


ylims!(ymin - margin, ymax + margin)


offset = 0.03 * (ymax - ymin)


for i in x
    texte = string(round(moyennes[i], digits=3),
                   " ± ",
                   round(ecarts[i], digits=3))


    annotate!(
        p,
        i,
        moyennes[i] + ecarts[i] + offset,
        text(texte, 12, "Times")
    )
end


display(p)