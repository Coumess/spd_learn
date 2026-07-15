using Pkg                           # Installer le package
# Pkg.add("PermutationTests")
# Pkg.add("Plots")
using DelimitedFiles                # Charger les packages
using PermutationTests
using Plots
using Random
using Statistics
using Printf

# Colonnes des CSV (ordre des activations dans test_stat.py)
labels = ["ReEig", "ExpT", "ExpP", "Cosh"]

# Formatage p-value façon papier (digits=3). Comme p >= minp, on ne peut
# écrire "p<0.001" que si le test peut réellement descendre sous 0.001.
fmt_p(p) = p < 0.001 ? "p<0.001" : @sprintf("p=%.3f", p)

#%%
# Charger le fichier des résultats (un seul dataset)
file = "results_4activations.csv"
y = readdlm(file , ',', skipstart=1)                    # skipstart=1 signifie qu'on enlève l'entête

# Nom de la database
println("Resultats utilisés : ", file)

# Vérificatione la taille de y (doit être (folds*seeds) x nbr de couches)
println("Taille de y : ", size(y))

# Transformer y en vecteur
y_vec = vec(permutedims(y))
#y_vec = vcat(y...)
# Vérification : y doit etre de taille (folds*seeds)*nbr_couches
println("Longueur y_vec : ", length(y_vec))
println("Première ligne y :")
println(y[1,:])

println("Premier bloc y_vec :")
println(y_vec[1:size(y,2)])

#%%
#---------------------------------------
# MEAN & STD (across folds and seeds)
#---------------------------------------
N = size(y, 1)                                             # nombre de runs = folds*seeds
K = size(y, 2)                                             # nombre de couches (activations)

println("\n=== MEAN ± STD (across folds and seeds) | N=$N runs ===")
for k in 1:K
    @printf("%-8s : %.3f ± %.3f\n", labels[k], mean(y[:,k]), std(y[:,k]))
end

#%%
#---------------------------------------
# TEST OMNIBUS
#---------------------------------------
N_rows = size(y, 1)
K_cols = size(y,2)
res = anovaTestRM(y_vec, (n=N_rows, k=K_cols))
println("\n=== RESULTAT ANOVA (omnibus, repeated-measures) ===")
println(res)
@printf("F = %.3f | %s | minp = %.2e\n", res.obsstat, fmt_p(res.p), res.minp)


#%%
#---------------------------------------
# TEST POST HOC
#---------------------------------------
# Paramètres
NK = N * K

println(NK)

# Construction des différences : TOUTES les paires (i<j)
pairs = [(i, j) for i in 1:K for j in i+1:K]               # 6 paires pour K=4
diffs = [y_vec[i:K:NK] .- y_vec[j:K:NK] for (i, j) in pairs]

# Test (correction multiple des comparaisons intégrée)
pht = studentMcTestRM(diffs)

println("\n=== RESULTATS POST-HOC (paired Student t, multiple-comparison corrected) ===")
@printf("minp = %.2e\n", pht.minp)
for (m, (i, j)) in enumerate(pairs)
    @printf("%-8s vs %-8s : t = %+.3f | %s\n",
            labels[i], labels[j], pht.obsstat[m], fmt_p(pht.p[m]))
end

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

#=
#%%
#---------------------------------------
# DONNEES FICTIVES
#---------------------------------------
# Paramètres
N = 25
K = 3
n_iter = 10000

pvals = zeros(n_iter)

# Test pour H0
for i in 1:n_iter

    # H0 vraie : aucune différence entre colonnes
    y = randn(N, K)

    # Format pour le test
    y_vec = vec(permutedims(y))

    # ANOVA
    res = anovaTestRM(y_vec, (n=N, k=K))

    # stocker p-value
    pvals[i] = res.p
end

# Taux de rejet
# ---------------------------
println("Empirical rejection rate (alpha=0.05): ",
        mean(pvals .< 0.05))

# HISTOGRAMME
# ---------------------------
histogram(
    pvals,
    bins = 20,
    normalize = true,
    label = "Empirical p-values",
    xlabel = "p-value",
    ylabel = "Density",
    title = "Distribution of p-values under H0"
)

# ligne théorique uniforme
plot!([0,1], [1,1],lw=2, linestyle=:dash,label="Uniform(0,1)")

#%%

# Paramètres
N = 25
K = 3
n_iter = 10000

pvals = zeros(n_iter)

# Test pour H1
for i in 1:n_iter

    # H1 vraie : différence entre colonnes
    y = randn(N, K)

    # une couche meilleure
    y[:,3] .+= 0.5

    # Format pour le test
    y_vec = vec(permutedims(y))

    # ANOVA
    res = anovaTestRM(y_vec, (n=N, k=K))

    # stocker p-value
    pvals[i] = res.p
end

# Taux de rejet
# ---------------------------
println("Empirical power (alpha=0.05): ",
        mean(pvals .< 0.05))

# HISTOGRAMME
# ---------------------------
histogram(
    pvals,
    bins = 20,
    normalize = true,
    label = "H1 p-values",
    xlabel = "p-value",
    ylabel = "Density",
    title = "Distribution of p-values under H1"
)

=#
