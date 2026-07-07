import os
import sys

import torch
import torch.nn as nn

# spd_learn n'est pas installe (pip install -e .) dans cet environnement :
# on ajoute la racine du repo au sys.path pour pouvoir l'importer depuis
# n'importe quel dossier.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from spd_learn.modules import CovLayer

"""
Objectif de ce script :

Comprendre pas a pas ce que fait le "cnn" de TSMNet (les 2 premieres
couches Conv2d + le CovLayer), AVANT le bloc SPDNet (BiMap + ReEig).

C'est cette partie qui fixe la dimension d'entree du BiMap :
    BiMap(in_features=n_spatiotemp_filters, out_features=n_bimap_filters)

=> in_features du BiMap = n_spatiotemp_filters (nombre de filtres de la
   2e conv), PAS n_chans (nombre d'electrodes EEG) !

Reference : spd_learn/models/tsmnet.py -> TSMNet.cnn / TSMNet.covpool
"""

# -------------------------------------------------------
# Hyperparametres (mêmes noms que dans TSMNet.__init__)
# -------------------------------------------------------
batch_size = 4
n_chans = 22               # nombre d'electrodes EEG
n_times = 500               # nombre d'echantillons temporels par epoch
n_temp_filters = 4           # filtres de la conv temporelle
temp_kernel_length = 25       # longueur du kernel temporel
n_spatiotemp_filters = 40      # filtres de la conv spatiale (= dim du BiMap en entree)

# -------------------------------------------------------
# Donnee brute EEG : (batch, n_chans, n_times)
# -------------------------------------------------------
x = torch.randn(batch_size, n_chans, n_times)
print(f"[0] Entree brute EEG            : {tuple(x.shape)}  (batch, n_chans, n_times)")

# TSMNet.forward fait x[:, None, ...] avant le cnn : on ajoute un canal
# "image" factice de taille 1, pour pouvoir utiliser des Conv2d comme sur
# une image (hauteur = electrodes, largeur = temps, canal = 1).
x_img = x[:, None, ...]
print(f"[1] Apres x[:, None, ...]        : {tuple(x_img.shape)}  (batch, 1, n_chans, n_times)")

# -------------------------------------------------------
# Couche 1 : convolution TEMPORELLE
# -------------------------------------------------------
# kernel_size=(1, temp_kernel_length) : hauteur=1 -> le kernel ne glisse
# JAMAIS le long des electrodes, seulement le long du temps.
# => chaque electrode est filtree independamment par les memes n_temp_filters
#    filtres temporels (un peu comme n_temp_filters filtres passe-bande appris).
# padding="same" -> la dimension temporelle est preservee.
temporal_conv = nn.Conv2d(
    1,
    n_temp_filters,
    kernel_size=(1, temp_kernel_length),
    padding="same",
    padding_mode="reflect",
)
x_temp = temporal_conv(x_img)
print(f"[2] Apres conv temporelle        : {tuple(x_temp.shape)}  (batch, n_temp_filters, n_chans, n_times)")

# -------------------------------------------------------
# Couche 2 : convolution SPATIALE
# -------------------------------------------------------
# kernel_size=(n_chans, 1) : hauteur=n_chans -> le kernel couvre TOUTES les
# electrodes a la fois (pas de padding -> "valid" conv), largeur=1 -> il ne
# glisse pas dans le temps.
# => a chaque instant, on combine lineairement les n_chans electrodes (x les
#    n_temp_filters canaux) en n_spatiotemp_filters "filtres spatiaux appris"
#    (analogue a des filtres CSP appris, cf. docstring de BiMap).
# La dimension "electrodes" disparait completement (n_chans -> 1).
spatial_conv = nn.Conv2d(n_temp_filters, n_spatiotemp_filters, (n_chans, 1))
x_spatiotemp = spatial_conv(x_temp)
print(f"[3] Apres conv spatiale          : {tuple(x_spatiotemp.shape)}  (batch, n_spatiotemp_filters, 1, n_times)")

# -------------------------------------------------------
# Flatten(start_dim=2) : fusionne les dims 2 et 3 -> (1 * n_times) = n_times
# Revient a un squeeze() de la dimension "electrodes" (=1).
# -------------------------------------------------------
flatten = nn.Flatten(start_dim=2)
x_filtered = flatten(x_spatiotemp)
print(f"[4] Apres Flatten(start_dim=2)   : {tuple(x_filtered.shape)}  (batch, n_spatiotemp_filters, n_times)")
print(
    "    -> On a maintenant un signal temporel avec n_spatiotemp_filters "
    "'electrodes virtuelles' au lieu des n_chans electrodes reelles."
)

# -------------------------------------------------------
# CovLayer : covariance sur l'axe temporel -> matrice SPD
# -------------------------------------------------------
covpool = CovLayer()
x_cov = covpool(x_filtered)
print(f"[5] Apres CovLayer (SPD)         : {tuple(x_cov.shape)}  (batch, n_spatiotemp_filters, n_spatiotemp_filters)")

is_sym = torch.allclose(x_cov, x_cov.transpose(-1, -2), atol=1e-5)
eigs = torch.linalg.eigvalsh(x_cov)
print(f"    -> Symetrique : {is_sym} | eigenvalues min={eigs.min().item():.4f}, max={eigs.max().item():.4f}")

print()
print("=" * 70)
print("Conclusion pour dimensionner un BiMap / bloc SPD custom :")
print(f"  - Le BiMap qui suit doit avoir in_features = n_spatiotemp_filters = {n_spatiotemp_filters}")
print(f"    (PAS n_chans = {n_chans})")
print("  - out_features du BiMap = dimension choisie pour le sous-espace SPD")
print("    (ex. n_bimap_filters=20 dans TSMNet par defaut)")
print("  - Le BiMap (ou ton bloc BiMap + activation ReEig/coshP/expT) doit")
print("    recevoir x_cov, PAS x_filtered ni x brut.")
print("=" * 70)


# =========================================================
# BONUS : comment choisir n_temp_filters ?
# =========================================================
"""
n_temp_filters = combien de filtres temporels APPRIS (banque de filtres,
un peu comme des passe-bandes) on applique a chaque electrode avant de
les combiner spatialement.

Deux choses la controlent : le NOMBRE de filtres (n_temp_filters) et leur
LONGUEUR (temp_kernel_length). Elles repondent a deux questions differentes :

  - temp_kernel_length -> QUELLES frequences un filtre peut representer
    (lie a la frequence d'echantillonnage de tes donnees, 250 Hz ici,
    cf. preprocessing/configs/config.yaml et extract_raw_data.jl)
  - n_temp_filters      -> COMBIEN de motifs temporels differents on
    apprend en parallele (taille de la "base spectrale")
"""

sfreq = 250.0  # Hz, frequence d'echantillonnage utilisee dans ce projet

print()
print("=" * 70)
print("BONUS : choix de n_temp_filters et temp_kernel_length")
print("=" * 70)

# ---------------------------------------------------------
# 1) temp_kernel_length <-> frequence minimale representable
# ---------------------------------------------------------
# Un kernel FIR de longueur L (en echantillons) ne peut representer une
# oscillation que s'il couvre au moins ~1 cycle. Periode max couverte :
#   T_max = L / sfreq  ->  f_min ~= sfreq / L
# C'est une regle d'ordre de grandeur (pas une limite stricte), mais elle
# explique pourquoi ShallowConvNet/TSMNet choisissent L=25 @ 250Hz :
for L in [13, 25, 50, 100]:
    f_min = sfreq / L
    duration_ms = 1000 * L / sfreq
    print(f"  temp_kernel_length={L:>4} ({duration_ms:>5.0f} ms) -> f_min ~= {f_min:5.1f} Hz")
print(
    "  -> Avec L=25 (100 ms) a 250 Hz, f_min ~= 10 Hz : pile la borne basse\n"
    "     de la bande mu (8-13 Hz), utile en motor imagery. Si ta tache\n"
    "     depend de frequences plus basses (ex. delta/theta < 8Hz), il\n"
    "     faut ALLONGER le kernel, sinon ces frequences ne sont pas\n"
    "     representables du tout, quel que soit n_temp_filters."
)

# ---------------------------------------------------------
# 2) Verification empirique : reponse en frequence de filtres temporels
# ---------------------------------------------------------
# On prend les vrais poids appris (ici random, non-entraines) de la conv
# temporelle et on regarde leur spectre (FFT) pour voir qu'ils se
# comportent deja comme une mini banque de filtres passe-bande.
weights = temporal_conv.weight.detach()  # (n_temp_filters, 1, 1, temp_kernel_length)
n_fft = 256
freqs = torch.fft.rfftfreq(n_fft, d=1 / sfreq)
print(f"\n  Spectre (FFT) des {n_temp_filters} filtres temporels appris (poids actuels, non-entraines) :")
for i in range(n_temp_filters):
    kernel = weights[i, 0, 0]
    spectrum = torch.fft.rfft(kernel, n=n_fft).abs()
    peak_freq = freqs[spectrum.argmax()]
    print(f"    filtre {i}: frequence dominante ~= {peak_freq:5.1f} Hz")
print(
    "  -> Meme avec des poids random, chaque filtre a deja une frequence\n"
    "     dominante differente : n_temp_filters, c'est le nombre de ces\n"
    "     'bandes' que le reseau peut apprendre a specialiser."
)

# ---------------------------------------------------------
# 3) Cout en parametres : n_temp_filters se multiplie dans la conv spatiale
# ---------------------------------------------------------
print(f"\n  Impact de n_temp_filters sur le nombre de parametres")
print(f"  (n_chans={n_chans}, n_spatiotemp_filters={n_spatiotemp_filters}, kernel_temp={temp_kernel_length}) :")
print(f"  {'n_temp_filters':>15} | {'params conv temp.':>18} | {'params conv spatiale':>21} | {'total':>8}")
for n_tf in [1, 2, 4, 8, 16, 32]:
    params_temp = n_tf * 1 * temp_kernel_length + n_tf           # weight + bias
    params_spatial = n_spatiotemp_filters * n_tf * n_chans + n_spatiotemp_filters
    print(f"  {n_tf:>15} | {params_temp:>18} | {params_spatial:>21} | {params_temp + params_spatial:>8}")
print(
    "  -> n_temp_filters n'agrandit PAS la matrice SPD finale (toujours\n"
    "     n_spatiotemp_filters x n_spatiotemp_filters), mais il multiplie\n"
    "     directement le nombre de parametres de la conv spatiale qui suit\n"
    "     (n_spatiotemp_filters * n_temp_filters * n_chans poids).\n"
    "     Plus de temp_filters = base spectrale plus riche a combiner\n"
    "     spatialement, mais plus de parametres a apprendre -> risque de\n"
    "     surapprentissage si peu d'essais/sujets dans le dataset."
)

print()
print("=" * 70)
print("Regles pratiques pour choisir n_temp_filters (et temp_kernel_length) :")
print("=" * 70)
print(
    """
  1. temp_kernel_length se choisit d'abord, a partir de la frequence
     d'echantillonnage et des bandes de frequence utiles pour ta tache
     (ex. mu/beta 8-30 Hz en motor imagery -> ~100 ms @ 250Hz -> L=25).

  2. n_temp_filters depend ENSUITE surtout :
       - de la richesse spectrale attendue du signal (plusieurs bandes
         d'interet -> plus de filtres utiles, comme en FBCSP ou l'on
         utilise souvent 4 a 9 bandes manuelles)
       - de la taille du dataset : peu d'essais/sujets -> valeur petite
         (4-8) pour eviter le surapprentissage (c'est le choix de TSMNet,
         pense pour du cross-subject/domain adaptation avec peu de
         donnees par domaine)
       - du budget de calcul : il multiplie le nombre de parametres de
         TOUTE la conv spatiale qui suit (cf. tableau ci-dessus)

  3. Il n'y a pas de formule exacte : en pratique on fixe temp_kernel_length
     via la connaissance du signal (frequences), puis on choisit
     n_temp_filters par validation croisee sur quelques valeurs
     (ex. [4, 8, 16]) et on garde celle qui generalise le mieux sur le
     jeu de validation -> donc OUI, ca depend de la base de donnees
     (nombre d'essais, nombre de sujets, tache/bandes de frequence).
"""
)
