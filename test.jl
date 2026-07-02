using Eegle

files = "C:/Users/coumesa/Documents/BCI/data/FII_BCI_Corpus/MI/Zhou2016/subject_01_sesssion_01.npz"

# Chargez un seul fichier pour tester (sans le rééchantillonner)
o_test = readNY(files[1])

# Affichez tous les champs disponibles dans votre objet EEG
println("Les champs de la structure EEG sont : ", propertynames(o_test))

# En général, dans les packages EEG en Julia, la fréquence s'appelle 'fs' ou 'srate'
# Si c'est 'fs', vous pouvez l'afficher comme ça :
println("La fréquence d'origine est de : ", o_test.fs, " Hz")