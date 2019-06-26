import numpy as np
import re

# vecteur = np.array(range(10))
# print(vecteur)
#
# vecteur2 = np.array(range(0, 10, 2))
# print(vecteur2)
#
# print("3e val = " + str(vecteur2[2]))
#
# print("val de 2 à 6 = ", end="")
# print(vecteur2[1:4])
#
# print("2 dern val = ", end="")
# nbre = len(vecteur2)
# print(vecteur2[nbre:(nbre-3):-1])
# print(vecteur2[(nbre-2):])
# print(vecteur2[-2:])
#
# matrice = np.array([(0, 0), (0, 0)])
# print(matrice)
# matrice = np.array(range(12)).reshape(3,4)
# print(matrice)

# matrice2 = np.matrix("1 2;3 4")
# print(matrice2)
# print(type(matrice2[0,0]))

# print("taille = " + str(matrice.size))

# Q16
# donnees = np.genfromtxt("donnees/Numpy_Exercice.csv", delimiter=";",  usecols=np.arange(0,18), encoding='UTF-8')

# Q17 Q18
# print("type de nan = " + str(type(np.NaN)))

# Q19
# matrice3 = matrice.astype(str)
# print(matrice3)
# matrice3 = np.array(matrice, dtype=float)
# print(matrice3)

# donnees = np.genfromtxt("donnees/Numpy_exercice.csv",
#                         delimiter=";",
#                         usecols=np.arange(0, 18),
#                         encoding='UTF-8',
#                         dtype=str)

# Q20
# print("taille de donnees = " + str(donnees.size))
# print("shape = " + str(donnees.shape[1]))

# print(donnees[0:4])

# Q21 Q22
# donnees = np.genfromtxt("donnees/Numpy_Exercice.csv", delimiter=";",
#                         encoding='UTF-8',
#                         skip_header=1, dtype=str,
#                         usecols=(0, 11, 12, 13, 14, 15, 16, 17))

# print(donnees[0:4])

# Q22
donnees = np.genfromtxt("donnees/Numpy_exercice.csv",
                        delimiter=";",
                        invalid_raise=False,
                        filling_values="",
                        usecols=np.arange(0, 18),
                        encoding='UTF-8',
                        skip_header=1, dtype=str)
mask = [(x == 0 or x >= 11) for x in range(18)]
nouvellesDonnees = donnees[:, mask]

# Q23
# premierExtraitDonnees = nouvellesDonnees[0:5]
# print(premierExtraitDonnees)

# Q24
# deuxiemeValeur = premierExtraitDonnees[0, 1]
# print(deuxiemeValeur)

# Q25 Q27
# pays, comptage = np.unique(nouvellesDonnees[:, 2], return_counts=True)
# print(pays)
# print(comptage)

# Q26
# mask1 = (pays == "Suisse")
# comptage1 = comptage[mask1]
# print(comptage1)

# Q28
# mask2 = (nouvellesDonnees[:, 2] == "Suisse")
# donneesSuisse = nouvellesDonnees[mask2, :]
# print(donneesSuisse)

# Q29
# mask3 = ((nouvellesDonnees[:, 2] == "Suisse") & ("Recherche" in nouvellesDonnees[:, -4]))
# print(nouvellesDonnees[mask3])

# Q30
# mask4 = ((nouvellesDonnees[:, 2] == "Suisse") |
#          (nouvellesDonnees[:, 2] == "France"))
# print(nouvellesDonnees[mask4])

# Q31
# Comptez le nombre d’enregistrement ayant le org_code ‘Ens. supérieur’
# puis remplacez tous les intitulés ‘Ens. supérieur’ par ‘Enseignement superieur’.
# mask5 = (nouvellesDonnees[:, -5] == "Ens. supérieur")
# nbenssup = nouvellesDonnees[mask5].shape[0]
# print(nbenssup)
# nouvellesDonnees[mask5, -5] = "Enseignement superieur"
# print(nouvellesDonnees[0:10, 0:-4])

# Q32
# mask6 = (nouvellesDonnees[:, 2] == "")
# if mask6.sum() == 0:
#     print("Aucun pays vide")

# Q33
# milleSubv = nouvellesDonnees[0:1000, -2]
# print(milleSubv)

# Q34
# print("type du tableau = " + str(nouvellesDonnees.dtype))

# Q35
# milleSubvFloat = milleSubv.astype(np.float)
# print(milleSubvFloat)

# Q36
# print("Somme des 1000 1eres subv = " + str(milleSubvFloat.sum()))

# Q37
# moyenneKSubv = milleSubvFloat.mean()
# print("Moyenne des 1000 1eres subv = " + str(moyenneKSubv))
# print("Avec 3 chiffres après la vrigule = " + "{:9.3f}".format(moyenneKSubv))

# Q38
# maxKSubv = milleSubvFloat.max()
# print("Max des 1000 1eres subv = " + str(maxKSubv))

# Q39
# maxKSubvStr = (milleSubv[np.abs((milleSubvFloat - maxKSubv)) <= 1e-7])[0]
# maskMaxKSubv = (nouvellesDonnees[0:1000, -2] == maxKSubvStr)
# paysMaxKSubv = (nouvellesDonnees[0:1000, :])[maskMaxKSubv, 2][0]
# print("Pays ayant cette subvention = " + paysMaxKSubv)

# Q40

# Q41
donnees5000Lignes = nouvellesDonnees[0:5000, :]
maskFrance = (donnees5000Lignes[:, 1] == "FR")
subvFrance = donnees5000Lignes[maskFrance, -2].astype(float)
sommeFr = np.sum(subvFrance)
moyenneFr = np.average(subvFrance)
maxFr = np.amax(subvFrance)
print("Somme, moyenne et max des subv Fr dans les 5000 1eres lignes =")
print("{0:9.2f} {1:9.2f} {2:9.2f}".format(sommeFr, moyenneFr, maxFr))

# Q42

# Q43
# Penser à utiliser \b qui dénote chaque changement de mot ou de séparateur
# site très utile pour tester : https://regex101.com/
# regex_year = re.compile("-20\d\d")
regex_year = re.compile(r"\b(20\d\d)")
years = []
for cours in nouvellesDonnees[:, 0]:
    # temp = [x[1:] for x in regex_year.findall(cours)]
    temp = regex_year.findall(cours)
    if len(temp) == 0:
        years.append("")
    elif len(temp) == 1:
        years.append(temp[0])
    else:
        years.append(tuple(temp))
col1modif = np.array(years)
print(col1modif[10575:])

# Q44
# on ne prend que les 6000 premieres lignes pour éviter les bugs dans les subventions
# limit = 10000
limit = nouvellesDonnees.shape[0]
pays = np.unique(nouvellesDonnees[:, 2])
dictPays = dict(zip(pays, np.zeros(pays.size)))
for elem in nouvellesDonnees[0:limit]:
    try:
        dictPays[elem[2]] += float(elem[-2])
    except ValueError:
        pass
print(dictPays)

# Q45
maxPays = ""
maxSubv = 0.0
for key, value in dictPays.items():
    if maxSubv < value:
        maxPays = key
        maxSubv = value
print("Pays avec le plus de subventions = " + maxPays + " : " + str(maxSubv))
