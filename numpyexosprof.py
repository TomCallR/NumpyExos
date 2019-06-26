import numpy as np
import re

vecteur = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
# vecteur = np.arange(10)\n
# vecteur = np.array(range(10))\n
# vecteur = np.ndarray(10)\n
print(vecteur)
print(np.arange(0, 10, 2))
# np.array([2 * x for x in range(10) if 2 * x != 10])
print(vecteur[2])
print(vecteur[1:6])
print(vecteur[-2:])
print(vecteur[8:])
print(vecteur[len(vecteur) - 2:])
# matrice = np.array([[1, 2, 3], [4, 5, 6]])\n
# matrice = np.ndarray([2, 3])\n
# matrice = np.array([vecteur, vecteur])\n
matrice = np.array(range(12)).reshape(3, 4)
print(matrice)
print(matrice.size)
donnees_index = np.genfromtxt("donnees/Numpy_Exercice.csv",
                              delimiter=";",
                              usecols=np.arange(0, 18),
                              encoding='UTF-8',
                              dtype=str,
                              skip_header=1)
donnees_columns = np.genfromtxt("donnees/Numpy_Exercice.csv",
                                delimiter=";",
                                usecols=np.arange(0, 18),
                                encoding='UTF-8',
                                dtype=None,
                                names=True)
print(type(donnees_index[0][0]))
print(donnees_columns)
print(donnees_columns["pays_lib"])
print(donnees_index.size)
print(donnees_columns.size)
mask = [(x == 0) | (x >= 11) for x in range(18)]
nouvellesdonnees_index = donnees_index[:, mask]
# donnees_index[:,[0] + [x for x in range(11, 18)]]
# np.delete(donnees_index, np.s_[1, 11])

nouvellesdonnees_columns = donnees_columns[list(donnees_columns.dtype.names[:1]
                                                + donnees_columns.dtype.names[11:])]
print(nouvellesdonnees_columns)
premierExtraitDonnees_index = nouvellesdonnees_index[:5, ]
premierExtraitDonnees_columns = nouvellesdonnees_columns[:5]
deuxiemeValeur_index = premierExtraitDonnees_index[0, 1]
deuxiemeValeur_columns = premierExtraitDonnees_columns[0][1]
pays_index = np.array(nouvellesdonnees_index[:, 2])
pays_columns = nouvellesdonnees_columns["pays_lib"]

nb_suisse_columns = (pays_columns == "Suisse").sum()
# sum(1 for p in pays_columns if p == "Suisse")
unique, counts = np.unique(pays_columns, return_counts=True)
data = dict(zip(unique, counts))
print(data['Suisse'])
# pays_names, pays_count = np.unique(pays_columns, return_counts=True)
# for p, c in zip(pays_names, pays_count):
#     print(p, c)
from collections import Counter

pays_count = Counter(pays_columns)
print(dict(pays_count))

# compteur = 0
# for elt in nouvellesdonnees_index[0:,-6]:
#     if elt =="Suisse":
#         print(nouvellesdonnees_index[compteur])
#     compteur +=1
# mask2 = (nouvellesdonnees_index[:, 2] == "Suisse")
# donneesSuisse = nouvellesdonnees_index[mask2, :]
# print(donneesSuisse)
# print(nouvellesdonnees_index[np.where((nouvellesdonnees_index[:, 2] == 'Suisse'))])
print(nouvellesdonnees_columns[nouvellesdonnees_columns["pays_lib"] == "Suisse"])

# mask2 = ((nouvellesdonnees_index[:, 2] == "Suisse") & (nouvellesdonnees_index[:, 3] == "Recherche"))
# donneesSuisse = nouvellesdonnees_index[mask2, :]
# print(donneesSuisse)
# print(nouvellesdonnees_index[np.where((nouvellesdonnees_index[:, 2] == 'Suisse')
#                                       & (nouvellesdonnees_index[:, 3] == 'Recherche') & (
#                                                   nouvellesdonnees_index[:, 4] == 'Recherche'))])
print(nouvellesdonnees_columns[(nouvellesdonnees_columns["pays_lib"] == "Suisse")
                               & (nouvellesdonnees_columns["org_lib"] == "Recherche")])

# mask2 = ((nouvellesdonnees_index[:, 2] == "Suisse") | (nouvellesdonnees_index[:, 2] == "France"))
# donneesSuisse = nouvellesdonnees_index[mask2, :]
# print(donneesSuisse)
# print(nouvellesdonnees_index[np.where((nouvellesdonnees_index[:, 2] == 'Suisse')
#       | (nouvellesdonnees_index[:, 2] == 'France'))])
print(nouvellesdonnees_columns[(nouvellesdonnees_columns["pays_lib"] == "Suisse")
                               | (nouvellesdonnees_columns["pays_lib"] == "France")])

# gh = donnees_index[:,13]# gh [(gh == "Ens. supérieur")] = "Enseignement supérieur"
# mask5 = (nouvellesdonnees_index[:, 3] == "Ens. supérieur" )
# donneesEnsSup = nouvellesdonnees_index[mask5, :]
# print(donneesEnsSup.shape[0])
# donneesEnsSup[donneesEnsSup == "Ens. supérieur"] = "Enseignement supérieur"
# print(donneesEnsSup)

ensSup = nouvellesdonnees_index[np.where((nouvellesdonnees_index[:, 3] == 'Ens. supérieur'))]
repEnsSup = np.char.replace(ensSup, 'Ens.', 'Enseignement')
print(repEnsSup[3])
print(len(repEnsSup))
# /!\\ Mauvais usage /!\\
# print(len(nouvellesdonnees_columns[nouvellesdonnees_columns["org_code"] == "Ens. supérieur"]))
# nouvellesdonnees_columns["org_code"] = np.core.defchararray.replace(
# nouvellesdonnees_columns["org_code"], "Ens. supérieur", "Enseignement superieur")
# print(nouvellesdonnees_columns)
# paysvides = ((nouvellesdonnees_index[np.where((nouvellesdonnees_index[:, 2] == ''))]))
# print(len(paysvides))
# mask6 = (nouvellesdonnees_index[:, 2] == "" )
# donneesVide = nouvellesdonnees_index[mask6, :]
# print(donneesVide.shape[0])

mask6 = (nouvellesdonnees_index[:, 2] == "")
if mask6.sum() == 0:
    print("Aucun pays vide")
# print(np.any(nouvellesdonnees_columns["pays_lib"] == ""))
tableau_index = nouvellesdonnees_index[:1000,-2]
# tableau_columns = nouvellesdonnees_columns["montant_subvention"][:1000]
# print(tableau_index.dtype)
# print(tableau_columns.dtype)
tableau_index = tableau_index.astype(np.float)
# tableau_columns = tableau_columns.astype(np.float)
# print(tableau_index.dtype)
# print(tableau_columns.dtype)
# print(tableau_index.sum())
# print(np.sum(tableau_columns))
# print(np.around(tableau_index.mean(), 3))
# print("{:.3f}".format(np.average(tableau_columns)))
# print(tableau_index.max())print(max(tableau_columns))

pmax_montant_subvention = max(tableau_index)
# popo = np.argmax(tableau_index)
# paysTop = nouvellesdonnees_index[popo][2]
# print(pmax_montant_subvention, popo, paysTop)
# index2 = -1
# for i in donnees_index[:1000,[16]].astype(float):
#     index2 += 1
#     if i == pmax_montant_subvention:
#         print(pays_index[index2])
print(nouvellesdonnees_columns[nouvellesdonnees_columns["montant_subvention"] == str(pmax_montant_subvention)][0][2])

# for entry in data:
#     entry[0] = re.findall(r"", entry[0])[0]
fr_top_5000_index = nouvellesdonnees_index[:5000,:]
fr_top_5000_columns = nouvellesdonnees_columns[:5000]

fr_top_5000_index = fr_top_5000_index[np.where(fr_top_5000_index[:, 2] == "France")]
fr_top_5000_columns = fr_top_5000_columns[fr_top_5000_columns["pays_lib"] == "France"]
print(fr_top_5000_index[:, -2].astype(np.float).sum())
print(fr_top_5000_columns["montant_subvention"].astype(np.float).sum())
print(fr_top_5000_index[:, -2].astype(np.float).mean())
print(fr_top_5000_columns["montant_subvention"].astype(np.float).mean())
print(fr_top_5000_index[:, -2].astype(np.float).max())
print(fr_top_5000_columns["montant_subvention"].astype(np.float).max())
for appel in fr_top_5000_index[:, 0]:
    print(appel[-4:])
#     print(appel.split('-')[-1])
#     import re
regex1 = r"-(\\d{4})"
#     regex2 = r"\\b(20\\d\\d)"
date_col = [(re.findall(regex1, appel) + ["xxxx"])[0]]
#     for appel in fr_top_5000_index[:,0]
#     fr_top_5000_index = np.insert(fr_top_5000_index, 8, date_col, axis=1)
pays_sbvts = {}
data_6000 = nouvellesdonnees_columns[:6000]
for pays in np.unique(pays_index):
    pays_sbvts[pays] = data_6000[data_6000["pays_lib"] == pays]["montant_subvention"].astype(np.float).sum()
#     print(pays_sbvts)
#     valeur_max = 0
#     pays_max = ""
#     for p in pays_sbvts:
#     if pays_sbvts[p] > valeur_max:
#     pays_max = p
#     valeur_max = pays_sbvts[p]
#     print(pays_max)
print(sorted(list(pays_sbvts.items()), key=lambda p: -p[1])[0][0])
