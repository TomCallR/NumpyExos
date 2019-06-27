import numpy as np

import plotly.graph_objs as go
import plotly.figure_factory as ff
from plotly.offline import download_plotlyjs, plot

##########################################
donnees = np.genfromtxt("donnees/Numpy_exercice.csv",
                        delimiter=";",
                        usecols=np.arange(0, 18),
                        encoding='UTF-8',
                        dtype=str,
                        skip_header=1)
mask = [(x == 0) | (x >= 11) for x in range(18)]
nouvellesdonnees = donnees[:, mask]
pays_index = np.array(nouvellesdonnees[:, 2])

pays_sbvts = {}
data6000 = nouvellesdonnees[:6000, :]
print(data6000.shape)
for pays in np.unique(pays_index):
    pays_sbvts[pays] = data6000[data6000[:, 2] == pays, -2].astype(np.float).sum()

##########################################
# trace1 = go.Pie(labels=list(pays_sbvts.keys()), values=list(pays_sbvts.values()))
# plot([trace1], filename="PieChartSubv1.html")
#
# randomdata2 = np.random.randn(500)
# trace2 = go.Histogram(x=randomdata2)
# plot([trace2], filename="HistoSubv1.html")

size3 = 100000
# randomdata3 = np.zeros(3 * size3)
# randomdata3[0::3] = (np.random.rand(size3) * 2) - float(size3)
# randomdata3[1::3] = (np.random.rand(size3) * 2) - float(size3)
# randomdata3[2::3] = np.multiply((np.square(randomdata3[0::3]) + np.square(randomdata3[1::3])),
#                                 (np.exp(1 / (randomdata3[0::3] + randomdata3[1::3]))))
# randomdata3 = np.array(list(zip(randomdata3[0::3], randomdata3[1::3], randomdata3[2::3])))
x3 = ((np.random.rand(size3) * 2) - 1) * 30
y3 = ((np.random.rand(size3) * 2) - 1) * 30
z3 = np.multiply((np.square(x3) + np.square(y3)), (np.log2(np.abs(1 / (x3 + y3)))))
# randomdata3 = np.array(list(zip(randomdata3[0::3], randomdata3[1::3], randomdata3[2::3])))
trace3 = go.Surface(x=x3, y=y3, z=z3)
layout3 = go.Layout(title='Test surface', autosize=True)
# fig3 = go.Figure(data=trace3, layout=layout3)
plot([trace3], filename="Surface1.html")


