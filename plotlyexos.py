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
trace1 = go.Pie(labels=list(pays_sbvts.keys()), values=list(pays_sbvts.values()))
plot([trace1], filename="PieChartSubv1.html")

randomdata1 = np.random.randn(500)
trace2 = go.Histogram(x=randomdata1)
plot([trace2], filename="HistoSubv1.html")
