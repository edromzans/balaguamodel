import pandas as pd
import numpy as np
from lmfit import report_fit, fit_report
# from example_brute import plot_results_brute
# import matplotlib.pyplot as plt
import pickle
from gradeparamsbrute import plot_results_brute


# UGRHI SP ##
# tagname = '58220000'
# -------------------------
# tagname = '3D-001'
# -------------------------
# tagname = '4C-007'
# -------------------------
# tagname = '4B-015'
# -------------------------
tagname = '5B-011'
# -------------------------
############

# diresults = '/vol0/evandro/resultados/'
# diresults = '/home/evandro/lcbiag/ProcessoOtimizacaoModelos/resultados/'
# diresults = '/dados/ProcessoOtimizacaoModelos/calibracaoBalagua/resultados/'

diresults = '/vol0/evandro/lcbiag/' \
    'ProcessoOtimizacaoModelos/calibracaoBalagua/resultados/'
dirInput = '/vol0/evandro/lcbiag/ProcessoOtimizacaoModelos/' \
    'calibracaoBalagua/dados/inputs/ugrhi_sp/'

# CRU
# arqvminres = tagname+'_ugrhi_bruteMinimizerResult.pkl'
# pngfigplot = '/vol0/evandro/lcbiag/' \
#     'ProcessoOtimizacaoModelos/calibracaoBalagua/' \
#     'plots/'+tagname+'_ugrhi_pltGridParams.png'
# input_df = pd.read_pickle(dirInput+tagname+'_ugrhi_sp.pkl')

# Xavier
arqvminres = tagname+'_xavier_ugrhi_bruteMinimizerResult.pkl'
pngfigplot = '/vol0/evandro/lcbiag/' \
    'ProcessoOtimizacaoModelos/calibracaoBalagua/' \
    'plots/'+tagname+'_xavier_ugrhi_pltGridParams.png'
input_df = pd.read_pickle(dirInput+tagname+'_xavier_ugrhi_sp.pkl')

# pngfigplot = '/dados/ProcessoOtimizacaoModelos/' \
#     'calibracaoBalagua/plots/plotaBruteParams.png'

# pngfigplot = '/home/evandro/lcbiag/' \
#     'ProcessoOtimizacaoModelos/calibracaoBalagua/' \
#     'plots/'+tagname+'_pltGridParams.png'

# pngfigplot = '/dados/' \
#     'ProcessoOtimizacaoModelos/calibracaoBalagua/' \
#     'plots/'+tagname+'_pltGridParams.png'

out = pickle.load(open(diresults+arqvminres, "rb"))

"""
Verificacao dos parametros candidatos caculados
"""

# dirInput = '/vol0/evandro/ProcessoOtimizacaoModelos/' \
#     'calibracaoBalagua/dados/inputs/ugrhi_sp/'
# dirInput = '/home/evandro/lcbiag/ProcessoOtimizacaoModelos/' \
#     'calibracaoBalagua/dados/inputs/ugrhi_sp/'
# dirInput = '/dados/ProcessoOtimizacaoModelos/' \
#     'calibracaoBalagua/dados/inputs/ugrhi_sp/'




etp = input_df.etp
p2 = input_df.p
q2 = input_df.q

posval = np.asarray(~np.isnan(etp) &
                    ~np.isnan(p2) &
                    ~np.isnan(q2)).nonzero()
posval = posval[0]

etp = etp[posval]
p2 = p2[posval]
q2 = q2[posval]

# arqinput = 'input.txt'
# entradados = dirdados+arqinput

ncand = len(out.candidates)

for ck in range(ncand):
    a1 = out.candidates[ck].params['a1']
    a2 = out.candidates[ck].params['a2']
    a22 = out.candidates[ck].params['a22']
    a3 = out.candidates[ck].params['a3']
    print(out.show_candidates(ck+1))
    # dadosobs = pd.read_table(
    #     entradados, header=None, delim_whitespace=True, names=[
    #         'ano', 'mes', 'dia', 'hora', 'minuto', 'segundo',
    #         'etp', 'p2', 'q2', 'escb'])
    m_func = len(etp)
    modeloerro = np.zeros(m_func, dtype='float64')
    #
    ts_mt = np.zeros(m_func, dtype='float64')
    ts_dt = np.zeros(m_func, dtype='float64')
    ts_u = np.zeros(m_func, dtype='float64')
    #
    # etp = np.float64(dadosobs['etp'])
    # p2 = np.float64(dadosobs['p2'])
    # q2 = np.float64(dadosobs['q2'])
    # escb = np.float64(dadosobs['escb'])
    #
    m1 = 500.
    r2 = np.float64(0)
    s2 = np.float64(0)
    n2 = np.float64(0)
    d2 = np.float64(0)
    m2 = np.float64(0)
    for kount in range(0, m_func):
        """
        Modelo de balanco de agua

        r2 => evapotranpiracao real
        s2 => escoamento lento
        n2 => precipitacao ativa
        f2 => escoamento rapido
        d2 => vazao com filtro
        """
        r2 = min(
            etp[kount]*(1.-a1**((p2[kount]+max(m1, 0.))/etp[kount])),
            (p2[kount]+max(m1, 0.))
        )
        s2 = a2*(max(m1, 0.)**a22)
        n2 = p2[kount]-etp[kount]*(1-np.exp(-p2[kount]/etp[kount]))
        f2 = a3*max(m1, 0.)*n2
        d2 = s2+f2
        m2 = m1 + p2[kount] - r2 - d2
        # print(d2, s2, f2, m2)
        ts_mt[kount] = m2
        ts_dt[kount] = d2
        ts_u[kount] = (np.sqrt(q2[kount]) - np.sqrt(d2))
        m1 = m2
        # print(kount)
    print('Media mt ---------------> ', np.average(ts_mt))
    print('Media u ----------------> ', np.average(ts_u),
          np.average(out.residual))

report_fit(out)

"""
Plota a grade de parametros
"""

plot_results_brute(out, best_vals=True, varlabels=None,
                   output=pngfigplot)

# plot_results_brute(out, best_vals=True, varlabels=None, output=None)
