import pandas as pd
import numpy as np
from lmfit import Minimizer, Parameters, report_fit
from lmfit.models import GaussianModel
import matplotlib.pyplot as plt
import pickle

# from matplotlib import rcParams
# rcParams['font.family'] = 'sans-serif'
# rcParams['font.sans-serif'] = ['Tahoma']

# UGRHI SP
# tagname = '58220000'
# -------------------------
# tagname = '3D-001'
# -------------------------
# tagname = '4C-007'
# -------------------------
# tagname = '4B-015'
# -------------------------
tagname = '5B-011'

dirInput = '/dados/ProcessoOtimizacaoModelos/' \
    'calibracaoBalagua/dados/inputs/ugrhi_sp/'
# dirInput = '/media/hd2TB/lcbiag/ProcessoOtimizacaoModelos/' \
#     'calibracaoBalagua/dados/inputs/ugrhi_sp/'
# dirInput = '/vol0/evandro/ProcessoOtimizacaoModelos/' \
#     'calibracaoBalagua/dados/inputs/ugrhi_sp/'
# dirInput = '/home/evandro/lcbiag/ProcessoOtimizacaoModelos/' \
#     'calibracaoBalagua/dados/inputs/ugrhi_sp/'

dirR = '/dados/ProcessoOtimizacaoModelos/calibracaoBalagua/resultados/'
arqvminres = tagname+'_ugrhi_bruteMinimizerResult.pkl'



# dirplot = '/dados/ProcessoOtimizacaoModelos/calibracaoBalagua/plots/'
# pngfigplot = dirplot+tagname+'_pltTsBalagua.png'

input_df = pd.read_pickle(dirInput+tagname+'_ugrhi_sp.pkl')

etp = np.asarray(input_df.etp)
p2 = np.asarray(input_df.p)
q2 = np.asarray(input_df.q)

# IDL where
posval = np.asarray(~np.isnan(etp) &
                    ~np.isnan(p2) &
                    ~np.isnan(q2)).nonzero()
posval = posval[0]

etp = etp[posval]
p2 = p2[posval]
q2 = q2[posval]


def residual(params, etp, p2, q2):
    a1 = params['a1']
    a2 = params['a2']
    a22 = params['a22']
    a3 = params['a3']
    #
    m_func = len(etp)
    modeloerro = np.zeros(m_func, dtype='float64')
    #
    m1 = np.float64(500.)  # estimativa de m1 inicial
    r2 = np.float64(0.)
    s2 = np.float64(0.)
    n2 = np.float64(0.)
    d2 = np.float64(0.)
    m2 = np.float64(0.)
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
        modeloerro[kount] = np.sqrt(q2[kount]) - np.sqrt(d2)  # Wanderwiele

        m1 = m2

    return modeloerro


outbrute = pickle.load(open(dirR +
                            tagname+'_ugrhi_bruteMinimizerResult.pkl', "rb"))

a1_bruteMR = outbrute.params['a1']
a2_bruteMR = outbrute.params['a2']
a22_bruteMR = outbrute.params['a22']
a3_bruteMR = outbrute.params['a3']

print(a1_bruteMR, a2_bruteMR, a22_bruteMR, a3_bruteMR,
      '<-----Resultado MR brute')

# Leastsq com resultado da grade de parametros - metodo bruto
params = Parameters()
params.add('a1', value=a1_bruteMR, min=0., max=1.)
params.add('a2', value=a2_bruteMR)
params.add('a22', value=a22_bruteMR, vary=False)
params.add('a3', value=a3_bruteMR)

otimiza = Minimizer(residual, params,
                    reduce_fcn=None,
                    calc_covar=True,
                    fcn_args=(etp, p2, q2))

# out_leastsq = otimiza.leastsq()
out_leastsq = otimiza.minimize(method='leastsq')  # Levenberg-Marquardt

# report_fit(out_leastsq.params)
report_fit(out_leastsq)

"""
Verificacao dos parametros caculados
"""
a1 = out_leastsq.params['a1']
a2 = out_leastsq.params['a2']
a22 = out_leastsq.params['a22']
a3 = out_leastsq.params['a3']

input_df = pd.read_pickle(dirInput+tagname+'_ugrhi_sp.pkl')

etp = np.asarray(input_df.etp)
p2 = np.asarray(input_df.p)
q2 = np.asarray(input_df.q)

datatempo = input_df.index

nxts = len(input_df)

ts_mt = np.empty(nxts)  # np.zeros(nxts, dtype='float64')  # np.empty(nxts)
ts_dt = np.empty(nxts)  # np.zeros(nxts, dtype='float64')
ts_u = np.empty(nxts)  # np.zeros(nxts, dtype='float64')
ts_Dm = np.empty(nxts)  # np.zeros(nxts, dtype='float64')
ts_r = np.empty(nxts)  # np.zeros(nxts, dtype='float64')
ts_mt.fill(np.nan)
ts_dt.fill(np.nan)
ts_u.fill(np.nan)
ts_Dm.fill(np.nan)
ts_r.fill(np.nan)

m1 = np.float64(500.)
r2 = np.float64(0.)
s2 = np.float64(0.)
n2 = np.float64(0.)
d2 = np.float64(0.)
m2 = np.float64(0.)

m_func = len(posval)
for kount in range(0, m_func):
    """
    Modelo de balanço de agua

    r2 => evapotranpiracao real
    s2 => escoamento lento
    n2 => precipitacao ativa
    f2 => escoamento rapido
    d2 => vazao com filtro
    """
    posck = posval[kount]
    r2 = min(
        etp[posck]*(1.-a1**((p2[posck]+max(m1, 0.))/etp[posck])),
        (p2[posck]+max(m1, 0.))
    )
    s2 = a2*(max(m1, 0.)**a22)
    n2 = p2[posck]-etp[posck]*(1-np.exp(-p2[posck]/etp[posck]))
    f2 = a3*max(m1, 0.)*n2
    d2 = s2+f2
    m2 = m1 + p2[posck] - r2 - d2
    # print(d2, s2, f2, m2)
    ts_mt[posck] = m2
    ts_dt[posck] = d2
    ts_u[posck] = (np.sqrt(q2[posck]) - np.sqrt(d2))
    ts_Dm[posck] = m2 - m1
    ts_r[posck] = r2

    m1 = m2
    # print(s2)

print('Media S---------------> ', np.nanmean(ts_mt))
print('Media u---------------> ', np.nanmean(ts_u))

'''
Saidas
'''

x_eixo = datatempo

# histograma de u
minbin = -5
maxbin = 5
nx = 30.
step = (maxbin - minbin) / nx

binarr = np.arange(nx+1)*step + minbin
hist_u, edges_u = np.histogram(ts_u, bins=binarr, density='True')

xhist = np.arange(nx)*step + minbin + step

# ajuste gaussiano
gmodel = GaussianModel()

print('parameter names: {}'.format(gmodel.param_names))
print('independent variables: {}'.format(gmodel.independent_vars))

gparams = gmodel.make_params()

gresult = gmodel.fit(hist_u, gparams, x=xhist)

print(gresult.fit_report())

# xi = x_eixo[posval[0]]
# xf = x_eixo[posval[-1]]

# Salva resultados para plots
# dirR = '/home/evandro/lcbiag/ProcessoOtimizacaoModelos/resultados/'
dirR = '/dados/ProcessoOtimizacaoModelos/calibracaoBalagua/resultados/'

pickle.dump((out_leastsq,
             x_eixo,
             ts_mt,
             ts_dt,
             ts_u,
             ts_Dm,
             ts_r,
             q2,
             p2,
             etp,
             xhist,
             hist_u,
             gresult),
            open(dirR+tagname+'_ugrhi_leastsqMinimizerResult.pkl', 'wb'))
