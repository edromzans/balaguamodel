import pandas as pd
import numpy as np
from lmfit import Minimizer, Parameters, report_fit
import pickle
import time


# dirInput = '/dados/ProcessoOtimizacaoModelos/' \
#     'calibracaoBalagua/dados/inputs/ugrhi_sp/'
# dirInput = '/media/hd2TB/lcbiag/ProcessoOtimizacaoModelos/' \
#     'calibracaoBalagua/dados/inputs/ugrhi_sp/'
# dirInput = '/vol0/evandro/ProcessoOtimizacaoModelos/' \
#     'calibracaoBalagua/dados/inputs/ugrhi_sp/'
dirInput = '/vol0/evandro/lcbiag/ProcessoOtimizacaoModelos/' \
    'calibracaoBalagua/dados/inputs/ugrhi_sp/'

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
# -------------------------

# CRU
# input_df = pd.read_pickle(dirInput+tagname+'_ugrhi_sp.pkl')
# Xavier
input_df = pd.read_pickle(dirInput+tagname+'_xavier_ugrhi_sp.pkl')


etp = input_df.etp
p2 = input_df.p
q2 = input_df.q

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
    # etp = np.float64(dadosobs['etp'])
    # p2 = np.float64(dadosobs['p2'])
    # q2 = np.float64(dadosobs['q2'])
    # escb = np.float64(dadosobs['escb'])
    #
    m1 = np.float64(500.)  # estimativa de m1 inicial
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
        modeloerro[kount] = np.sqrt(q2[kount]) - np.sqrt(d2)  # Wanderwiele

        # modeloerro[kount] = np.sqrt((q2[kount] - d2)**2.)  # erro quad. med.
        # modeloerro[kount] = q2[kount] - d2
        # print(d2, s2, f2, m2)
        # print(kount, m_func, q2[kount], etp[kount],  d2, modeloerro[kount])
        # print(kount)
        # time.sleep(1)

        m1 = m2

    return modeloerro


"""
Processo de otimizacao de parametros
"""
# params = Parameters()
# params.add('a1', min=0.0, max=1, brute_step=0.03333)
# params.add('a2', min=0.0, max=3, brute_step=0.02)
# params.add('a22', min=0.5, max=2.0, brute_step=0.5)
# params.add('a3', min=0.5e-04, max=6e-04, brute_step=1.83333e-05)

params = Parameters()
params.add('a1', min=0., max=1., brute_step=0.02)
params.add('a2', min=0.0, max=1.8, brute_step=0.018)
params.add('a22', min=0.5, max=2.0, brute_step=0.5)
params.add('a3', min=1e-04, max=10e-04, brute_step=1.8e-05)


"""
Calcula o numero de possibilidades a serem testadas
"""
pa1 = ((params['a1'].max - params['a1'].min) /
       params['a1'].brute_step)
pa2 = ((params['a2'].max - params['a2'].min) /
       params['a2'].brute_step)
pa22 = ((params['a22'].max - params['a22'].min) /
        params['a22'].brute_step)
pa3 = ((params['a3'].max - params['a3'].min) /
       params['a3'].brute_step)

print('-----------> Possibilidades <----------')
print('pa1 ', pa1)
print('pa2 ', pa2)
print('pa22 ', pa22)
print('pa3 ', pa3)
print('Total = ', pa1*pa2*pa22*pa3)

otimiza = Minimizer(residual, params,
                    reduce_fcn=None,
                    calc_covar=True,
                    fcn_args=(etp, p2, q2))

out = otimiza.brute(workers=50)

dirMR = '/vol0/evandro/resultados/'
pickle.dump(out,
            open(dirMR+tagname+'_xavier_ugrhi_bruteMinimizerResult.pkl',
                 'wb'))

# report_fit(out.params)
report_fit(out)
