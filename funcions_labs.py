import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from IPython.display import display, Math

def llegeix_dades(path, separador='\t', comentari='#', skip=0):
    dades = np.loadtxt(path, delimiter=separador, comments=comentari, skiprows=skip)
    return dades.T if dades.ndim > 1 else dades
def dM(v):
    display(Math(v))

def arrodoneix_r2(r2):
    error = 1 - r2
    if (r2<0) or (error <= 0):
        # R2 és 1 o més gran (pot passar per redondeig), mostrem 2 decimals
        return f"{r2:.2f}"
    decimals = max(0, int(np.ceil(-np.log10(error))))
    return f"{r2:.{decimals}f}"
    
def format_resultat(valor, incertesa):
    from math import log10, floor, isclose

    if isclose(valor, 0) and isclose(incertesa, 0):
        return r"0"

    if incertesa != 0:
        exp = round(log10(abs(incertesa)) / 3) * 3
        mant_valor = valor / 10**exp
        mant_incertesa = incertesa / 10**exp

        exp_incert = floor(log10(abs(mant_incertesa)))
        primera_cifra = int(mant_incertesa / 10**exp_incert)
        xifres = 2 if primera_cifra == 1 else 1
        decimals = max(0, -exp_incert + (xifres - 1))
        
        if not isclose(exp,0):
            format_str=fr"({mant_valor:.{decimals}f} \pm {mant_incertesa:.{decimals}f}) \times 10^{{{exp}}}"
        else:
            format_str = fr"{mant_valor:.{decimals}f} \pm {mant_incertesa:.{decimals}f}"
        return format_str

    else:
        exp = round(log10(abs(valor)) / 3) * 3 if valor != 0 else 0
        mant_valor = valor / 10**exp
        format_str = fr"{mant_valor:.3g}"
        if not isclose(exp,0):
            format_str=format_str+ fr" \times 10^{{{exp}}}"
        format_str=format_str+ r""
        return format_str

def ajusta_dades(x, y, funcio_model, incert_y=None, x_min=None, x_max=None, exclude=None, v_i=None):
    #aquesta funció ajusta les dades a un cert model. En cas que les dades y tinguin incerteses es fa servir minims quadrats ponderats.
    x = np.asarray(x)
    y = np.asarray(y)
    if incert_y is not None:
        incert_y = np.asarray(incert_y)

    # Filtratge del rang
    mask = np.ones_like(x, dtype=bool)
    if x_min is not None:
        mask &= x >= x_min
    if x_max is not None:
        mask &= x <= x_max
    # Exclusió de punts
    if exclude is not None:
        if isinstance(exclude[0], int):
            mask[exclude] = False
        else:
            mask &= ~np.isin(x, exclude)
    x_fit = x[mask]
    y_fit = y[mask]
    incert_y_fit = incert_y[mask] if incert_y is not None else None
    if v_i is not None:
        popt, pcov = curve_fit(funcio_model, x_fit, y_fit, sigma=incert_y_fit, absolute_sigma=True,p0=v_i)
    else:
        popt, pcov = curve_fit(funcio_model, x_fit, y_fit, sigma=incert_y_fit, absolute_sigma=True)
    perrors = np.sqrt(np.diag(pcov))
    residuals = y_fit - funcio_model(x_fit, *popt)
    chi2 = np.sum((residuals / incert_y_fit) ** 2) if incert_y_fit is not None else np.sum(residuals**2)
    dof = len(x_fit) - len(popt)
    chi2_reduced = chi2 / dof if dof > 0 else np.nan
    rmse = np.sqrt(np.mean(residuals**2))
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y_fit - np.mean(y_fit))**2)
    r2 = 1 - ss_res / ss_tot
    resultats = {
        'popt': popt,
        'perrors': perrors,
        'chi2_reduced': chi2_reduced, 
        'rmse': rmse,
        'R2': r2,
        'mask': mask  # afegim la màscara per fer servir després a la gràfica
    }
    dM(funcio_model(1, *popt, option="Text"))
    text = ''
    for i, (v, e) in enumerate(zip(popt, perrors)):
        text += r'a_{{{}}} = {}\quad '.format(i, format_resultat(v, e)) 
    text += r'\\ \chi^2_{{\nu}} = {:.2f}\quad '.format(chi2_reduced)
    text += r'\textrm{{RMSE}} = {}\quad '.format(format_resultat(rmse, 0))
    text += r'R^2 = {}'.format(arrodoneix_r2(r2))
    dM(text)
    return resultats

def mostra_dades(x, y, funcio_model=None, incert_x=None, incert_y=None,
                 log_x=False, log_y=False, titol=None,
                 xlabel='x', ylabel='y', forma='o',
                 x_min=None, x_max=None, exclude=None, 
                 mida_figura=(4.5, 4),label_dades="Dades",
                valors_inicials=None, ponderacio=True):
    x = np.asarray(x)
    y = np.asarray(y)
    incert_x = np.asarray(incert_x) if incert_x is not None else None
    incert_y = np.asarray(incert_y) if incert_y is not None else None

    fig, ax = plt.subplots(constrained_layout=True,figsize=mida_figura)
    mask = np.ones_like(x, dtype=bool)
    if x_min is not None:
        mask &= x >= x_min
    if x_max is not None:
        mask &= x <= x_max
    if exclude is not None:
        if isinstance(exclude[0], int):
            mask[exclude] = False
        else:
            mask &= ~np.isin(x, exclude)

        # Mostra dades excloses (només si n'hi ha)
    if not np.all(mask):
        x_excluded = x[~mask]
        y_excluded = y[~mask]
        incert_x_ex = incert_x[~mask] if incert_x is not None else None
        incert_y_ex = incert_y[~mask] if incert_y is not None else None

        ax.errorbar(x_excluded, y_excluded, xerr=incert_x_ex, yerr=incert_y_ex,
                    fmt='s', color='grey', label='Exclosos', alpha=0.5)
        label_dades = label_dades+ ' usades'

    # Mostra dades utilitzades per a l’ajust
    ax.errorbar(x[mask], y[mask], xerr=incert_x[mask] if incert_x is not None else None,
                yerr=incert_y[mask] if incert_y is not None else None,
                fmt=forma, label=label_dades, capsize=3)

    if funcio_model is not None:
        if ponderacio:
            resultats = ajusta_dades(x, y, funcio_model, incert_y=incert_y,
                                 x_min=x_min, x_max=x_max, exclude=exclude,v_i=valors_inicials)
        else:
            resultats = ajusta_dades(x, y, funcio_model,
                                 x_min=x_min, x_max=x_max, exclude=exclude,v_i=valors_inicials)
        popt = resultats['popt']

       # x_dense = np.linspace(min(x[mask]), max(x[mask]), 500)
        x_dense = np.linspace(min(x), max(x), 500)
        y_fit = funcio_model(x_dense, *popt)
        ax.plot(x_dense, y_fit, '-', label='Ajust')

    if log_x:
        ax.set_xscale('log')
    if log_y:
        ax.set_yscale('log')

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if titol:
        ax.set_title(titol)

    ax.legend()
    ax.grid(True)
    plt.show()


def model_lineal(x, a, b, **kwargs):
    if kwargs.get("option") == "Text":
        return r"y = a_0 x + a_1"
    return a * x + b

def model_exponencial(x, a, b, **kwargs):
    if kwargs.get("option") == "Text":
        return r"y = a_0 e^{a_1 x}"
    return a * np.exp(b * x)

def model_potencial(x, a, b, **kwargs):
    if kwargs.get("option") == "Text":
        return r" y =a_0 x^{a_1}"
    return a * x**b
    

def model_polinomi(x, *coeficients, **kwargs):
    if kwargs.get("option") == "Text":
        termes = r"y  = "
        for i, _ in enumerate(coeficients):
            if i == 0:
                termes=termes+fr"a_{{{i}}}"
            else:
                termes=termes +fr"+ a_{{{i}}} x^{i}"
        return termes
    return sum(c * x**i for i, c in enumerate(coeficients))

def model_gaussià(x, A, x0, sigma, **kwargs):
    if kwargs.get("option") == "Text":
        return r"y= a_0 e^{-\frac{(x - a_1)^2}{2 a_2^2}}"
    return A * np.exp(-((x - x0)**2) / (2 * sigma**2))

def model_gaussià_base(x, A, x0, sigma, a, b, **kwargs):
    if kwargs.get("option") == "Text":
        return r"y = a_0 e^{-\frac{(x - a_1)^2}{2 a_2^2}} + a_3 x + a_4"
    return A * np.exp(-((x - x0)**2) / (2 * sigma**2)) + a * x + b

def model_lorentziana(x, A, x0, gamma, **kwargs):
    if kwargs.get("option") == "Text":
        return r"y = \frac{a_0 a_2^2}{(x - a_1)^2 + a_2^2}"
    return A * gamma**2 / ((x - x0)**2 + gamma**2)
    
def model_sinus(x, a0, a1, a2, **kwargs):
    if kwargs.get("option") == "Text":
        return r"y =a_0 \sin( a_1 x + a_2)"
    return a0 *np.sin( a1*x + a2)

