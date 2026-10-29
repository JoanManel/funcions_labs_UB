import numpy as np
import pandas as pd  
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, least_squares
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
    from math import floor, log10

    if not np.isfinite(valor) or not np.isfinite(incertesa):
        raise ValueError("El valor i la incertesa han de ser finits.")
    if incertesa < 0:
        raise ValueError("La incertesa no pot ser negativa.")
    if valor == 0 and incertesa == 0:
        return r"0"

    if incertesa > 0:
        # Fem servir un exponent d'enginyeria determinat pel valor, no per la
        # incertesa, perquè la mantissa del resultat sigui llegible.
        referencia = abs(valor) if valor != 0 else incertesa
        exp = floor(log10(referencia) / 3) * 3
        mant_valor = valor / 10**exp
        mant_incertesa = incertesa / 10**exp

        exp_incert = floor(log10(abs(mant_incertesa)))
        primera_cifra = int(mant_incertesa / 10**exp_incert)
        xifres = 2 if primera_cifra == 1 else 1
        decimals = -exp_incert + (xifres - 1)
        # Els decimals negatius arrodoneixen a desenes, centenes, etc.
        mant_valor = round(mant_valor, decimals)
        mant_incertesa = round(mant_incertesa, decimals)
        decimals = max(0, decimals)  # El format de text requereix decimals >= 0.
        
        if exp != 0:
            format_str=fr"({mant_valor:.{decimals}f} \pm {mant_incertesa:.{decimals}f}) \times 10^{{{exp}}}"
        else:
            format_str = fr"{mant_valor:.{decimals}f} \pm {mant_incertesa:.{decimals}f}"
        return format_str

    else:
        exp = floor(log10(abs(valor)) / 3) * 3 if valor != 0 else 0
        mant_valor = valor / 10**exp
        format_str = fr"{mant_valor:.3g}"
        if exp != 0:
            format_str=format_str+ fr" \times 10^{{{exp}}}"
        return format_str

def ajusta_dades(x, y, funcio_model, incert_y=None, x_min=None, x_max=None,
                 exclude=None, v_i=None, incert_x=None):
    #aquesta funció ajusta les dades a un cert model. En cas que les dades y tinguin incerteses es fa servir minims quadrats ponderats.
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape != y.shape:
        raise ValueError("x i y han de tenir la mateixa forma.")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("x i y només poden contenir valors finits.")

    if incert_y is not None:
        incert_y = np.asarray(incert_y, dtype=float)
        if incert_y.ndim == 0:
            incert_y = np.full(x.shape, float(incert_y))
        if incert_y.shape != y.shape:
            raise ValueError("incert_y ha de ser escalar o tenir la mateixa forma que y.")
        if not np.all(np.isfinite(incert_y)) or np.any(incert_y <= 0):
            raise ValueError("Totes les incerteses de y han de ser finites i positives.")
    if incert_x is not None:
        incert_x = np.asarray(incert_x, dtype=float)
        if incert_x.ndim == 0:
            incert_x = np.full(x.shape, float(incert_x))
        if incert_x.shape != x.shape:
            raise ValueError("incert_x ha de ser escalar o tenir la mateixa forma que x.")
        if not np.all(np.isfinite(incert_x)) or np.any(incert_x <= 0):
            raise ValueError("Totes les incerteses de x han de ser finites i positives.")
        if incert_y is None:
            raise ValueError(
                "Per considerar incert_x en l'ajust també cal proporcionar incert_y."
            )

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
    incert_x_fit = incert_x[mask] if incert_x is not None else None
    incert_y_fit = incert_y[mask] if incert_y is not None else None
    if len(x_fit) == 0:
        raise ValueError("No queda cap punt per fer l'ajust.")

    # Si hi ha incerteses en tots dos eixos fem una regressió ortogonal:
    # s'ajusten simultàniament els paràmetres i els valors reals de x.
    if incert_x_fit is not None:
        if v_i is None:
            beta0, _ = curve_fit(
                funcio_model, x_fit, y_fit, sigma=incert_y_fit,
                absolute_sigma=True
            )
        else:
            beta0 = np.asarray(v_i, dtype=float)

        n_parametres = len(beta0)
        inicials = np.concatenate((beta0, x_fit))

        def residus_ortogonals(variables):
            parametres = variables[:n_parametres]
            x_reals = variables[n_parametres:]
            return np.concatenate((
                (x_fit - x_reals) / incert_x_fit,
                (y_fit - funcio_model(x_reals, *parametres)) / incert_y_fit
            ))

        ajust_ortogonal = least_squares(residus_ortogonals, inicials)
        if not ajust_ortogonal.success:
            raise RuntimeError(
                "La regressió ortogonal no ha convergit: "
                + ajust_ortogonal.message
            )
        if np.linalg.matrix_rank(ajust_ortogonal.jac) < ajust_ortogonal.jac.shape[1]:
            raise RuntimeError(
                "No s'han pogut estimar les incerteses dels paràmetres: "
                "la regressió ortogonal no està ben determinada."
            )

        popt = ajust_ortogonal.x[:n_parametres]
        x_ajustat = ajust_ortogonal.x[n_parametres:]
        cov_total = np.linalg.pinv(
            ajust_ortogonal.jac.T @ ajust_ortogonal.jac
        )
        pcov = cov_total[:n_parametres, :n_parametres]
        chi2 = 2 * ajust_ortogonal.cost
        metode = "ortogonal"
    else:
        # Amb incerteses de y, aquestes es tracten com desviacions estàndard
        # absolutes. Sense incerteses, curve_fit estima la variància residual.
        absolute_sigma = incert_y_fit is not None
        if v_i is not None:
            popt, pcov = curve_fit(
                funcio_model, x_fit, y_fit, sigma=incert_y_fit,
                absolute_sigma=absolute_sigma, p0=v_i
            )
        else:
            popt, pcov = curve_fit(
                funcio_model, x_fit, y_fit, sigma=incert_y_fit,
                absolute_sigma=absolute_sigma
            )
        x_ajustat = x_fit
        metode = "ponderat_y" if incert_y_fit is not None else "no_ponderat"

    if not np.all(np.isfinite(pcov)) or np.any(np.diag(pcov) < 0):
        raise RuntimeError(
            "No s'han pogut estimar les incerteses dels paràmetres. "
            "Comprova que hi hagi prou punts i que el model estigui ben determinat."
        )

    perrors = np.sqrt(np.diag(pcov))
    residuals = y_fit - funcio_model(x_ajustat, *popt)
    if incert_x_fit is None:
        chi2 = np.sum((residuals / incert_y_fit) ** 2) if incert_y_fit is not None else np.sum(residuals**2)
    dof = len(x_fit) - len(popt)
    if dof <= 0 and incert_x_fit is None and incert_y_fit is None:
        raise ValueError(
            "Calen més punts que paràmetres per estimar les incerteses "
            "d'una regressió sense incerteses experimentals."
        )
    chi2_reduced = chi2 / dof if dof > 0 else np.nan
    rmse = np.sqrt(np.mean(residuals**2))
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y_fit - np.mean(y_fit))**2)
    r2 = 1 - ss_res / ss_tot
    resultats = {
        'popt': popt,
        'perrors': perrors,
        'pcov': pcov,
        'chi2_reduced': chi2_reduced,
        'dof': dof,
        'metode': metode,
        'x_ajustat': x_ajustat,
        'rmse': rmse,
        'R2': r2,
        'mask': mask  # afegim la màscara per fer servir després a la gràfica
    }
    display(Math(funcio_model(1, *popt, option="Text")))
    text = ''
    for i, (v, e) in enumerate(zip(popt, perrors)):
        text += r'a_{{{}}} = {}\\\\ ,\,\, '.format(i, format_resultat(v, e)) 
    text += r'\chi^2_{{\nu}} = {:.2f}\\\\ ,\,\,'.format(chi2_reduced)
    text += r'\textrm{{RMSE}} = {}'.format(format_resultat(rmse, 0))
    text += r'\\\\ ,\,\,R^2 = {}'.format(arrodoneix_r2(r2))
    display(Math(text))
    return resultats

def mostra_dades(x, y, funcio_model=None, incert_x=None, incert_y=None,
                 log_x=False, log_y=False, titol=None,
                 xlabel='x', ylabel='y', forma='o',
                 x_min=None, x_max=None, exclude=None, 
                 mida_figura=(4.5, 4), label_dades="Dades",
                 valors_inicials=None, ponderacio=True,
                 ax=None, color=None):
    """Representa les dades i retorna la parella ``(ax, resultats)``.

    ``resultats`` és el diccionari retornat per ``ajusta_dades`` quan es
    proporciona un model d'ajust, i ``None`` quan només es representen dades.
    """
    
    x = np.asarray(x)
    y = np.asarray(y)
    if x.shape != y.shape:
        raise ValueError("x i y han de tenir la mateixa forma.")
    if incert_x is not None:
        incert_x = np.asarray(incert_x, dtype=float)
        if incert_x.ndim == 0:
            incert_x = np.full(x.shape, float(incert_x))
        if incert_x.shape != x.shape:
            raise ValueError("incert_x ha de ser escalar o tenir la mateixa forma que x.")
    if incert_y is not None:
        incert_y = np.asarray(incert_y, dtype=float)
        if incert_y.ndim == 0:
            incert_y = np.full(y.shape, float(incert_y))
        if incert_y.shape != y.shape:
            raise ValueError("incert_y ha de ser escalar o tenir la mateixa forma que y.")
    ax2=ax
    if ax is None:
            fig, ax = plt.subplots(constrained_layout=True, figsize=mida_figura)

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
                   color='grey', label='Exclosos '+label_dades, alpha=0.5, fmt=forma)
        label_dades = label_dades+ ' usades'

    # Mostra dades utilitzades per a l’ajust
    ax.errorbar(x[mask], y[mask], xerr=incert_x[mask] if incert_x is not None else None,
                yerr=incert_y[mask] if incert_y is not None else None,
                fmt=forma, label=label_dades, capsize=3,color=color)

    if funcio_model is not None:
        if ponderacio:
            resultats = ajusta_dades(x, y, funcio_model,
                                 incert_x=incert_x, incert_y=incert_y,
                                 x_min=x_min, x_max=x_max, exclude=exclude,v_i=valors_inicials)
        else:
            resultats = ajusta_dades(x, y, funcio_model,
                                 x_min=x_min, x_max=x_max, exclude=exclude,v_i=valors_inicials)
        popt = resultats['popt']

       # x_dense = np.linspace(min(x[mask]), max(x[mask]), 500)
        x_dense = np.linspace(min(x), max(x), 500)
        y_fit = funcio_model(x_dense, *popt)
        ax.plot(x_dense, y_fit, '-', label='Ajust '+label_dades ,color=color)
    else:
        resultats=None

    if log_x:
        ax.set_xscale('log')
    if log_y:
        ax.set_yscale('log')

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
       # només si és figura nova
    if titol and ax2 is None:
        ax.set_title(titol)
    if ax2 is None:
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(True)
        plt.show()
    return ax, resultats

def mostra_varies_series(series, mida_figura=(6,4), xlabel="x", ylabel="y", titol=None):
    fig, ax = plt.subplots(constrained_layout=True, figsize=mida_figura)

    for s in series:
        # passem tots els paràmetres directament a mostra_dades
        mostra_dades(ax=ax, **s)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if titol:
        ax.set_title(titol)
    ax.legend()
    ax.grid(True)
    plt.show()
    return ax
    
def model_proporcional(x, a, **kwargs):
    if kwargs.get("option") == "Text":
        return r"y = a_0 x "
    return a * x 

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
