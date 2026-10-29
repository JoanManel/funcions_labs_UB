# funcions_labs_UB
Aquesta llibreria conté funcions de suport en els laboratoris docents del Grau de Física de la Universitat de Barcelona. Inclou eines per a la lectura de dades, ajustos de corbes, representació gràfica i models matemàtics habituals.

## Versió per als estudiants

- [Documentació de la llibreria](Documentacio_funcions_labs.pdf)
- [Quadern d'exemples actualitzat](Exemples_f_labs.ipynb) i [versió PDF](Exemples_f_labs.pdf)
- [Dades d'exemple](Dades_Exemple): cal conservar aquesta carpeta al costat del quadern.

Dependències: numpy, pandas, matplotlib, scipy i IPython.

`mostra_dades` retorna `(ax, resultats)`; sense model, `resultats` és `None`. Admet un eix existent amb `ax` i un color amb `color`.
`mostra_varies_series(series, mida_figura=(6,4), xlabel="x", ylabel="y", titol=None)` representa diverses sèries i retorna l'eix.

El quadern vigent és `Exemples_f_labs.ipynb` (amb E majúscula). Es conserven el quadern antic amb e minúscula i les dades de l'arrel per compatibilitat.

# Funcions principals

## `llegeix_dades`

Llegeix un fitxer de dades delimitat per tabuladors, comes o altres separadors.

```python
llegeix_dades(path, separador='\t', comentari='#', skip=0)
```

**Retorna:** Array o tupla d'arrays amb les columnes de dades.

---

## `format_resultat`

Formata un valor amb incertesa en notació científica en format LaTeX.

```python
format_resultat(valor, incertesa)
```

---

## `ajusta_dades`

Fa un ajust als punts de dades proporcionats segons un model donat. Està pensada per ser cridada des de la funció `mostra_dades`, encara que també es pot usar de manera independent.

```python
ajusta_dades(x, y, funcio_model, incert_y=None, x_min=None, x_max=None, 
             exclude=None, v_i=None, incert_x=None, exclude_x=None)
```

**Retorna:** Diccionari amb paràmetres ajustats, errors, χ², RMSE, R² i màscara de punts usats.

---

## `mostra_dades`

Representa gràficament les dades (amb error) i opcionalment hi superposa un ajust.

```python
mostra_dades(x, y, funcio_model=None, incert_x=None, incert_y=None,
             log_x=False, log_y=False, titol=None,
             xlabel='x', ylabel='y', forma='o',
             x_min=None, x_max=None, exclude=None, 
             mida_figura=(4.5, 4), label_dades="Dades",
             valors_inicials=None, ponderacio=True, ax=None, color=None, exclude_x=None)
```

# Models matemàtics disponibles

Tots aquests models accepten un argument opcional `option="Text"` que retorna una cadena LaTeX del model:

* `model_lineal(x, a, b)`: y = a · x + b
* `model_exponencial(x, a, b)`: y = a · exp(b · x)
* `model_potencial(x, a, b)`: y = a · x^b
* `model_polinomi(x, *coeficients)`: y = a₀ + a₁x + a₂x² + ...
* `model_gaussià(x, A, x0, sigma)`: y = A · exp(-(x - x0)² / (2·sigma²))
* `model_gaussià_base(x, A, x0, sigma, a, b)`: Gaussiana + fons lineal
* `model_lorentziana(x, A, x0, gamma)`: Lorentziana
* `model_sinus(x, a0, a1, a2)`: y = a0 · sin(a1 · x + a2)


## Exclusió de punts

En `ajusta_dades` i `mostra_dades`, `exclude` indica sempre índexs de les
**dades originals**, començant per 0, abans de filtrar amb `x_min` o `x_max`.
`exclude=None` i `exclude=[]` no exclouen cap punt. S’accepten llistes o arrays
NumPy unidimensionals d’enters. Els índexs negatius, fora de rang, decimals
i booleans es rebutgen.

```python
exclude=[1, 2]                # Segon i tercer punt
exclude=np.array([1, 2])      # Mateix resultat
exclude_x=[1.5, 3.0]          # Tots els punts amb aquests valors de x
```

`exclude_x` compara els valors exactament, sense tolerància, i exclou també
els valors repetits. Es pot combinar amb `exclude` i amb els límits del rang.
També es pot incloure en els diccionaris de `mostra_varies_series`.

**Canvi de comportament:** si abans passaves valors de x decimals a `exclude`,
ara has de passar-los a `exclude_x`.

## Font de la documentació

[Documentacio.tex](Documentacio.tex) conté la documentació actualitzada, amb figures a [docs/figures](docs/figures).

Per regenerar les figures i compilar des de l'arrel:

```sh
python docs/genera_figures.py
pdflatex Documentacio.tex
pdflatex Documentacio.tex
```

Cal una instal·lació LaTeX amb català per a Babel, listings i els paquets indicats al preàmbul. El resultat és `Documentacio.pdf`.
**El PDF anterior, `Documentacio_funcions_labs.pdf`, encara no s'ha regenerat amb aquestes correccions.** El font LaTeX és la referència actualitzada.
