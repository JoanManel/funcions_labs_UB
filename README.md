# funcions_labs_UB
Aquesta llibreria conté funcions de suport en els laboratoris docents del Grau de Física de la Universitat de Barcelona. Inclou eines per a la lectura de dades, ajustos de corbes, representació gràfica i models matemàtics habituals.

# Funcions principals

## `llegeix_dades`

Llegeix un fitxer de dades delimitat per tabuladors, comes o altres separadors.

```python
llegeix_dades(path, separador='\t', comentari='#', skip=0)
````

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
             exclude=None, v_i=None)
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
             valors_inicials=None, ponderacio=True)
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

