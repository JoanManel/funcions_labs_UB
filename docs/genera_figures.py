"""Regenera les cinc figures executant els exemples numerats del LaTeX."""
from pathlib import Path
import re
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import funcions_labs as labs


def main():
    import os
    os.chdir(ROOT)
    source = (ROOT / 'Documentacio.tex').read_text(encoding='utf-8')
    examples = source.split(r'\section{Exemples extrets del notebook}', 1)[1]
    sections = re.split(r'\\subsection\*\{[1-6]\. ', examples)[1:]
    names = ['fig1_lineal.png', 'fig2_exclosos.png', 'fig3_exponencial.png',
             'fig4_sin.png', 'exemple_varies_series.png']
    target = ROOT / 'docs' / 'figures'
    target.mkdir(parents=True, exist_ok=True)
    namespace = {}
    exec('import numpy as np\nfrom funcions_labs import *', namespace)
    original_display, original_show = labs.display, plt.show
    labs.display = lambda *args: None
    plt.show = lambda: None
    try:
        for section, name in zip(sections[1:], names):
            code = re.search(r'\\begin\{lstlisting\}(.*?)\\end\{lstlisting\}',
                             section, re.S).group(1)
            exec(compile(code, name, 'exec'), namespace)
            plt.gcf().savefig(target / name, dpi=160, bbox_inches='tight')
            plt.close('all')
            print(name)
    finally:
        labs.display, plt.show = original_display, original_show


if __name__ == '__main__':
    main()
