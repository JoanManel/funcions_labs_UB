import unittest
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from unittest.mock import patch
import funcions_labs as f

class Exclusions(unittest.TestCase):
    def test_empty(self):
        for value in (None, [], np.array([])):
            np.testing.assert_array_equal(f.mascara_exclusions(4, value), [True]*4)
    def test_indices(self):
        for value in ([1, 2], np.array([1, 2]), np.array([1, 2], dtype='uint64'), [1, 1, 2]):
            np.testing.assert_array_equal(f.mascara_exclusions(4, value), [True, False, False, True])
    def test_invalid(self):
        for value in ([-1], [4], [1.0], [True], [[1]], 1):
            with self.subTest(value=value), self.assertRaises(ValueError):
                f.mascara_exclusions(4, value)
    def test_fit_plot_consistency(self):
        x = np.array([10.,11.,12.,12.,14.,15.,16.,17.])
        y = 2*x+3+np.array([.1,-.1,.2,-.2,.1,-.1,.2,-.2])
        expected = [False,True,False,False,True,False,True,False]
        options = dict(exclude=np.array([5]), exclude_x=[12.], x_min=11, x_max=16)
        with patch.object(f, 'display'), patch.object(plt, 'show'):
            for weighted in (True, False):
                ax, result = f.mostra_dades(x,y,funcio_model=f.model_lineal,incert_y=.2,ponderacio=weighted,**options)
                np.testing.assert_array_equal(result['mask'], expected)
                np.testing.assert_array_equal(ax.lines[1].get_xdata(), x[expected])
                plt.close(ax.figure)
            direct = f.ajusta_dades(x,y,f.model_lineal,incert_y=.2,**options)
            np.testing.assert_array_equal(direct['mask'], expected)
    def test_all_excluded(self):
        with self.assertRaisesRegex(ValueError, 'No queda cap punt'):
            f.ajusta_dades(np.arange(4.), np.arange(4.), f.model_lineal, exclude=[0,1,2,3])

if __name__ == '__main__':
    unittest.main()
