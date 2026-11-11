# -*- coding: utf-8 -*-
from __future__ import absolute_import

from sympy.codegen import Assignment
from sympy.codegen.ast import none
from sympy.core import (
    Derivative, Expr, Function, Mod, Symbol, symbols, Eq, Le, Gt, zoo, oo, Rational
)
from sympy.core.numbers import pi
from sympy.functions import acos, Piecewise, sign, sin
from sympy.logic import And, Or
from sympy.matrices import SparseMatrix, MatrixSymbol
from sympy.printing.lambdarepr import LambdaPrinter
from sympy.printing.pycode import (
    MpmathPrinter, NumPyPrinter, PythonCodePrinter, pycode, SciPyPrinter,
    SymPyPrinter
)
from sympy.tensor import IndexedBase
from sympy.utilities.pytest import raises

x, y, z = symbols('x y z')


def test_PythonCodePrinter():
    prntr = PythonCodePrinter()
    assert not prntr.module_imports
    assert prntr.doprint(x**y) == 'x**y'
    assert prntr.doprint(Mod(x, 2)) == 'x % 2'
    assert prntr.doprint(And(x, y)) == 'x and y'
    assert prntr.doprint(Or(x, y)) == 'x or y'
    assert not prntr.module_imports
    assert prntr.doprint(pi) == 'math.pi'
    assert prntr.module_imports == {'math': {'pi'}}
    assert prntr.doprint(acos(x)) == 'math.acos(x)'
    assert prntr.doprint(Assignment(x, 2)) == 'x = 2'
    assert prntr.doprint(Piecewise((1, Eq(x, 0)),
                        (2, x>6))) == '((1) if (x == 0) else (2) if (x > 6) else None)'
    assert prntr.doprint(Piecewise((2, Le(x, 0)),
                        (3, Gt(x, 0)), evaluate=False)) == '((2) if (x <= 0) else'\
                                                        ' (3) if (x > 0) else None)'
    assert prntr.doprint(sign(x)) == '(0.0 if x == 0 else math.copysign(1, x))'


def test_pycode_indexed():
    p, q = IndexedBase('p'), IndexedBase('q')
    i, j = symbols('i j', integer=True)
    assert pycode(p[0]) == 'p[0]'
    assert PythonCodePrinter().doprint(p[0]) == 'p[0]'
    assert pycode(p[i, j]) == 'p[i, j]'
    assert pycode(p[2, i, j]) == 'p[2, i, j]'
    assert pycode(p[i + 1, 2*j]) == 'p[i + 1, 2*j]'
    assert pycode(p[sin(i)]) == 'p[math.sin(i)]'
    assert pycode(p[sin(i)], fully_qualified_modules=False) == 'p[sin(i)]'
    assert pycode(p[q[i] + 1]) == 'p[q[i] + 1]'
    assert pycode(p[Symbol('if')]) == 'p[if_]'
    raises(ValueError, lambda: pycode(p[Symbol('if')], error_on_reserved=True))
    assert pycode(p[i + 1, 2*j], human=False) == (
        set(), set(), 'p[i + 1, 2*j]')
    printer = PythonCodePrinter()
    assert printer.doprint(p[sin(i)]) == 'p[math.sin(i)]'
    assert printer.module_imports == {'math': {'sin'}}


def test_pycode_indexed_unsupported_child():
    p = IndexedBase('p')
    i = Symbol('i', integer=True)
    f = Function('f')
    derivative = Derivative(f(i), i)
    number_symbols, unsupported, code = pycode(p[derivative], human=False)
    assert number_symbols == set()
    assert unsupported == {derivative}
    assert code == 'p[Derivative(f(i), i)]'


def test_pycode_indexed_inherited_printers():
    p = IndexedBase('p')
    i, j = symbols('i j', integer=True)
    for printer in (MpmathPrinter, NumPyPrinter, SciPyPrinter, SymPyPrinter,
                    LambdaPrinter):
        assert printer().doprint(p[i, j]) == 'p[i, j]'
    for printer, module in ((MpmathPrinter, 'mpmath'),
                            (NumPyPrinter, 'numpy'),
                            (SciPyPrinter, 'numpy')):
        assert printer().doprint(p[sin(i)]) == 'p[%s.sin(i)]' % module


def test_MpmathPrinter():
    p = MpmathPrinter()
    assert p.doprint(sign(x)) == 'mpmath.sign(x)'
    assert p.doprint(Rational(1, 2)) == 'mpmath.mpf(1)/mpmath.mpf(2)'

def test_NumPyPrinter():
    p = NumPyPrinter()
    assert p.doprint(sign(x)) == 'numpy.sign(x)'
    A = MatrixSymbol("A", 2, 2)
    assert p.doprint(A**(-1)) == "numpy.linalg.inv(A)"
    assert p.doprint(A**5) == "numpy.linalg.matrix_power(A, 5)"


def test_SciPyPrinter():
    p = SciPyPrinter()
    expr = acos(x)
    assert 'numpy' not in p.module_imports
    assert p.doprint(expr) == 'numpy.arccos(x)'
    assert 'numpy' in p.module_imports
    assert not any(m.startswith('scipy') for m in p.module_imports)
    smat = SparseMatrix(2, 5, {(0, 1): 3})
    assert p.doprint(smat) == 'scipy.sparse.coo_matrix([3], ([0], [1]), shape=(2, 5))'
    assert 'scipy.sparse' in p.module_imports


def test_pycode_reserved_words():
    s1, s2 = symbols('if else')
    raises(ValueError, lambda: pycode(s1 + s2, error_on_reserved=True))
    py_str = pycode(s1 + s2)
    assert py_str in ('else_ + if_', 'if_ + else_')


class CustomPrintedObject(Expr):
    def _numpycode(self, printer):
        return 'numpy'

    def _mpmathcode(self, printer):
        return 'mpmath'


def test_printmethod():
    obj = CustomPrintedObject()
    assert NumPyPrinter().doprint(obj) == 'numpy'
    assert MpmathPrinter().doprint(obj) == 'mpmath'


def test_codegen_ast_nodes():
    assert pycode(none) == 'None'


def test_issue_14283():
    prntr = PythonCodePrinter()

    assert prntr.doprint(zoo) == "float('nan')"
    assert prntr.doprint(-oo) == "float('-inf')"

def test_NumPyPrinter_print_seq():
    n = NumPyPrinter()

    assert n._print_seq(range(2)) == '(0, 1,)'
