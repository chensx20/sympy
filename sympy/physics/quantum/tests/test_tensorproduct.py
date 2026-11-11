from sympy import I, S, Float, Integer, Mul, Pow, symbols, Matrix

from sympy.physics.quantum.commutator import Commutator as Comm
from sympy.physics.quantum.tensorproduct import TensorProduct
from sympy.physics.quantum.tensorproduct import TensorProduct as TP
from sympy.physics.quantum.tensorproduct import (
    tensor_product_simp, tensor_product_simp_Mul)
from sympy.physics.quantum.anticommutator import AntiCommutator
from sympy.physics.quantum.qexpr import QuantumError
from sympy.physics.quantum.dagger import Dagger
from sympy.physics.quantum.qubit import Qubit, QubitBra
from sympy.physics.quantum.operator import OuterProduct
from sympy.physics.quantum.density import Density
from sympy.core.trace import Tr
from sympy.physics.paulialgebra import Pauli
from sympy.utilities.pytest import raises

A, B, C = symbols('A,B,C', commutative=False)
x = symbols('x')

mat1 = Matrix([[1, 2*I], [1 + I, 3]])
mat2 = Matrix([[2*I, 3], [4*I, 2]])


def test_tensor_product_dagger():
    assert Dagger(TensorProduct(I*A, B)) == \
        -I*TensorProduct(Dagger(A), Dagger(B))
    assert Dagger(TensorProduct(mat1, mat2)) == \
        TensorProduct(Dagger(mat1), Dagger(mat2))


def test_tensor_product_abstract():

    assert TP(x*A, 2*B) == x*2*TP(A, B)
    assert TP(A, B) != TP(B, A)
    assert TP(A, B).is_commutative is False
    assert isinstance(TP(A, B), TP)
    assert TP(A, B).subs(A, C) == TP(C, B)


def test_tensor_product_expand():
    assert TP(A + B, B + C).expand(tensorproduct=True) == \
        TP(A, B) + TP(A, C) + TP(B, B) + TP(B, C)


def test_tensor_product_commutator():
    assert TP(Comm(A, B), C).doit().expand(tensorproduct=True) == \
        TP(A*B, C) - TP(B*A, C)
    assert Comm(TP(A, B), TP(B, C)).doit() == \
        TP(A, B)*TP(B, C) - TP(B, C)*TP(A, B)


def test_tensor_product_simp():
    assert tensor_product_simp(TP(A, B)*TP(B, C)) == TP(A*B, B*C)


def test_tensor_product_power_identity():
    identity = TP(1, 1)
    for tensor in (identity, TP(1, Pauli(3))):
        assert tensor**0 == S.One
        assert tensor**1 == tensor
        assert tensor**2 == identity
        assert tensor*tensor == identity
        assert (tensor*tensor).expand(tensorproduct=True) == identity
        assert tensor_product_simp(tensor*tensor) == identity
        power = Pow(tensor, 2, evaluate=False)
        assert tensor_product_simp(power) == identity
        assert power.expand(tensorproduct=True) == identity
        assert tensor_product_simp(
            Mul(3, power, evaluate=False)) == 3*identity
    for exponent in (3, 4, -1, -2, -3):
        expected = TP(1, Pauli(3)**exponent)
        assert TP(1, Pauli(3))**exponent == expected
        assert tensor_product_simp(
            Pow(TP(1, Pauli(3)), exponent, evaluate=False)) == expected


def test_tensor_product_power_integer():
    for tensor in (TP(A, B), TP(A, B, C), TP(A*B, B*C)):
        assert tensor**0 == S.One
        assert tensor**1 == tensor
        for exponent in (Integer(2), Integer(5), Integer(-1), Integer(-3),
                         Integer(10)**10):
            expected = TP(*[arg**exponent for arg in tensor.args])
            assert tensor**exponent == expected
            assert tensor_product_simp(
                Pow(tensor, exponent, evaluate=False)) == expected
    assert TP(A*B, C)**2 == TP((A*B)**2, C**2)
    assert TP(A*B, C)**2 != TP(A**2*B**2, C**2)
    assert tensor_product_simp((2*x*TP(A, B))**3) == 8*x**3*TP(A**3, B**3)
    assert tensor_product_simp((2*TP(A, B))**-2) == TP(A**-2, B**-2)/4


def test_tensor_product_power_unsupported():
    n = symbols('n', integer=True)
    tensor = TP(A, B)
    for exponent in (S.Half, -S.Half, Float(2), x, n):
        power = Pow(tensor, exponent, evaluate=False)
        assert tensor**exponent == power
        assert tensor_product_simp(power) == power
        assert power.expand(tensorproduct=True) == power


def test_tensor_product_simp_power_factors():
    left = TP(A, B)
    right = TP(B, C)
    ordinary = Mul(A, A, evaluate=False)
    assert tensor_product_simp_Mul(ordinary) == ordinary
    for exponent in (0, 1, 2, -2):
        power = Pow(left, exponent, evaluate=False)
        for expression, expected in (
            (Mul(3*x, power, right, evaluate=False),
             3*x*TP(A**exponent*B, B**exponent*C)),
            (Mul(3*x, right, power, evaluate=False),
             3*x*TP(B*A**exponent, C*B**exponent)),
            (Mul(right, power, right, evaluate=False),
             TP(B*A**exponent*B, C*B**exponent*C)),
        ):
            assert tensor_product_simp(expression) == expected
            assert tensor_product_simp_Mul(expression) == expected
        assert tensor_product_simp_Mul(
            Mul(3*x, power, evaluate=False)) == 3*x*left**exponent
    assert tensor_product_simp_Mul(
        Mul(Pow(left, 2, evaluate=False), left, evaluate=False)) == left**3
    assert tensor_product_simp_Mul(
        Mul(3, Pow(left, 1, evaluate=False), left,
            evaluate=False)) == 3*left**2
    assert tensor_product_simp_Mul(
        Mul(Pow(left, 2, evaluate=False),
            Pow(left, -2, evaluate=False), evaluate=False)) == TP(1, 1)
    assert tensor_product_simp(
        Pow(Mul(left, right, evaluate=False), 3, evaluate=False)
    ) == TP((A*B)**3, (B*C)**3)
    assert tensor_product_simp(
        Mul(Pow(TP(A, B, C), -2, evaluate=False), TP(C, A, B),
            evaluate=False)) == TP(A**-2*C, B**-2*A, C**-2*B)


def test_tensor_product_simp_power_coefficients():
    A, B, C, D = symbols('A B C D', commutative=False)
    left, right = TP(A, B), TP(C, D)
    for coefficient in (2, -2*x):
        base = Mul(coefficient, left, right, evaluate=False)
        for exponent in (2, 3, -1, -2):
            power = Pow(base, exponent, evaluate=False)
            powered = TP((A*C)**exponent, (B*D)**exponent)
            expected = coefficient**exponent*powered
            assert tensor_product_simp(power) == expected
            assert tensor_product_simp(expected) == expected
            for expression, combined in (
                (Mul(3, power, evaluate=False), 3*expected),
                (Mul(3, power, right, evaluate=False),
                 3*coefficient**exponent*TP(
                     (A*C)**exponent*C, (B*D)**exponent*D)),
                (Mul(3, right, power, evaluate=False),
                 3*coefficient**exponent*TP(
                     C*(A*C)**exponent, D*(B*D)**exponent)),
            ):
                assert tensor_product_simp(expression) == combined
                assert tensor_product_simp_Mul(expression) == combined
                assert tensor_product_simp(combined) == combined
    repeated = Mul(3, left, left, evaluate=False)
    assert tensor_product_simp(repeated) == 3*TP(A**2, B**2)
    assert tensor_product_simp_Mul(repeated) == 3*TP(A**2, B**2)
    n = symbols('n', integer=True)
    for exponent in (S.Half, Float(2), x, n):
        power = Pow(Mul(2, left, right, evaluate=False),
                    exponent, evaluate=False)
        expected = (2*TP(A*C, B*D))**exponent
        assert tensor_product_simp(power) == expected
        assert tensor_product_simp(expected) == expected
        for factors in ((power, right), (right, power)):
            expression = Mul(3, *factors, evaluate=False)
            blocked = Mul(3, *[expected if arg == power else arg
                               for arg in factors])
            assert tensor_product_simp(expression) == blocked
            assert tensor_product_simp_Mul(expression) == blocked
            assert tensor_product_simp(blocked) == blocked


def test_tensor_product_simp_power_blocked():
    n = symbols('n', integer=True)
    left, right = TP(A, B), TP(B, C)
    for exponent in (S.Half, Float(2), x, n):
        power = Pow(left, exponent, evaluate=False)
        for factors in ((power, right), (right, power),
                        (right, power, left)):
            expression = Mul(3*x, *factors, evaluate=False)
            expected = Mul(3*x, *factors)
            assert tensor_product_simp(expression) == expected
            assert tensor_product_simp_Mul(expression) == expected
        mixed = Mul(Pow(right, 2, evaluate=False), power, evaluate=False)
        assert tensor_product_simp(mixed) == right**2*power


def test_tensor_product_simp_power_arity():
    left, right = TP(A, B), TP(A, B, C)
    for first, second in ((left, right), (right, left)):
        for exponent in (1, 2, -1):
            power = Pow(first, exponent, evaluate=False)
            raises(QuantumError, lambda: tensor_product_simp(
                Mul(power, second, evaluate=False)))
            raises(QuantumError, lambda: tensor_product_simp(
                Mul(second, power, evaluate=False)))


def test_tensor_product_simp_power_recursive():
    power = Pow(TP(A, B), 2, evaluate=False)
    expressions = (
        power, power + TP(B, C),
        Pow(power, 3, evaluate=False),
        Comm(power, TP(B, C)), AntiCommutator(power, TP(B, C)),
        Mul(2, power, TP(B, C), evaluate=False),
    )
    for expression in expressions:
        simplified = tensor_product_simp(expression)
        assert tensor_product_simp(simplified) == simplified
    assert tensor_product_simp(Pow(power, 3, evaluate=False)) == TP(A**6, B**6)
    assert tensor_product_simp(power + TP(B, C)) == TP(A**2, B**2) + TP(B, C)
    assert tensor_product_simp(Comm(power, TP(B, C))) == Comm(
        TP(A**2, B**2), TP(B, C))
    assert tensor_product_simp(AntiCommutator(power, TP(B, C))) == AntiCommutator(
        TP(A**2, B**2), TP(B, C))


def test_issue_5923():
    # most of the issue regarding sympification of args has been handled
    # and is tested internally by the use of args_cnc through the quantum
    # module, but the following is a test from the issue that used to raise.
    assert TensorProduct(1, Qubit('1')*Qubit('1').dual) == \
        TensorProduct(1, OuterProduct(Qubit(1), QubitBra(1)))


def test_eval_trace():
    # This test includes tests with dependencies between TensorProducts
    #and density operators. Since, the test is more to test the behavior of
    #TensorProducts it remains here

    A, B, C, D, E, F = symbols('A B C D E F', commutative=False)

    # Density with simple tensor products as args
    t = TensorProduct(A, B)
    d = Density([t, 1.0])
    tr = Tr(d)
    assert tr.doit() == 1.0*Tr(A*Dagger(A))*Tr(B*Dagger(B))

    ## partial trace with simple tensor products as args
    t = TensorProduct(A, B, C)
    d = Density([t, 1.0])
    tr = Tr(d, [1])
    assert tr.doit() == 1.0*A*Dagger(A)*Tr(B*Dagger(B))*C*Dagger(C)

    tr = Tr(d, [0, 2])
    assert tr.doit() == 1.0*Tr(A*Dagger(A))*B*Dagger(B)*Tr(C*Dagger(C))

    # Density with multiple Tensorproducts as states
    t2 = TensorProduct(A, B)
    t3 = TensorProduct(C, D)

    d = Density([t2, 0.5], [t3, 0.5])
    t = Tr(d)
    assert t.doit() == (0.5*Tr(A*Dagger(A))*Tr(B*Dagger(B)) +
                        0.5*Tr(C*Dagger(C))*Tr(D*Dagger(D)))

    t = Tr(d, [0])
    assert t.doit() == (0.5*Tr(A*Dagger(A))*B*Dagger(B) +
                        0.5*Tr(C*Dagger(C))*D*Dagger(D))

    #Density with mixed states
    d = Density([t2 + t3, 1.0])
    t = Tr(d)
    assert t.doit() == ( 1.0*Tr(A*Dagger(A))*Tr(B*Dagger(B)) +
                        1.0*Tr(A*Dagger(C))*Tr(B*Dagger(D)) +
                        1.0*Tr(C*Dagger(A))*Tr(D*Dagger(B)) +
                        1.0*Tr(C*Dagger(C))*Tr(D*Dagger(D)))

    t = Tr(d, [1] )
    assert t.doit() == ( 1.0*A*Dagger(A)*Tr(B*Dagger(B)) +
                        1.0*A*Dagger(C)*Tr(B*Dagger(D)) +
                        1.0*C*Dagger(A)*Tr(D*Dagger(B)) +
                        1.0*C*Dagger(C)*Tr(D*Dagger(D)))
