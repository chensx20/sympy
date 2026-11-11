from sympy.core.containers import Tuple
from sympy.core.expr import unchanged
from sympy.core.numbers import I, oo, pi
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.functions.elementary.trigonometric import sin
from sympy.matrices.expressions.matexpr import MatrixSymbol
from sympy.matrices.immutable import ImmutableMatrix
from sympy.sets.contains import Contains
from sympy.sets.fancysets import ComplexRegion
from sympy.sets.sets import (Complement, DisjointUnion, FiniteSet, Intersection,
                             Interval, ProductSet, Set, Union)
from sympy.testing.pytest import raises

def test_contains_basic():
    raises(TypeError, lambda: Contains(S.Integers, 1))
    assert Contains(2, S.Integers) is S.true
    assert Contains(-2, S.Naturals) is S.false

    i = Symbol('i', integer=True)
    assert Contains(i, S.Naturals) == Contains(i, S.Naturals, evaluate=False)


def test_issue_6194():
    x = Symbol('x')
    assert unchanged(Contains, x, Interval(0, 1))
    assert Interval(0, 1).contains(x) == (S.Zero <= x) & (x <= 1)
    assert Contains(x, FiniteSet(0)) != S.false
    assert Contains(x, Interval(1, 1)) != S.false
    assert Contains(x, S.Integers) != S.false


def test_issue_10326():
    assert Contains(oo, Interval(-oo, oo)) == False
    assert Contains(-oo, Interval(-oo, oo)) == False


def test_binary_symbols():
    x = Symbol('x')
    y = Symbol('y')
    z = Symbol('z')
    assert Contains(x, FiniteSet(y, Eq(z, True))
        ).binary_symbols == {y, z}


def test_as_set():
    x = Symbol('x')
    y = Symbol('y')
    # Parameterized membership sets are not supported.
    raises(NotImplementedError, lambda:
           Contains(x, FiniteSet(y)).as_set())


def test_as_set_real_domain():
    x = Symbol('x')
    for target, expected in [
            (S.Reals, S.Reals),
            (S.Integers, S.Integers),
            (S.Rationals, S.Rationals),
            (Interval(0, 1), Interval(0, 1)),
            (S.Complexes, S.Reals),
            (FiniteSet(1, I), FiniteSet(1)),
            (FiniteSet(I), S.EmptySet)]:
        result = Contains(x, target).as_set()
        assert isinstance(result, Set)
        assert result == expected

    reals = Contains(x, S.Reals).as_set()
    relation = reals.as_relational(x)
    for value, expected in [(0, S.true), (I, S.false),
                            (oo, S.false), (-oo, S.false)]:
        assert reals.contains(value) is expected
    for value, expected in [(0, S.true), (oo, S.false), (-oo, S.false)]:
        assert relation.subs(x, value) is expected

    interval = Interval.Lopen(0, 1)
    result = Contains(x, interval).as_set()
    assert result == interval
    assert result.contains(0) is S.false
    assert result.contains(1) is S.true
    assert result.contains(S.Half) is S.true
    assert result.contains(-oo) is S.false
    assert result.contains(oo) is S.false
    assert Contains(Symbol('r', real=True), interval).as_set() == interval


def test_as_set_complex_compositions():
    x = Symbol('x')
    complement = Complement(S.Complexes, FiniteSet(I), evaluate=False)
    for target, expected in [
            (S.Complexes - FiniteSet(I), S.Reals),
            (complement, S.Reals),
            (Intersection(S.Complexes, S.Integers, evaluate=False), S.Integers),
            (Union(S.Complexes, FiniteSet(2), evaluate=False), S.Reals),
            (Union(complement, S.Integers, evaluate=False), S.Reals),
            (Intersection(Union(S.Complexes, FiniteSet(I), evaluate=False),
                          complement, evaluate=False), S.Reals)]:
        assert Contains(x, target).as_set() == expected


def test_as_set_complex_regions():
    x = Symbol('x')
    square = ComplexRegion(Interval(0, 1)*Interval(0, 1))
    disk = ComplexRegion(Interval(0, 1)*Interval(0, 2*pi), polar=True)
    for target, expected in [
            (square, Interval(0, 1)),
            (ComplexRegion(Interval.Lopen(0, 1)*Interval(0, 1)),
             Interval.Lopen(0, 1)),
            (ComplexRegion(Interval(0, 1)*Interval(1, 2)), S.EmptySet),
            (disk, Interval(-1, 1)),
            (ComplexRegion(Interval(0, 1)*Interval(0, pi/2), polar=True),
             Interval(0, 1)),
            (ComplexRegion(Interval(0, 1)*Interval.open(0, pi), polar=True),
             FiniteSet(0)),
            (square - FiniteSet(0), Interval.Lopen(0, 1)),
            (square | FiniteSet(2), Union(Interval(0, 1), FiniteSet(2))),
            (disk - FiniteSet(0),
             Union(Interval.Ropen(-1, 0), Interval.Lopen(0, 1))),
            (disk | FiniteSet(2), Union(Interval(-1, 1), FiniteSet(2)))]:
        result = Contains(x, target).as_set()
        assert isinstance(result, Set)
        assert result == expected
        for value in (-1, 0, S.Half, 1, 2, I, -oo, oo):
            assert result.contains(value) == expected.contains(value)


def test_as_set_unsupported():
    x = Symbol('x')
    y = Symbol('y')
    for expression in (x + 1, x**2, sin(x), x + y):
        error = raises(NotImplementedError,
            lambda: Contains(expression, S.Reals).as_set())
        assert 'direct scalar numeric symbols' in str(error.value)
    error = raises(NotImplementedError,
        lambda: Contains(x, Interval(0, y)).as_set())
    assert 'parameterized membership sets' in str(error.value)
    error = raises(NotImplementedError, lambda:
        Contains(x, ComplexRegion(Interval(0, y)*Interval(0, 1))).as_set())
    assert 'parameterized membership sets' in str(error.value)
    error = raises(NotImplementedError, lambda:
        Contains(x, ComplexRegion(Interval(0, y)*Interval(0, pi),
                                  polar=True)).as_set())
    assert 'parameterized membership sets' in str(error.value)

    matrix = ImmutableMatrix([1, 2])
    for argument in (Tuple(x, y), MatrixSymbol('M', 2, 2),
                     matrix, FiniteSet(x), Symbol('n', commutative=False)):
        error = raises(NotImplementedError,
            lambda: Contains(argument, S.Reals, evaluate=False).as_set())
        assert 'direct scalar numeric symbols' in str(error.value)
    for target in (ProductSet(S.Reals, S.Reals), FiniteSet(Tuple(1, 2)),
                   FiniteSet(matrix), FiniteSet(FiniteSet(1)),
                   FiniteSet(1, Tuple(1, 2)),
                   DisjointUnion(S.Integers, S.Reals),
                   Union(S.Reals, DisjointUnion(S.Integers), evaluate=False)):
        error = raises(NotImplementedError,
            lambda: Contains(x, target, evaluate=False).as_set())
        assert 'scalar numeric membership sets' in str(error.value)


def test_as_set_constants():
    assert Contains(2, S.Reals) is S.true
    assert Contains(I, S.Reals) is S.false
    assert Contains(2, S.Reals).as_set() is S.UniversalSet
    assert Contains(I, S.Reals).as_set() is S.EmptySet
    for value in (2, I):
        error = raises(NotImplementedError,
            lambda: Contains(value, S.Reals, evaluate=False).as_set())
        assert 'direct scalar numeric symbols' in str(error.value)
    assert Contains(Symbol('x'), S.EmptySet, evaluate=False).as_set() is S.EmptySet


def test_type_error():
    # Pass in a parameter not of type "set"
    raises(TypeError, lambda: Contains(2, None))
