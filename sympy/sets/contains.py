from sympy.core import S
from sympy.core.kind import NumberKind
from sympy.core.relational import Eq, Ne
from sympy.core.symbol import Symbol
from sympy.logic.boolalg import BooleanFunction
from sympy.utilities.misc import func_name
from .sets import Set, SetKind


class Contains(BooleanFunction):
    """
    Asserts that x is an element of the set S.

    Examples
    ========

    >>> from sympy import Symbol, Integer, S, Contains
    >>> Contains(Integer(2), S.Integers)
    True
    >>> Contains(Integer(-2), S.Naturals)
    False
    >>> i = Symbol('i', integer=True)
    >>> Contains(i, S.Naturals)
    Contains(i, Naturals)

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Element_%28mathematics%29
    """
    @classmethod
    def eval(cls, x, s):

        if not isinstance(s, Set):
            raise TypeError('expecting Set, not %s' % func_name(s))

        ret = s.contains(x)
        if not isinstance(ret, Contains) and (
                ret in (S.true, S.false) or isinstance(ret, Set)):
            return ret

    @property
    def binary_symbols(self):
        return set().union(*[i.binary_symbols
            for i in self.args[1].args
            if i.is_Boolean or i.is_Symbol or
            isinstance(i, (Eq, Ne))])

    def as_set(self):
        """Rewrite direct scalar symbol membership as a set of real values.

        The target must be a symbol-free set of scalar numbers. The result
        is its intersection with the reals, even if the symbol has no real
        assumption. Inverse images of expressions and parameterized targets
        are not implemented. Unevaluated constant membership is unsupported.

        Examples
        ========

        >>> from sympy import Contains, S, Symbol
        >>> x = Symbol('x')
        >>> Contains(x, S.Reals).as_set()
        Reals
        >>> Contains(x, S.Complexes).as_set()
        Reals
        """
        x, s = self.args
        if not isinstance(x, Symbol) or x.kind is not NumberKind:
            raise NotImplementedError(
                'as_set is only implemented for direct scalar numeric symbols')
        if s.free_symbols:
            raise NotImplementedError(
                'as_set is not implemented for parameterized membership sets')
        # Complexes has no args for its inherited _kind; EmptySet has no elements.
        if s not in (S.Complexes, S.EmptySet) and s.kind != SetKind(NumberKind):
            raise NotImplementedError(
                'as_set is only implemented for scalar numeric membership sets')
        return s.intersect(S.Reals)
