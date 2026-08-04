import collections
import collections.abc
for name in ('Mapping', 'MutableMapping', 'Sequence', 'Set', 'Iterable', 'Container', 'Callable'):
    if not hasattr(collections, name):
        setattr(collections, name, getattr(collections.abc, name))
