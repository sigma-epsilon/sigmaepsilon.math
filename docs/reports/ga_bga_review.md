# Review: `GeneticAlgorithm` / `BinaryGeneticAlgorithm`

Files reviewed:
- `src/sigmaepsilon/math/optimize/ga.py` (`Genom`, `GeneticAlgorithm`)
- `src/sigmaepsilon/math/optimize/bga.py` (`BinaryGeneticAlgorithm`)
- `src/sigmaepsilon/math/optimize/state.py` (`OptimizerState`)

Verified with `poetry run python` reproductions against the current working tree
(branch `feature/101-create-bga-for-native-binary-optimization`), not just static reading.

## Summary

The design (template-method GA driven by a coroutine, pydantic-backed `Genom`/
`OptimizerState`, an explicit `Status` enum, and a `to_scipy()` bridge) is solid and
above-average for this kind of library. However, there are three confirmed, reproducible
bugs, one of which crashes a documented, tested public code path, and a metaclass
mismatch that quietly defeats the "abstract base class" contract.

## Does it follow best practices?

Partially. The template-method pattern, docstrings (NumPy style, with math and doctest
examples), and deprecation handling are good practice. But:

- `@abstractmethod` is used from `abc` without the class inheriting `ABC` or using
  `ABCMeta`. The decorator is purely decorative here — it does not prevent
  instantiation of an incomplete subclass, and does not even prevent instantiating
  `GeneticAlgorithm` itself (which fails for an unrelated reason, see Finding 1).
- `__slots__` is declared on the base class but not on the subclass, silently
  reintroducing a `__dict__` on every instance (see Finding 1). If `__slots__` is
  meant to be a real constraint/memory optimization, it needs to be declared (as
  `__slots__ = ()` or the actual extra slot names) on every subclass.
- Randomness goes through the global `numpy.random` singleton exclusively
  (`np.random.randint`, `np.random.rand`, `np.random.choice`) with no seeding hook, so
  runs are not reproducible and two `GeneticAlgorithm` instances in the same process
  are not isolated from each other's random draws.

## Confirmed bugs (reproduced)

### 1. `elitism=None` crashes `solve()`

`BinaryGeneticAlgorithm.select` (`bga.py:187-188`) does:

```python
winners, others = self.divide(fitness)
winners = winners.tolist()
```

`GeneticAlgorithm.divide` (`ga.py:559-560`) returns `[], list(range(self.nPop))` (plain
Python lists) when `elitism is None`, but returns NumPy arrays otherwise. Calling
`.tolist()` on the plain list raises:

```
AttributeError: 'list' object has no attribute 'tolist'
```

Reproduced:
```python
from sigmaepsilon.math.optimize import BinaryGeneticAlgorithm as BGA
bga = BGA(lambda x: x[0], [[-1, 1]], nPop=8, elitism=None, maxiter=3)
bga.solve()  # AttributeError: 'list' object has no attribute 'tolist'
```

`elitism=None` is a documented, user-facing way to "turn off" elitism, and
`tests/test_ga.py::test_BGA_elitism_eq_None` exists — but that test only calls
`BGA.evolve()` once. Because `evolver()` is a generator, the *first* `next()` call
after priming only runs `populate()` and yields before `select()` is ever invoked; the
crash only happens on the second generation onward (i.e. inside real `solve()` runs, or
`evolve(cycles=2+)`). This is a real coverage gap masking a real bug, not just a
theoretical one.

### 2. Typo'd attribute (`_pnenotypes`) crashes the base class, and is silently swallowed by a `__slots__` omission in the subclass

`GeneticAlgorithm.__init__` (`ga.py:215`) has:

```python
self._pnenotypes = None   # dead code, meant to be self._phenotypes
```

`__slots__` on `GeneticAlgorithm` (`ga.py:157-179`) lists `_phenotypes`, but not
`_pnenotypes`. Since `GeneticAlgorithm`'s only base is `object`, this line should raise
`AttributeError` immediately in `__init__`. Reproduced directly:

```python
from sigmaepsilon.math.optimize.ga import GeneticAlgorithm
GeneticAlgorithm(lambda x: x[0], [[-1, 1]], nPop=8)
# AttributeError: 'GeneticAlgorithm' object has no attribute '_pnenotypes'
```

It "works" for `BinaryGeneticAlgorithm` only by accident: `BinaryGeneticAlgorithm` does
not declare its own `__slots__`, so Python silently gives every instance a `__dict__`,
and the stray `self._pnenotypes = None` lands there instead of raising. Net effect:

- The real `_phenotypes` slot is untouched by this line (dead code — no functional
  impact on `BinaryGeneticAlgorithm` beyond a wasted attribute).
- The `__slots__` memory optimization on `GeneticAlgorithm` is void for every subclass
  in the codebase, because none of them re-declare `__slots__ = ()`.
- The base class is not directly usable/testable in isolation, which is presumably why
  this was never caught — nothing instantiates `GeneticAlgorithm` directly in the test
  suite, only `BinaryGeneticAlgorithm`.

### 3. `select`'s actual contract silently narrows the abstract method it implements

`GeneticAlgorithm.select` is declared as:

```python
@abstractmethod
def select(self, genotypes: ndarray, phenotypes: ndarray) -> ndarray:
    """... Both `genotypes` and `phenotypes` must be provided."""
```

i.e. two *required* positional arguments. `BinaryGeneticAlgorithm.select` (`bga.py:168`)
overrides it with two *optional*, defaulted-to-`None` arguments, and explicitly raises
`NotImplementedError` the moment either is actually supplied. So the only concrete
implementation in the codebase directly contradicts the documented contract of the
method it "implements" — this is a Liskov substitution violation and is confusing for
anyone writing a second subclass by following the base class docstring.

## Three best points

1. **Coroutine-driven evolution loop (`evolver`/`evolve`/`solve`)** — using a Python
   generator to hold the "current generation" state lets callers single-step evolution
   (`evolve(1)`) for debugging/visualization, or run to convergence via `solve()`,
   without duplicating the loop body. Clean, idiomatic use of generators for stateful
   iteration.
2. **Template-method structure** (`populate`/`encode`/`decode`/`crossover`/`mutate`/
   `select`/`stopping_criteria` as extension points) cleanly separates the
   representation-agnostic GA machinery (elitism, champion tracking, stopping,
   function-evaluation bookkeping) from the binary-encoding-specific logic in
   `BinaryGeneticAlgorithm`. Adding e.g. a real-valued or permutation-based GA should be
   a matter of subclassing `GeneticAlgorithm` again.
3. **Structured, typed state objects** — `Genom` and `OptimizerState` are pydantic
   models (validated, serializable), `Status` is an explicit `Enum`, and
   `OptimizerState.to_scipy()` gives a drop-in `scipy.optimize.OptimizeResult`, making
   the GA composable with code that already expects SciPy's optimizer result shape.
   The `fittness` → `fitness` typo rename is also handled gracefully with a
   `DeprecationWarning` rather than a silent break.

## Three weakest points

1. **`elitism=None` crashes `solve()`** — Finding 1 above. This is the most severe issue
   since it's a documented, user-facing configuration that fails on the main public
   entry point (not just an edge case buried in a private method).
2. **`__slots__`/`abstractmethod` are not actually enforced** — Findings 2 and 3 above.
   The class *looks* like it has a strict, ABC-enforced, memory-slim contract, but none
   of that is actually true for any real subclass, which is worse than not attempting it
   at all (false confidence).
3. **No reproducibility / RNG isolation** — every stochastic operation
   (`populate`, `crossover`, `mutate`, `select`, `random_parents_generator`) draws from
   the global `numpy.random` state. There is no `seed`/`random_state` constructor
   argument or per-instance `Generator`. This makes runs non-reproducible, makes
   parallel/concurrent GA instances in the same process interfere with each other's
   randomness, and makes regression testing of specific outcomes impossible without
   monkeypatching `numpy.random` globally.

## Recommended improvements

- Fix the three confirmed bugs: correct `_pnenotypes` → remove entirely (it's dead
  code once fixed, `_phenotypes` is already reset via the `genotypes` setter); make
  `divide()` return NumPy arrays in the `elitism is None` branch too (or make `select`
  not assume a NumPy return type); either declare `__slots__ = ()` on
  `BinaryGeneticAlgorithm` (and any future subclass) or drop `__slots__` from the base
  class if it isn't going to be consistently maintained.
- Either make `GeneticAlgorithm` a real `ABC` (`class GeneticAlgorithm(ABC):`) so
  incomplete subclasses fail fast at instantiation, or drop `@abstractmethod` and
  document these as "override me" hooks with `NotImplementedError` bodies — pick one,
  the current half-measure gives false assurance.
- Add a `seed`/`random_state` parameter, backed by `numpy.random.default_rng(seed)`
  stored per-instance, and thread it through every stochastic call site instead of the
  module-global `np.random`.
- Reconcile `select`'s abstract signature with reality: either make `genotypes`/
  `phenotypes` genuinely optional (and drop "must be provided" from the base
  docstring), or have `BinaryGeneticAlgorithm.select` actually support them instead of
  raising `NotImplementedError`.
- Vectorize/parallelize `evaluate()` for the non-symbolic path — currently a plain
  Python `for x in phenotypes` loop calling `self.fnc(x)` once per individual; for
  population sizes in the hundreds and expensive objective functions this is the
  dominant cost and is an obvious candidate for `multiprocessing`/`joblib`/vectorized
  evaluation support.
- Consider extending the family with a real-valued (continuous, non-binary-encoded) GA
  subclass, pluggable selection strategies (roulette/rank-based, not just BGA's
  hardcoded 3-way tournament) as injectable strategy objects, and a population-diversity
  metric surfaced on `OptimizerState` to help detect premature convergence — currently
  the only stopping signal is "champion age", which says nothing about the rest of the
  population.
