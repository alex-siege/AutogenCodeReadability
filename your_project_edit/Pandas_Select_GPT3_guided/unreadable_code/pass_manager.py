from typing import Callable, Optional
from typing import Callable
from functools import wraps
from inspect import unwrap
from typing import Callable, List, Optional
import logging

logger = logging.getLogger(__name__)

__all__ = [
    "PassManager",
    "inplace_wrapper",
    "log_hook",
    "loop_pass",
    "this_before_that_pass_constraint",
    "these_before_those_pass_constraint",
]


# for callables which modify object inplace and return something other than
# the object on which they act


def inplace_wrapper(fn: Callable) -> Callable:
    """
    Convenience wrapper for passes which modify an object inplace. This
    wrapper makes them return the modified object instead.

    Args:
        fn (Callable[Object, Any])

    Returns:
        wrapped_fn (Callable[Object, Object])
    """

    @wraps(fn)
    def wrapped_fn(gm):
        """
        Calls the original function and returns the modified object.

        Args:
            gm (Object): The object to be modified.

        Returns:
            modified_obj (Object): The modified object.
        """

        # Call the original function and store the modified object
        modified_obj = fn(gm)

        # Return the modified object
        return modified_obj

    return wrapped_fn


def log_hook(fn: Callable, level=logging.INFO) -> Callable:
    """
    Logs callable output.

    This is useful for logging output of passes. Note inplace_wrapper replaces
    the pass output with the modified object. If we want to log the original
    output, apply this wrapper before inplace_wrapper.

    Args:
        fn (Callable[Type1, Type2])
        level: logging level (e.g. logging.INFO)

    Returns:
        wrapped_fn (Callable[Type1, Type2])
    """

    @wraps(fn)
    def wrapped_fn(gm):
        val = fn(gm)
        logger.log(level, "Ran pass %s	 Return value: %s", fn, val)
        return val

    return wrapped_fn


def loop_pass(
        base_pass: Callable,
        n_iter: Optional[int] = None,
        predicate: Optional[Callable] = None):
    """Convenience wrapper for passes which need to be applied multiple times.

    Exactly one of `n_iter` or `predicate` must be specified.

    Args:
        base_pass (Callable[Object, Object]): pass to be applied in loop
        n_iter (int, optional): number of times to loop pass
        predicate (Callable[Object, bool], optional):
    """
    assert (
        n_iter is not None) ^ (
        predicate is not None), "Exactly one of `n_iter` or `predicate` must be specified."

    @wraps(base_pass)
    def new_pass(source):
        """Apply the base_pass in a loop for n_iter times or until predicate is False.

        Args:
            source (object): input data

        Returns:
            object: output data after applying the base_pass repeatedly
        """
        output = source
        if n_iter is not None and n_iter > 0:
            output = _apply_in_loop(base_pass, output)
        elif predicate is not None:
            output = _apply_until_predicate(base_pass, output)
        else:
            raise RuntimeError(
                f"loop_pass must be given positive int n_iter (given {n_iter}) xor predicate (given {predicate})"
            )
        return output

    return new_pass


def _apply_in_loop(base_pass, output):
    """Apply the base_pass in a loop for n_iter times.

    Args:
        base_pass (Callable[Object, Object]): pass to be applied in loop
        output (object): the input data

    Returns:
        object: output data after applying the base_pass repeatedly
    """
    for _ in range(n_iter):
        output = base_pass(output)
    return output


def _apply_until_predicate(base_pass, output):
    """Apply the base_pass until predicate is False.

    Args:
        base_pass (Callable[Object, Object]): pass to be applied in loop
        output (object): the input data

    Returns:
        object: output data after applying the base_pass repeatedly
    """
    while predicate(output):
        output = base_pass(output)
    return output


# Pass Schedule Constraints:
#
# Implemented as 'depends on' operators. A constraint is satisfied iff a list
# has a valid partial ordering according to this comparison operator.
def _validate_pass_schedule_constraint(
        constraint: Callable[[Callable, Callable], bool], passes: List[Callable]):
    """
    Validates the pass schedule constraint by checking if the given constraint function is satisfied
    for all pairs of passes in the list.

    Args:
        constraint: A function that takes two passes as arguments and returns a boolean indicating
                    whether the constraint is satisfied.
        passes: A list of passes.

    Raises:
        RuntimeError: If the constraint is violated, raises an error indicating the passes
                      that violate the constraint.
    """
    for i, pass_a in enumerate(passes):
        for j, pass_b in enumerate(passes[i + 1:]):
            # check if the constraint is satisfied for each pair
            if constraint(pass_a, pass_b):
                continue
            # raise error if the constraint is violated
            raise RuntimeError(
                f"Pass schedule constraint violated. Expected {pass_a} before {pass_b}"
                f" but found {pass_a} at index {i} and {pass_b} at index {j} in pass list.")


def this_before_that_pass_constraint(this: Callable, that: Callable) -> bool:
    """
    Defines a partial order ('depends on' function) where `this` must occur
    before `that`.
    """
    def depends_on(a: Callable, b: Callable) -> bool:
        """
        Checks if `a` depends on `b`.

        Args:
            a (Callable): The first function to check.
            b (Callable): The second function to check.

        Returns:
            bool: True if `a` depends on `b`, False otherwise.
        """
        return a != that or b != this

    return depends_on


def these_before_those_pass_constraint(these: Callable, those: Callable):
    """
    Defines a partial order ('depends on' function) where `these` must occur
    before `those`. Where the inputs are 'unwrapped' before comparison.

    For example, the following pass list and constraint list would be invalid.
    ```
    passes = [
        loop_pass(pass_b, 3),
        loop_pass(pass_a, 5),
    ]

    constraints = [
        these_before_those_pass_constraint(pass_a, pass_b)
    ]
    ```

    Args:
        these (Callable): pass which should occur first
        those (Callable): pass which should occur later

    Returns:
        depends_on (Callable[[Object, Object], bool]
    """

    def depends_on(a: Callable, b: Callable):
        if unwrap(a) == those and unwrap(b) == these:
            return False
        return True

    return depends_on


class PassManager:
    """
    Construct a PassManager.

    Collects passes and constraints. This defines the pass schedule, manages
    pass constraints and pass execution.

    Args:
        passes (Optional[List[Callable]]): list of passes. A pass is a
            callable which modifies an object and returns modified object
        constraint (Optional[List[Callable]]): list of constraints. A
            constraint is a callable which takes two passes (A, B) and returns
            True if A depends on B and False otherwise. See implementation of
            `this_before_that_pass_constraint` for example.
    """
    passes: List[Callable]
    constraints: List[Callable]
    _validated: bool = False

    def __init__(self, passes=None, constraints=None):
        self.passes = passes or []
        self.constraints = constraints or []

    @classmethod
    def build_from_passlist(cls, passes):
        """
        Builds a PassManager object from a list of passes.
        """
        pm = PassManager(passes)
        # TODO(alexbeloi): add constraint management/validation
        return pm

    def add_pass(self, _pass: Callable):
        """
        Adds a pass to the pass list.
        """
        self.passes.append(_pass)
        self._validated = False

    def add_constraint(self, constraint):
        """
        Adds a constraint to the constraint list.
        """
        self.constraints.append(constraint)
        self._validated = False

    def remove_pass(self, _passes: List[Callable]):
        """
        Removes specified passes from the pass list.
        """
        if _passes is None:
            return
        passes_left = []
        for ps in self.passes:
            if ps.__name__ not in _passes:
                passes_left.append(ps)
        self.passes = passes_left
        self._validated = False

    def replace_pass(self, _target, _replacement):
        """
        Replaces a pass in the pass list with another pass.
        """
        passes_left = []
        for ps in self.passes:
            if ps.__name__ == _target.__name__:
                passes_left.append(_replacement)
            else:
                passes_left.append(ps)
        self.passes = passes_left
        self._validated = False

    def validate(self):
        """
        Validates that current pass schedule defined by `self.passes` is valid
        according to all constraints in `self.constraints`
        """
        if self._validated:
            return
        for constraint in self.constraints:
            _validate_pass_schedule_constraint(constraint, self.passes)
        self._validated = True

    def __call__(self, source):
        """
        Executes the pass scheduler on the given source object.
        """
        self.validate()
        out = source
        for _pass in self.passes:
            out = _pass(out)
        return out
