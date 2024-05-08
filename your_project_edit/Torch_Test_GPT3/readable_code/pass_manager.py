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
    Convenience wrapper for functions that modify an object inplace. This
    wrapper makes them return the modified object instead.

    Args:
        fn (Callable[Object, Any])

    Returns:
        wrapped_fn (Callable[Object, Object])
    """

    @wraps(fn)
    def wrapped_fn(obj_before_modification):
        obj_after_modification = fn(obj_before_modification)
        # Return the modified object
        return obj_after_modification

    return wrapped_fn


def log_hook(fn: Callable, level=logging.INFO) -> Callable:
    """
    Logs callable output.

    This decorator function logs the output of the given callable function.

    Args:
        fn (Callable): The function to log the output of.
        level (int): The logging level (default is logging.INFO).

    Returns:
        Callable: The wrapped function that logs the output.
    """
    @wraps(fn)
    def wrapped_fn(data):
        """
        This is an inner function that logs the output of the provided function.

        Args:
            data: The input data to the function.

        Returns:
            Return value of the provided function along with logging information.
        """
        result = fn(data)
        logger.log(level, "Ran pass %s\t Return value: %s", fn, result)
        return result

    return wrapped_fn


def loop_pass(
        base_pass: Callable,
        n_iter: Optional[int] = None,
        predicate: Optional[Callable] = None):
    """
    Convenience wrapper for passes which need to be applied multiple times.

    Exactly one of `n_iter`or `predicate` must be specified.

    Args:
        base_pass (Callable[Object, Object]): pass to be applied in loop
        n_iter (int, optional): number of times to loop pass
        predicate (Callable[Object, bool], optional):

    """
    assert (n_iter is not None) ^ (
        predicate is not None
    ), "Exactly one of `n_iter`or `predicate` must be specified."

    @wraps(base_pass)
    def new_pass(source):
        """
        Apply the provided pass either for specified number of iterations or until a predicate is met.

        Args:
            source (Object): input data

        Returns:
            Object: output data after applying pass
        """
        output = source

        if n_iter is not None and n_iter > 0:
            for _ in range(n_iter):
                output = base_pass(output)

        elif predicate is not None:
            while predicate(output):
                output = base_pass(output)

        else:
            raise RuntimeError(
                f"loop_pass must be given positive int n_iter (given "
                f"{n_iter}) xor predicate (given {predicate})"
            )

        return output

    return new_pass


# Pass Schedule Constraints:
#
# Implemented as 'depends on' operators. A constraint is satisfied iff a list
# has a valid partial ordering according to this comparison operator.
def _validate_pass_schedule_constraint(
    constraint: Callable[[Callable, Callable], bool], passes: List[Callable]
):
    """
    Validate the pass schedule constraint based on the given constraint function.

    Args:
    - constraint: Function that checks if two passes satisfy the constraint.
    - passes: List of passes to validate.

    Raises:
    - RuntimeError: If the pass schedule constraint is violated.
    """
    for i, pass_a in enumerate(passes):
        for j, pass_b in enumerate(passes[i + 1:]):
            if not constraint(pass_a, pass_b):
                raise RuntimeError(
                    f"pass schedule constraint violated. Expected {pass_a} before {pass_b}"
                    f" but found {pass_a} at index {i} and {pass_b} at index {j} in pass list.")


def this_before_that_pass_constraint(this: Callable, that: Callable):
    """
    Defines a partial order ('depends on' function) where `this` must occur
    before `that`.
    """

    def depends_on(a: Callable, b: Callable):
        """
        Check if 'a' depends on 'b' such that 'this' must occur before 'that'.

        Parameters:
        a (Callable): The first function to check dependency.
        b (Callable): The second function to check dependency.

        Returns:
        bool: True if 'a' depends on 'b' preserving the order 'this' before 'that', False otherwise.
        """
        if a == that and b == this:
            return False
        return True

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
        """
        Initialize a PassManager instance.

        Args:
            passes (Optional[List[Callable]]): list of passes
            constraints (Optional[List[Callable]]): list of constraints
        """
        self.passes = passes or []
        self.constraints = constraints or []

    @classmethod
    def build_from_passlist(cls, passes):
        """
        Construct a PassManager from a list of passes.

        Args:
            passes (List[Callable]): list of passes to be added

        Returns:
            PassManager: a new PassManager object with the provided passes
        """
        pm = PassManager(passes)
        # TODO(alexbeloi): add constraint management/validation
        return pm

    def add_pass(self, new_pass: Callable):
        """
        Add a pass to the PassManager.

        Args:
            new_pass (Callable): pass to be added
        """
        self.passes.append(new_pass)
        self._validated = False

    def add_constraint(self, constraint):
        """
        Add a constraint to the PassManager.

        Args:
            constraint (Callable): constraint to be added
        """
        self.constraints.append(constraint)
        self._validated = False

    def remove_pass(self, passes_to_remove: List[Callable]):
        """
        Remove specified passes from the PassManager.

        Args:
            passes_to_remove (List[Callable]): passes to be removed
        """
        if passes_to_remove is None:
            return
        self.passes = [
            ps for ps in self.passes if ps.__name__ not in passes_to_remove]
        self._validated = False

    def replace_pass(self, target_pass, replacement_pass):
        """
        Replace a pass with a new pass in the PassManager.

        Args:
            target_pass (Callable): pass to be replaced
            replacement_pass (Callable): pass to replace with
        """
        self.passes = [replacement_pass if ps.__name__ ==
                       target_pass.__name__ else ps for ps in self.passes]
        self._validated = False

    def validate(self):
        """
        Validate that the current pass schedule is valid based on constraints.
        """
        if not self._validated:
            for constraint in self.constraints:
                _validate_pass_schedule_constraint(constraint, self.passes)
            self._validated = True

    def __call__(self, source):
        """
        Execute passes on the source object.

        Args:
            source: input source object

        Returns:
            modified source object after applying passes
        """
        self.validate()
        out = source
        for _pass in self.passes:
            out = _pass(out)
        return out
