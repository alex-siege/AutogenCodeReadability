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
    Logs the output of the callable function.

    It's useful when we need to log the output of passes. The inplace_wrapper replaces
    the output of the pass with the modified object and if we want to log the original
    output, apply this wrapper before the inplace_wrapper.

    Args:
        fn (Callable[Type1, Type2])
        level: logging level (e.g. logging.INFO)

    Returns:
        wrapped_fn (Callable[Type1, Type2])
    """

    # use 'wraps' to keep the original function's signature when the decorator
    # is applied
    @wraps(fn)
    def wrapped_fn(gm):  # the wrapper function to insert logging
        val = fn(gm)  # run the given function and store its output
        logger.log(
            level,
            "Ran pass %s	 Return value: %s",
            fn,
            val)  # log the details
        return val  # return the original output

    return wrapped_fn  # return the wrapped function


def loop_pass(
        base_pass: Callable,
        n_iter: Optional[int] = None,
        predicate: Optional[Callable] = None):
    """
    This function is a wrapper for passes which must be applied multiple times.
    It takes exactly one of `n_iter` (number of iterations) or `predicate` as a parameter.

    Args:
        base_pass (Callable): A function that needs to be applied in a loop.
        n_iter (Optional, int): The number of times the base pass is looped over data, defaulted to None.
        predicate (Optional, Callable): A function that returns a boolean to terminate the loop, defaulted to None.
    """
    # Ensuring that one and only one of n_iter or predicate is specified
    assert (
        n_iter is not None) ^ (
        predicate is not None), "Exactly one of `n_iter` or `predicate` must be specified."

    def _apply_in_loop(base_func, data, iteration_count):
        """
        This private function applies the base pass for a specific number of iterations.

        Args:
            base_func (Callable): The function to be applied.
            data (object): The data on which the function is applied.
            iteration_count (int): The number of times the function is applied.

        Returns:
            object: The data after the function is applied for the given number of iterations.
        """
        for i in range(iteration_count):
            data = base_func(data)

        return data

    def _apply_until_predicate(base_func, data):
        """
        This private function applies the base pass until the predicate function returns False.

        Args:
            base_func (Callable): The function to be applied.
            data (object): The data on which the function is applied.

        Returns:
            object: The data after the function is applied until predicate returns False.
        """
        while predicate(data):
            data = base_func(data)

        return data

    def new_pass(source):
        """
        This function applies the base pass in a loop either for n_iter times or until predicate returns False.

        Args:
            source (object): The data which needs to be processed.

        Returns:
            object: The output data after the base pass has been repeatedly applied.
        """
        # If n_iter was specified, apply base pass for n_iter times.
        if n_iter is not None and n_iter > 0:
            processed_data = _apply_in_loop(base_pass, source, n_iter)
        # If predicate was specified, apply base pass until predicate returns
        # False.
        elif predicate is not None:
            processed_data = _apply_until_predicate(base_pass, source)
        else:
            # Handle case where neither n_iter nor predicate were correctly
            # specified
            raise RuntimeError(
                f"loop_pass must be given positive int n_iter (given {n_iter}) or predicate (given {predicate})"
            )

        return processed_data

    return new_pass


def _apply_in_loop(base_pass, output, n_iter):
    """Apply the base_pass in a loop for n_iter times.

    Args:
        base_pass (Callable[Object, Object]): pass to be applied in loop
        output (object): the input data
        n_iter (int): the number of iterations the base_pass should be applied

    Returns:
        object: output data after applying the base_pass repeatedly
    """
    # Loop n_iter times
    for _ in range(n_iter):
        # Apply base_pass to output
        output = base_pass(output)
    # Return the final output after all iterations
    return output


def _apply_until_predicate(base_function, data):
    """
    Function to repeatedly apply a given function until a predicate on the input evaluates to False

    Args:
        base_function (function): The function to be repeatedly applied
        data (object): The data on which the function is to be applied

    Returns:
        object: The result of applying the function until predicate is False
    """

    # Loop continues until the predicate function returns False for the data
    while predicate(data):
        # The base function is applied to the data
        data = base_function(data)

    # The final transformed data is returned
    return data


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


def this_before_that_pass_constraint(
        first_function: Callable,
        second_function: Callable) -> bool:
    """
    Defines a partial order ('depends_on' relationship) where `first_function`
    must occur before `second_function`.

    Args:
        first_function (Callable): The first function in the order.
        second_function (Callable): The second function in the order.

    Returns:
        bool: A boolean value that indicates the success of the 'depends_on' relationship.
    """

    # Nested function definition for checking dependencies.
    def depends_on(first_check: Callable, second_check: Callable) -> bool:
        """
        Checks if `first_check` depends on `second_check`.

        Args:
            first_check (Callable): The first function to compare.
            second_check (Callable): The second function to compare.

        Returns:
            bool: True if `first_check` depends on `second_check`, False otherwise.
        """
        # Return true only if `first_check` isn't `second_function`
        # or `second_check` isn't `first_function`
        return first_check != second_function or second_check != first_function

    # Return the output of the depends_on function
    return depends_on


def these_before_those_pass_constraint(
        first_pass: Callable,
        second_pass: Callable):
    """
    This function orders two passes, with `first_pass` occurring before `second_pass`.
    The order forms a constraint where the function 'unwrap' is applied to the inputs before comparing them.

    Example of invalid case:
    passes = [
        loop_pass(pass_b, 3),
        loop_pass(pass_a, 5),
    ]

    constraints = [
        these_before_those_pass_constraint(pass_a, pass_b)
    ]

    Args:
        first_pass (Callable): function execution that should occur first
        second_pass (Callable): function execution that should occur later

    Returns:
      depends_on function (Callable[[Object, Object], bool]):
    """

    def depends_on(a: Callable, b: Callable):
        """
        Nested function that checks if a pass depends on another pass.
        If  'b' pass is the first_pass and 'a' pass is the second_pass in the order,
        then the function will return False as it breaks the defined constraint.
        In all other cases, it returns True, indicating no constraint violation.

        Args:
          a (Callable): a function
          b (Callable): a function

        Returns:
           bool : True if no constraint violation, False otherwise
        """
        # unwrap the functions a and b
        unwrapped_a = unwrap(a)
        unwrapped_b = unwrap(b)

        # If the second_pass occurs before the first_pass, return False
        if unwrapped_a == second_pass and unwrapped_b == first_pass:
            return False

        # In all other cases, no constraint is violated, return True
        return True

    # return the depends_on function
    return depends_on


class PassManager:
    """
    Collects passes and constraints defining the pass schedule.
    Manages pass constraints and pass execution.
    """

    def __init__(self, passes=None, constraints=None):
        """Initialize pass manager with optional lists of passes and constraints."""
        self.passes = passes if passes else []
        self.constraints = constraints if constraints else []
        self._validated = False

    @classmethod
    def build_from_passlist(cls, passes):
        """
        Create a PassManager object from a list of passes.
        """
        return PassManager(passes)

    def add_pass(self, new_pass):
        """Add a pass to the pass list and update validation status."""
        self.passes.append(new_pass)
        self._validated = False

    def add_constraint(self, constraint):
        """Add a constraint to the constraint list and update validation status."""
        self.constraints.append(constraint)
        self._validated = False

    def remove_pass(self, target_passes):
        """
        Remove specified passes from the pass list.
        """
        if not target_passes:
            return
        self.passes = self._get_remaing_passes_after_removal(target_passes)
        self._validated = False

    def _get_remaing_passes_after_removal(self, target_passes):
        """Return a list of passes after removing the target passes."""
        return [
            current_pass for current_pass in self.passes if current_pass.__name__ not in target_passes]

    def replace_pass(self, target, replacement):
        """
        Replace a target pass in the pass list with another pass.
        """
        self.passes = self._get_replaced_passes_list(target, replacement)
        self._validated = False

    def _get_replaced_passes_list(self, target, replacement):
        """Return a list of passes after replacing the target pass."""
        return [replacement if current_pass.__name__ ==
                target.__name__ else current_pass for current_pass in self.passes]

    def validate(self):
        """
        Validate the current pass schedule according to all constraints.
        """
        if not self._validated:  # Skip validation if already done
            for constraint in self.constraints:
                self._validate_pass_schedule_constraint(constraint)
            self._validated = True

    def _validate_pass_schedule_constraint(self, constraint):
        """Validate individual pass schedule constraint."""
        return constraint(self.passes)

    def __call__(self, source):
        """Execute the pass scheduler on the source object."""
        self.validate()
        return self._execute_passes_on_source(source)

    def _execute_passes_on_source(self, source):
        """Apply all passes to the source."""
        for single_pass in self.passes:
            source = single_pass(source)
        return source
