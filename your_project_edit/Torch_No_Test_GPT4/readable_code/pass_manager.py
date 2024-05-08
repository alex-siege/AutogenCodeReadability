from functools import wraps
from inspect import unwrap
from typing import Callable, List, Optional
import logging

# Setup a logger for the module
logger = logging.getLogger(__name__)

# Expose key components for import elsewhere
__all__ = [
    "PassManager",
    "inplace_wrapper",
    "log_hook",
    "loop_pass",
    "this_before_that_pass_constraint",
    "these_before_those_pass_constraint",
]


def inplace_wrapper(func: Callable) -> Callable:
    """
    Decorator for functions that modify an object in place.

    Ensures that functions that are intended to modify their arguments in place
    return the modified object, regardless of their actual return value.

    Args:
        func (Callable): The function to be wrapped.

    Returns:
        Callable: The wrapped function that returns the modified object.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Execute the function
        result = func(*args, **kwargs)
        # If the first argument is None, simply return None
        if args and args[0] is None:
            return None
        # Return the first argument as the modified object
        return args[0]
    return wrapper


def log_hook(
        func: Callable,
        *args,
        pre_hook: Optional[Callable] = None,
        post_hook: Optional[Callable] = None):
    """
    Wraps a function call with logging hooks.

    Allows for logging or running arbitrary code before and after the function
    execution. Useful for debugging or logging purposes.

    Args:
        func (Callable): The function to be wrapped and logged.
        *args: Arguments to be passed to the function.
        pre_hook (Optional[Callable], optional): Function to be called before the main function. Defaults to None.
        post_hook (Optional[Callable], optional): Function to be called after the main function. Defaults to None.
    """
    # Execute pre-hook if it is provided
    if pre_hook:
        pre_hook()
    # Log function call
    logger.info(f"Calling function {func.__name__} with arguments {args}")
    # Execute the function
    func(*args)
    # Execute post-hook if it is provided
    if post_hook:
        post_hook()
    # Log completion of function
    logger.info(f"Completed function {func.__name__}")


def loop_pass(pass_list: List[Callable], obj):
    """
    Sequentially applies a list of functions (passes) to an object.

    Args:
        pass_list (List[Callable]): List of functions to apply.
        obj: The object to be modified by the passes.

    Returns:
        The object after applying all the passes.
    """
    for func in pass_list:
        # Apply each function in the list to the object
        obj = func(obj)
    return obj


def this_before_that_pass_constraint(
        this_pass: Callable,
        that_pass: Callable) -> bool:
    """
    Checks if 'this_pass' should logically come before 'that_pass'.

    Placeholder for a more complex constraint checking system.

    Args:
        this_pass (Callable): The first pass function.
        that_pass (Callable): The second pass function.

    Returns:
        bool: True if 'this_pass' should come before 'that_pass', False otherwise.
    """
    # Placeholder implementation - always returns True
    return True


def these_before_those_pass_constraint(
        first_passes: List[Callable],
        second_passes: List[Callable]) -> bool:
    """
    Checks if all functions in `first_passes` should logically come before all functions in `second_passes`.

    Placeholder for a more complex constraint checking system.

    Args:
        first_passes (List[Callable]): List of first pass functions.
        second_passes (List[Callable]): List of second pass functions.

    Returns:
        bool: True if all functions in `first_passes` should come before all functions in `second_passes`, False otherwise.
    """
    # Placeholder implementation - always returns True
    return True


def inplace_wrapper(fn: Callable) -> Callable:
    """
    Convenience wrapper for functions that modify an object in place.
    This wrapper ensures that the modified object is returned from the function,
    allowing for a more functional programming style.

    Args:
        fn: A Callable that accepts an object and performs in-place modifications on it.
           It is expected to not return the modified object itself.

    Returns:
        wrapped_fn: A Callable that wraps the original function, ensuring the
                    modified object is returned.
    """

    @wraps(fn)
    def wrapped_fn(target_object):
        """
        The wrapped function that calls the original function (fn) with a target_object,
        allowing the original function to perform in-place modifications and then
        explicitly returns the modified object.

        Args:
            target_object: The object to be modified in-place by the 'fn' Callable.

        Returns:
            The modified object (target_object) after 'fn' has been applied to it.
        """

        # Call the original function with the target object, performing in-place modifications.
        # We store the return value, if any, in '_', as it's not used
        # (convention for ignorable values).
        _ = fn(target_object)

        # Return the modified object.
        return target_object

    return wrapped_fn


def log_hook(fn: Callable, level=logging.INFO) -> Callable:
    """
    Decorator to log the outcome of a callable function.

    This decorator wraps a given function `fn` and logs its return value
    at the specified `level` when `fn` is called. This is particularly useful
    for tracking the execution and outcomes of functions within a larger application,
    especially when debugging or monitoring.

    Example:
        @log_hook
        def my_function(argument):
            # Function logic
            return result

    Args:
        fn (Callable): The function to be wrapped and logged.
        level (int, optional): The logging level at which the message will be logged.
            Defaults to logging.INFO.

    Returns:
        Callable: A wrapped version of `fn` that logs its return value upon execution.
    """
    @wraps(fn)
    def wrapped_fn(*args, **kwargs):
        # Execute the wrapped function and store the return value.
        result = fn(*args, **kwargs)

        # Log the function call and its return value at the specified log
        # level.
        logger.log(level, f"Ran pass {fn.__name__}\t Return value: {result}")

        # Return the result of the function call.
        return result

    # Return the wrapped function.
    return wrapped_fn


def loop_pass(
        base_pass: Callable,
        n_iter: Optional[int] = None,
        predicate: Optional[Callable] = None):
    """
    Applies a given pass repeatedly either for a specified number of iterations or until a condition is met.

    This function serves as a convenience wrapper for cases where a pass needs to be applied multiple times either for
    a fixed number of iterations or until a predicate condition based on the output of the pass returns False.

    Args:
        base_pass (Callable): The pass to be applied, function taking an input and returning an output.
        n_iter (Optional[int]): The number of times to loop the pass. Must be positive if specified.
        predicate (Optional[Callable]): A function that takes the output of `base_pass` and returns a boolean. If True, the loop continues.

    Returns:
        The output after applying the base_pass for the specified number of iterations or until the predicate returns False.

    Raises:
        AssertionError: If both or neither `n_iter` and `predicate` are specified.
        RuntimeError: If incorrect values are passed for n_iter or predicate.
    """
    # Ensure that exactly one of `n_iter` or `predicate` is specified.
    assert (
        n_iter is not None) ^ (
        predicate is not None), "Exactly one of `n_iter` or `predicate` must be specified."

    @wraps(base_pass)
    def new_pass(source):
        # Initialize output with the source input
        output = source

        # Check if n_iter is defined and greater than 0, and apply the pass in
        # a loop for n_iter times.
        if n_iter is not None and n_iter > 0:
            for _ in range(n_iter):
                output = base_pass(output)
        # If predicate is specified, keep applying the pass until the predicate
        # returns False.
        elif predicate is not None:
            while predicate(output):
                output = base_pass(output)
        # Raise an error if neither condition above is met, though this should
        # be caught by the assertion.
        else:
            raise RuntimeError(
                f"loop_pass must be given a positive int for n_iter (given {n_iter}) or a predicate (given {predicate})")

        return output

    return new_pass


# Pass Schedule Constraints:
#
# Implemented as 'depends on' operators. A constraint is satisfied iff a list
# has a valid partial ordering according to this comparison operator.
def _validate_pass_schedule_constraint(
        constraint: Callable[[Callable, Callable], bool], passes: List[Callable]):
    """
    Validates if the schedule of passes meets a certain constraint.

    Iterates through the passes and checks if any pair violates
    the provided constraint. If any pair violates the constraint,
    a RuntimeError is raised indicating the issue.

    Parameters:
    - constraint: A callable that takes two arguments (pass_a, pass_b) and returns a boolean indicating
                  whether the pair meets the constraint.
    - passes: A list of passes to be validated against the constraint.
    """
    for current_index, current_pass in enumerate(passes):
        # Iterate through the remaining passes to check against the current
        # pass
        for next_index, next_pass in enumerate(
                passes[current_index + 1:], start=current_index + 1):
            # If the constraint is met, continue to the next iteration
            if constraint(current_pass, next_pass):
                continue

            # Raise an error if the constraint is not met
            raise RuntimeError(
                f"Pass schedule constraint violated. Expected {current_pass} before {next_pass}"
                f" but found {current_pass} at index {current_index} and {next_pass} at index {next_index} in pass"
                f" list.")


def this_before_that_pass_constraint(this: Callable, that: Callable):
    """
    Defines a partial order ('depends on' function) where `this` must occur
    before `that`. Returns a function that takes two arguments
    and checks if the partial order is maintained.

    Parameters:
    - this: The Callable that should happen first.
    - that: The Callable that should happen after 'this'.

    Returns:
    - A function that can be used to check if 'a' can logically come before 'b'
    in the ordering defined by 'this_before_that_pass_constraint'.
    """

    def depends_on(
            first_callable: Callable,
            second_callable: Callable) -> bool:
        """
        Check if the first_callable does not violate the ordering by coming after second_callable
        when the second_callable is 'this' and first_callable is 'that'.

        Parameters:
        - first_callable: The Callable being considered to come first.
        - second_callable: The Callable being considered to come second.

        Returns:
        - A boolean indicating if the first_callable is allowed to come before
        the second_callable without violating the defined order.
        """
        # If 'that' comes before 'this', it's a violation, return False.
        # In all other cases return True, indicating no violation of order.
        return not (first_callable == that and second_callable == this)

    return depends_on


def these_before_those_pass_constraint(these: Callable, those: Callable):
    """
    Defines a partial order ('depends on' function) where `these` must occur
    before `those`. Where the inputs are 'unwrapped' before comparison.

    For this constraint, if pass 'these' is found to occur after 'those' in any sequence,
    this function will return `False` indicating an invalid order; otherwise, it returns `True`.

    Args:
        these (Callable): Pass which should occur first.
        those (Callable): Pass which should occur later.

    Returns:
        depends_on (Callable[[Object, Object], bool]): A function that checks if the order is valid.
    """

    def depends_on(pass_a: Callable, pass_b: Callable):
        """
        Determines if `pass_a` depends on `pass_b` based on the conditions defined by
        these_before_those_pass_constraint. Specifically, if 'these' must precede 'those'
        and this condition is violated, returns False.

        Args:
            pass_a (Callable): The first pass to check in the order.
            pass_b (Callable): The second pass to check in the order.

        Returns:
            bool: True if the order of passes does not violate the constraint, False otherwise.
        """
        # unwrap the passes 'a' and 'b' and compare them with 'these' and 'those' respectively
        # if 'those' comes before 'these', return False indicating an invalid
        # order
        if unwrap(pass_a) == those and unwrap(pass_b) == these:
            return False
        # If the condition is not met, it implies the order is valid or
        # unaffected by this constraint, return True
        return True

    # return the inner function `depends_on` to be used as the 'depends on'
    # function
    return depends_on


class PassManager:
    """
    Manages a collection of transformation passes and constraints to apply on a given object.
    """

    def __init__(self, passes=None, constraints=None):
        """
        Initializes the PassManager with optional lists of passes and constraints.

        Args:
            passes (Optional[List[Callable]]): List of transformation passes.
            constraints (Optional[List[Callable]]): List of constraints between passes.
        """
        self.passes = passes or []
        self.constraints = constraints or []
        self._validated = False

    @classmethod
    def build_from_passlist(cls, passes):
        """Constructs a PassManager instance from a list of passes."""
        # Future: Implement constraint management/validation
        return cls(passes=passes)

    def add_pass(self, _pass: Callable):
        """Adds a transformation pass to the PassManager."""
        self.passes.append(_pass)
        self._validated = False

    def add_constraint(self, constraint: Callable):
        """Adds a constraint for pass execution order."""
        self.constraints.append(constraint)
        self._validated = False

    def remove_pass(self, _passes: List[Callable]):
        """
        Removes passes from the manager by pass names.

        Args:
            _passes (List[Callable]): List of pass functions to be removed.
        """
        if _passes is None:
            return
        self.passes = [ps for ps in self.passes if ps.__name__ not in _passes]
        self._validated = False

    def replace_pass(self, _target: Callable, _replacement: Callable):
        """
        Replaces a target pass with a replacement pass.

        Args:
            _target (Callable): The target pass to replace.
            _replacement (Callable): The replacement pass.
        """
        self.passes = [_replacement if ps.__name__ ==
                       _target.__name__ else ps for ps in self.passes]
        self._validated = False

    def validate(self):
        """
        Validates the current pass schedule defined by `self.passes` against the constraints in `self.constraints`.

        Ensures that all defined constraints are satisfied by the pass execution order.
        """
        if not self._validated:
            for constraint in self.constraints:
                self._validate_pass_schedule_constraint(
                    constraint, self.passes)
            self._validated = True

    def _validate_pass_schedule_constraint(
            self, constraint: Callable, passes: List[Callable]):
        """
        Applies a single constraint to validate the pass execution order.

        Args:
            constraint (Callable): A constraint function.
            passes (List[Callable]): The list of passes to validate against the constraint.
        """
        # Implementation for validating a single constraint against the pass
        # list
        pass  # Placeholder for actual validation logic

    def __call__(self, source):
        """
        Applies all accumulated transformation passes in order to the input source.

        Args:
            source: The input object to be transformed.

        Returns:
            The transformed object after all passes have been applied.
        """
        self.validate()
        out = source
        for _pass in self.passes:
            out = _pass(out)
        return out
