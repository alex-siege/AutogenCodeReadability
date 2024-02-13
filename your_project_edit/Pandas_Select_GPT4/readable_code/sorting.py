""" miscellaneous sorting / groupby utilities """
from __future__ import annotations

from collections import defaultdict
from typing import (
    TYPE_CHECKING,
    Callable,
    DefaultDict,
    cast,
)

import numpy as np

from pandas._libs import (
    algos,
    hashtable,
    lib,
)
from pandas._libs.hashtable import unique_label_indices

from pandas.core.dtypes.common import (
    ensure_int64,
    ensure_platform_int,
)
from pandas.core.dtypes.generic import (
    ABCMultiIndex,
    ABCRangeIndex,
)
from pandas.core.dtypes.missing import isna

from pandas.core.construction import extract_array

if TYPE_CHECKING:
    from collections.abc import (
        Hashable,
        Iterable,
        Sequence,
    )

    from pandas._typing import (
        ArrayLike,
        AxisInt,
        IndexKeyFunc,
        Level,
        NaPosition,
        Shape,
        SortKind,
        npt,
    )

    from pandas import (
        MultiIndex,
        Series,
    )
    from pandas.core.arrays import ExtensionArray
    from pandas.core.indexes.base import Index


def ensure_monotonic(target, ascending):
    """Check if the target is monotonic according to the sort order.
    
    Parameters
    ----------
    target : Index
        The index to check.
    ascending : bool, or list of bools
        The sort order.

    Returns
    -------
    bool
        True if the target is monotonic according to the sort order, False otherwise.
    """
    is_monotonic = (
        (np.all(ascending) and target.is_monotonic_increasing) or 
        (not np.any(ascending) and target.is_monotonic_decreasing)
    )
    return is_monotonic


def sort_multi_index(target, ascending, na_position):
    """Sort MultiIndex and return the indexer.
    
    Parameters
    ----------
    target : ABCMultiIndex
        The multi-index to sort.
    ascending : bool
        The sort order.
    na_position : NaPosition
        The position of NA values.

    Returns
    -------
    ndarray
        The indexer for sorting.
    """
    codes = [level.codes for level in target._get_codes_for_sorting()]
    return lexsort_indexer(
        codes, orders=ascending, na_position=na_position, codes_given=True
    )


def get_indexer_indexer(target, level, ascending, kind, na_position, sort_remaining, key):
    """
    Helper method that returns the indexer according to input parameters for
    the sort_index method of DataFrame and Series.
    """
    # Apply key mapping to the target and sort levels to be monotonic
    target = ensure_key_mapped(target, key, levels=level)
    target = target._sort_levels_monotonic()

    # Check if the target needs sorting based on the level
    if level is not None:
        _, indexer = target.sortlevel(
            level,
            ascending=ascending,
            sort_remaining=sort_remaining,
            na_position=na_position
        )
    # Check for monotonicity to possibly avoid sorting
    elif ensure_monotonic(target, ascending):
        return None
    # Handle sorting of MultiIndex
    elif isinstance(target, ABCMultiIndex):
        indexer = sort_multi_index(target, ascending, na_position)
    # Handle sorting of regular Index
    else:
        # Ensure ascending is a boolean because it can't be a Sequence for a regular Index
        indexer = nargsort(
            target, kind=kind, ascending=bool(ascending), na_position=na_position
        )

    return indexer



def get_group_index(labels, shape: Shape, sort: bool, xnull: bool) -> npt.NDArray[np.int64]:
    """
    Calculate the offset index for each label combination given a multidimensional shape.
    
    Parameters
    ----------
    labels : sequence of arrays
        Integers identifying levels at each location.
    shape : tuple[int, ...]
        Number of unique levels at each location.
    sort : bool
        If True, the function preserves the lexical ranks of labels.
    xnull : bool
        If True, -1 labels (nulls) are passed through.

    Returns
    -------
    np.ndarray
        An array of type int64 where elements represent the offset index for 
        each label combination.
    """

    def _int64_cut_off(shape) -> int:
        """
        Determines how many shape elements can be processed before the product exceeds int64.
        """
        product_limit = lib.i8max
        product = 1
        for index, dimension in enumerate(shape):
            product *= int(dimension)
            if product >= product_limit:
                return index
        return len(shape)

    def maybe_lift(label, dimension: int) -> tuple[np.ndarray, int]:
        """
        Adjusts labels and dimensions to avoid negative values in output.
        """
        # If label contains -1, increment label and dimension to ensure positive values
        return (label + 1, dimension + 1) if (label == -1).any() else (label, dimension)

    # Ensure all label arrays are of type int64
    labels = [ensure_int64(label_array) for label_array in labels]

    # Copy the shape to avoid modifications of the input tuple
    dimensions = list(shape)

    # Process labels and dimensions to lift -1 values if `xnull` is False
    if not xnull:
        labels, dimensions = zip(*[maybe_lift(lab, size) for lab, size in zip(labels, dimensions)])

    def _create_group_indices(labels, dimensions):
        # Initialize the output array
        group_indices = np.zeros_like(labels[0], dtype=np.int64)
        stride = np.prod(dimensions[1:], dtype=np.int64)

        for i, (label, dimension) in enumerate(zip(labels, dimensions)):
            group_indices += label * stride
            # Recalculate stride based on next dimension to avoid overflow
            stride = stride // dimension if dimension else np.int64(0)
            
            # If xnull is true, mark -1 labels as such in the output
            if xnull and (label == -1).any():
                group_indices[label == -1] = -1

        return group_indices

    # Process labels in chunks to avoid overflow, combine chunks after processing
    while True:
        cutoff_level = _int64_cut_off(dimensions)

        # Calculate indices for the current chunk
        current_indices = _create_group_indices(labels[:cutoff_level], dimensions[:cutoff_level])

        if cutoff_level == len(dimensions):  # All chunks processed
            return current_indices

        # Prepare the next chunk by compressing current results (if `sort` is True)
        compressed_indices, unique_indices = compress_group_index(current_indices, sort=sort)

        # Update labels and dimensions for the next chunk
        labels = [compressed_indices] + labels[cutoff_level:]
        dimensions = [len(unique_indices)] + dimensions[cutoff_level:]



def get_compressed_ids(
    labels, sizes: Shape
) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.int64]]:
    """
    Compress the group index for a cartesian product of all possible labels into a smaller space.

    This function computes compressed IDs that act as offsets into a list of unique 
    observation group IDs, thus saving on memory for huge spaces.

    Parameters
    ----------
    labels : list of label arrays
        Arrays of labels that define groups in a dataset.
        
    sizes : tuple[int]
        The sizes of each level in `labels`.

    Returns
    -------
    tuple
        - comp_ids: An array of integer pointers representing compressed group IDs.
        - obs_group_ids: An array of int64 representing unique observation group IDs.
    """
    # Calculate group index for all possible combinations of labels
    group_index = get_group_index(labels, sizes, sort=True, xnull=False)
    
    # Compress the group index to obtain compressed IDs and observation group IDs
    compressed_ids, observation_group_ids = compress_group_index(group_index, sort=True)
    
    return compressed_ids, observation_group_ids



def is_int64_overflow_possible(shape: Shape) -> bool:
    the_prod = 1
    for x in shape:
        the_prod *= int(x)

    return the_prod >= lib.i8max


def _decons_group_index(comp_labels: npt.NDArray[np.intp], shape: Shape) -> list[npt.NDArray[np.intp]]:
    """
    Deconstruct the group indices into labels for each dimension.

    Parameters:
    comp_labels (np.ndarray): The flattened labels representing multi-dimensional groups.
    shape (Shape): The shape tuple representing the multi-dimensional space.

    Returns:
    list[npt.NDArray[np.intp]]: A list of np.ndarrays, where each array represents labels for a dimension.

    Raises:
    ValueError: If an overflow is detected which suggests incorrect factorization.
    """
    if is_int64_overflow_possible(shape):
        raise ValueError("cannot deconstruct factorized group indices!")

    label_list = []
    for axis_labels in _generate_axis_labels(comp_labels, shape):
        label_list.append(axis_labels)
    return label_list[::-1]


def _generate_axis_labels(comp_labels: npt.NDArray[np.intp], shape: Shape):
    """
    Generate labels for each axis/dimension given the composite labels and the shape.

    Parameters:
    comp_labels (np.ndarray): The composite labels representing multi-dimensional groups.
    shape (Shape): The shape tuple representing the dimensions of groups.

    Yields:
    np.ndarray: Labels for a single axis.
    """
    factor = 1
    offset = np.array(0)

    for axis_size in reversed(shape):
        labels = _calc_axis_labels(comp_labels, offset, factor, axis_size)
        _apply_mask_to_labels(labels, comp_labels)
        
        yield labels

        offset = labels * factor
        factor *= axis_size


def _calc_axis_labels(comp_labels, offset, factor, axis_size):
    """
    Calculate the labels for a specific axis.

    Parameters:
    comp_labels (np.ndarray): The composite labels representing multi-dimensional groups.
    offset (np.ndarray): The offset to be applied when calculating labels.
    factor (int): The factor to be applied for modulo and division operations.
    axis_size (int): The size of the current axis being processed.

    Returns:
    np.ndarray: The calculated labels for the current axis.
    """
    return (comp_labels - offset) % (factor * axis_size) // factor


def _apply_mask_to_labels(labels, comp_labels):
    """
    Apply a mask to the labels where composite labels are negative.

    Parameters:
    labels (np.ndarray): The labels to which the mask will be applied.
    comp_labels (np.ndarray): The composite labels used to determine the mask.

    Modifies:
    labels (np.ndarray): The same labels array passed in, but with masked values updated.
    """
    np.putmask(labels, comp_labels < 0, -1)



def _update_shape_for_exclusion(shape: tuple[int], lift: np.ndarray) -> tuple[int]:
    """
    Update the shape to account for label exclusions marked with -1.

    Parameters
    ----------
    shape : tuple[int]
        Original shape of the array.
    lift : np.ndarray
        Array indicating which axes have the null exclusion represented by -1.

    Returns
    -------
    tuple[int]
        Updated shape.
    """
    # Update the array shape by incrementing dimensions for excluding nulls
    updated_shape = np.asarray(shape, dtype=np.intp) + lift
    return tuple(updated_shape)


def decons_obs_group_ids(
    comp_ids: npt.NDArray[np.intp],
    obs_ids: npt.NDArray[np.intp],
    shape: tuple[int],
    labels: Sequence[npt.NDArray[np.signedinteger]],
    xnull: bool,
) -> list[npt.NDArray[np.intp]]:
    """
    Reconstruct labels from observed group ids.

    Parameters
    ----------
    comp_ids : np.ndarray
        Component IDs from which to reconstruct labels.
    obs_ids: np.ndarray
        Observation IDs used for label reconstruction.
    shape : tuple[int]
        Shape of the full label array.
    labels : Sequence[np.ndarray]
        List of label arrays.
    xnull : bool
        If true, nulls (marked with -1) are excluded from the reconstruction.

    Returns
    -------
    list[np.ndarray]
        List of reconstructed label arrays.
    """
    # Determine if any label set contains the null label (-1)
    contains_null_label = np.fromiter(((label_set == -1).any() for label_set in labels), dtype=np.intp)
    
    if not xnull:
        # Update shape if exclusion of null labels is needed
        shape = _update_shape_for_exclusion(shape, contains_null_label)

    # Fast path: Deconstruct labels if there's no risk of int64 overflow
    if not is_int64_overflow_possible(shape):
        reconstructed_labels = _decons_group_index(obs_ids, shape)
        # Adjust for null label exclusion if necessary
        if xnull or not contains_null_label.any():
            return reconstructed_labels
        else:
            return [label - offset for label, offset in zip(reconstructed_labels, contains_null_label)]

    # Slow path: Get unique label indices for components and reconstruct labels
    unique_indices = unique_label_indices(comp_ids)
    return [label_set[unique_indices].astype(np.intp, subok=False, copy=True) for label_set in labels]



def lexsort_indexer(
    keys: Sequence[ArrayLike | Index | Series],
    orders=None,
    na_position: str = "last",
    key: Callable | None = None,
    codes_given: bool = False,
) -> npt.NDArray[np.intp]:
    """
    Performs lexical sorting on a set of keys.

    Parameters
    ----------
    keys : Sequence[ArrayLike | Index | Series]
        Sequence of arrays to be sorted by the indexer.
        Sequence[Series] is only if key is not None.
    orders : bool or list of booleans, optional
        Determines the sorting order for each element in keys. If a list,
        it must be the same length as keys. This determines whether the
        corresponding element in keys should be sorted in ascending
        (True) or descending (False) order. If bool, applied to all
        elements as above. If None, defaults to True.
    na_position : {'first', 'last'}, default 'last'
        Determines placement of NA elements in the sorted list ("last" or "first")
    key : Callable, optional
        Callable key function applied to every element in keys before sorting.
    codes_given: bool, default False
        Avoid categorical materialization if codes are already provided.

    Returns
    -------
    np.ndarray[np.intp]
        An ndarray of indices that sort the keys lexically.
    """
    validate_na_position(na_position)
    orders = normalize_orders(orders, keys)
    labels = generate_labels(keys, orders, na_position, key, codes_given)

    # Reverse labels for lexsort, as the last key is the primary sort key
    return np.lexsort(labels[::-1])

def validate_na_position(na_position: str):
    """Validates that the `na_position` parameter is acceptably defined."""
    if na_position not in ["last", "first"]:
        raise ValueError(f"invalid na_position: {na_position}")

def normalize_orders(orders, keys):
    """
    Normalizes the `orders` parameter to be a list of booleans representing the
    sort order for each key.
    """
    if isinstance(orders, bool):
        return [orders] * len(keys)
    elif orders is None:
        return [True] * len(keys)
    else:
        return orders

def generate_labels(keys, orders, na_position, key, codes_given):
    """Generates the labels used for lexical sorting."""
    from pandas.core.arrays import Categorical
    labels = []
    
    for key_array, order in zip(keys, orders):
        key_array = ensure_key_mapped(key_array, key)
        codes, n_categories = get_codes_and_categories(key_array, codes_given)
        codes = handle_na_positions(codes, n_categories, na_position)
        
        # If order is False, we sort in descending order
        if not order:
            mask_descending = codes != -1
            codes = np.where(mask_descending, n_categories - codes - 1, codes)
        
        labels.append(codes)
    
    return labels

def get_codes_and_categories(key_array, codes_given):
    """
    Gets the codes and categories or category counts for a given array-like key.
    It uses the existing codes if `codes_given` is True, otherwise, it calculates them.
    """
    if codes_given:
        codes = cast(np.ndarray, key_array)
        num_categories = codes.max() + 1 if codes.size else 0
    else:
        categorical = Categorical(key_array, ordered=True)
        codes = categorical.codes
        num_categories = len(categorical.categories)
    return codes, num_categories

def handle_na_positions(codes, num_categories, na_position):
    """
    Adjusts the code labels for NA values based on the `na_position` parameter.
    """
    mask_na = codes == -1
    if na_position == "last" and mask_na.any():
        codes = np.where(mask_na, num_categories, codes)
    
    return codes



def nargsort(
    items: ArrayLike | Index | Series,
    kind: SortKind = "quicksort",
    ascending: bool = True,
    na_position: str = "last",
    key: Callable | None = None,
    mask: npt.NDArray[np.bool_] | None = None,
) -> npt.NDArray[np.intp]:
    """
    Sorts an array or array-like items, handling NaNs and allowing customization of sort order and NaN position.

    Parameters
    ----------
    items : array-like (np.ndarray, ExtensionArray, Index, or Series)
    kind : str, default is 'quicksort', the sorting algorithm to be used.
    ascending : bool, default True, sort in ascending order.
    na_position : str, default 'last', position of NaNs after sorting ('first' or 'last').
    key : Callable, optional, a custom key function to influence the sort order.
    mask : np.ndarray[bool], optional, to specify the NaN positions, used by ExtensionArray.argsort.

    Returns
    -------
    np.ndarray[np.intp], indices that would sort the array.
    """
    if key:
        items = apply_sort_key(items, key)
        return nargsort(items, kind=kind, ascending=ascending, na_position=na_position, key=None, mask=mask)

    # Handle RangeIndex and ExtensionArray types separately.
    if isinstance(items, ABCRangeIndex):
        return sort_range_index(items, ascending)
    elif isinstance(items, ABCMultiIndex):
        raise_multiindex_error()

    items, mask = prep_items_and_mask(items, mask)

    # Sort numpy arrays.
    if isinstance(items, np.ndarray):
        return sort_numpy_array(items, mask, kind, ascending, na_position)

def apply_sort_key(items, key):
    """Apply a sort key function to items."""
    return ensure_key_mapped(items, key)

def sort_range_index(items, ascending):
    """Sort a RangeIndex."""
    return items.argsort(ascending=ascending)

def raise_multiindex_error():
    """Raise an error for unsupported MultiIndex type."""
    raise TypeError("nargsort does not support MultiIndex. Use index.sort_values instead.")

def prep_items_and_mask(items, mask):
    """Prepare items and mask for sorting."""
    if not isinstance(items, np.ndarray):
        items = extract_array(items)
    if mask is None:
        mask = np.asarray(isna(items))
    return items, mask

def sort_numpy_array(items, mask, kind, ascending, na_position):
    """Sort a numpy array."""
    idx = np.arange(items.size)
    non_nan_elements, non_nan_idx = filter_nans(items, mask, idx)
    sorted_idx = apply_sorting(non_nan_elements, non_nan_idx, kind, ascending)
    return combine_with_nan_indices(sorted_idx, mask, na_position, ascending)

def filter_nans(items, mask, idx):
    """Separate non-NaN elements and their indices."""
    non_nans = items[~mask]
    non_nan_idx = idx[~mask]
    if not ascending:
        non_nans = non_nans[::-1]
        non_nan_idx = non_nan_idx[::-1]
    return non_nans, non_nan_idx

def apply_sorting(non_nans, non_nan_idx, kind, ascending):
    """Apply sorting to non-NaN elements."""
    sorted_indices = non_nan_idx[non_nans.argsort(kind=kind)]
    if not ascending:
        sorted_indices = sorted_indices[::-1]
    return sorted_indices

def combine_with_nan_indices(sorted_idx, mask, na_position, ascending):
    """Combine sorted indices with NaN indices according to na_position."""
    nan_idx = np.nonzero(mask)[0]
    # Place the NaNs at the end or the beginning depending on na_position
    if na_position == "last":
        combined_indexer = np.concatenate([sorted_idx, nan_idx])
    elif na_position == "first":
        combined_indexer = np.concatenate([nan_idx, sorted_idx])
    else:
        raise ValueError(f"Invalid na_position: {na_position}")
    return ensure_platform_int(combined_indexer)



def nargminmax(values: ExtensionArray, method: str, axis: AxisInt = 0):
    """
    Calculate the index of the minimum or maximum value in an ExtensionArray,
    handling missing values appropriately.

    Parameters
    ----------
    values : ExtensionArray
        The array to perform the operation on.
    method : str
        A string indicating which operation to perform: "argmax" or "argmin".
    axis : AxisInt, default 0
        The axis along which to operate. By default, flattened input is used.

    Returns
    -------
    int or np.ndarray
        The index or array of indices of the minimum or maximum value along the given axis.
    """
    assert method in {"argmax", "argmin"}
    # Select the appropriate NumPy function based on the method
    func = np.argmax if method == "argmax" else np.argmin

    # Create a boolean mask indicating missing values
    missing_values_mask = np.asarray(isna(values))
    # Get the array values suitable for argsort
    sortable_values = values._values_for_argsort()

    if sortable_values.ndim > 1:
        # Multidimensional case: handle missing values and axis
        return _calc_argminmax_multidim(sortable_values, missing_values_mask, func, axis)

    # Single-dimensional case
    return _nanargminmax(sortable_values, missing_values_mask, func)

def _calc_argminmax_multidim(arr_values, mask, func, axis):
    """
    Calculate argmin/argmax for multidimensional arrays, taking into account missing values.

    Parameters
    ----------
    arr_values : np.ndarray
        The array of values from which to find the min/max.
    mask : np.ndarray
        The boolean mask indicating missing values in the original array.
    func : function
        The NumPy function (np.argmax or np.argmin) to apply.
    axis : AxisInt
        The axis along which to operate.

    Returns
    -------
    np.ndarray
        An array of indices of the minimum or maximum value along the given axis.
    """
    if mask.any():
        if axis == 1:
            zipped = zip(arr_values, mask)
        else:
            zipped = zip(arr_values.T, mask.T)
        # Apply the _nanargminmax function to each pair of (values, mask)
        return np.array([_nanargminmax(v, m, func) for v, m in zipped])
    # If no missing values, apply the function directly
    return func(arr_values, axis=axis)



def _nanargminmax(values: np.ndarray, mask: npt.NDArray[np.bool_], func) -> int:
    """
    Helper function to return the index of the minimum or maximum value in 'values',
    ignoring the NaNs as indicated by 'mask'. The operation (min or max) is defined by 'func'.
    
    Parameters:
    values (np.ndarray): An array of numerical values which may contain NaNs.
    mask (npt.NDArray[np.bool_]): A boolean array where True indicates a NaN in the corresponding 'values' array.
    func (callable): A numpy function like np.nanargmin or np.nanargmax to apply.
    
    Returns:
    int: The index in `values` of the found minimum or maximum non-NaN value.
    """
    # Generate an array of indices corresponding to the 'values' array
    idx = np.arange(values.shape[0])

    # Filter out NaNs from 'values' using the provided 'mask'
    non_nans = values[~mask]

    # Correspondingly, filter out indices of NaNs from the 'idx' array
    non_nan_idx = idx[~mask]

    # Apply the function (e.g., np.nanargmin or np.nanargmax) and return the index of the result
    return non_nan_idx[func(non_nans)]



def _ensure_key_mapped_multiindex(
    index: MultiIndex, key: Callable, level=None
) -> MultiIndex:
    """
    Apply a key function to specified levels of a MultiIndex and return a new MultiIndex.

    Parameters
    ----------
    index : MultiIndex
        The original MultiIndex to which the key function is to be applied.
    key : Callable
        A function that takes an Index and returns an Index of the same shape. The function
        is applied to the specified levels. The name of the level can be used to distinguish
        between levels.
    level : list-like, int, or str, optional
        The level(s) of the index to which the key function is applied. Can be a single level or a list
        of levels. If None, the function is applied to all levels.

    Returns
    -------
    MultiIndex
        A new MultiIndex with the key function applied to the levels.
    """
    
    # Determine the levels to apply key function.
    sort_levels = _get_sort_levels(index, level)
    
    # Apply the key function to the specified levels to create a new list of index labels.
    new_levels = [
        _apply_key_to_level(index, key, level, sort_levels)
        for level in range(index.nlevels)
    ]
    
    # Create a new MultiIndex using the modified levels.
    return type(index).from_arrays(new_levels)

def _get_sort_levels(index: MultiIndex, level):
    """
    Process input level(s) and return a list of levels as integers to which the key function is applied.

    Parameters
    ----------
    index : MultiIndex
        The MultiIndex from the original function.
    level : list-like, int, or str, or None
        Level information from the original function.

    Returns
    -------
    list
        A list of levels as integers.
    """
    if level is None:
        return list(range(index.nlevels))  # Apply to all levels.
    elif isinstance(level, (str, int)):
        return [index._get_level_number(level)]  # Single level specified.
    else:
        return [index._get_level_number(lev) for lev in level]  # Multiple levels specified.

def _apply_key_to_level(index: MultiIndex, key: Callable, level: int, sort_levels: list):
    """
    Apply the key function to a level of the index if it's specified in sort_levels.

    Parameters
    ----------
    index : MultiIndex
        The MultiIndex from the original function.
    key : Callable
        The key function from the original function.
    level : int
        The current level being processed.
    sort_levels : list
        A list of levels as integers to which the key function is applied.

    Returns
    -------
    Index
        The modified level if the key function was applied, or the original level otherwise.
    """
    level_values = index._get_level_values(level)
    return key(level_values) if level in sort_levels else level_values



# Ensure the given key is applied to the provided values correctly without altering shape or type

def ensure_key_mapped(
    values: ArrayLike | Index | Series, key: Callable | None, levels=None
) -> ArrayLike | Index | Series:
    """
    Ensure that a key function, when applied to values, does not change the shape.
    If values are part of a multi-index, it applies the key function to the specified levels.
    Parameters
    ----------
    values : Series, DataFrame, Index subclass, or ndarray
    key : Optional[Callable], a function to apply to the values array
    levels : Optional[List], levels to apply the key function to in case of MultiIndex
    Returns
    -------
    ArrayLike | Index | Series: Transformed values after applying the key function.
    """
    from pandas.core.indexes.api import Index
    
    # If no key function is provided, return the original values
    if not key:
        return values

    # Handle the case of MultiIndex separately
    if isinstance(values, ABCMultiIndex):
        return _ensure_key_mapped_multiindex(values, key, level=levels)

    # Apply the key function to the copied values and validate the result
    result = apply_key_function(values, key)
    validate_key_result(values, result)

    # Convert the result to the type of the original values or raise TypeError
    return convert_result_to_original_type(values, result)


def apply_key_function(values, key):
    """Apply the key function to the values and return the result."""
    return key(values.copy())


def validate_key_result(values, result):
    """
    Ensure that applying the key function does not change the shape of the values.
    Raise a ValueError if the shape is altered.
    """
    if len(result) != len(values):
        raise ValueError(
            "User-provided `key` function must not change the shape of the array."
        )


def convert_result_to_original_type(values, result):
    """
    Convert the result into the original type of values, handling Index and other types.
    Raise a TypeError if conversion is not possible.
    """
    from pandas.core.indexes.api import Index

    try:
        # Create a new Index if the original values were Index
        if isinstance(values, Index):
            return Index(result)
        else:
            # Try to convert back to the original type
            return type(values)(result)
    except TypeError:
        # The result type is not compatible with the original values type
        raise TypeError(
            f"User-provided `key` function returned an invalid type {type(result)} "
            f"which could not be converted to {type(values)}."
        )



def get_flattened_list(
    comp_ids: npt.NDArray[np.intp],
    ngroups: int,
    levels: Iterable[Index],
    labels: Iterable[np.ndarray],
) -> list[tuple]:
    """
    Convert a list of compressed group IDs to a list of key tuples 
    using corresponding levels and labels.
    
    :param comp_ids: A numpy array of compressed IDs
    :param ngroups: Number of distinct groups
    :param levels: An iterable of Index levels
    :param labels: An iterable of numpy arrays for levels' labels
    :return: A list of tuples representing the keys for each group
    """
    
    # Convert the comp_ids to an int64 data type without copying the array
    comp_ids = comp_ids.astype(np.int64, copy=False)
    
    # Dictionary to map group indices to corresponding values
    arrays: DefaultDict[int, list[int]] = defaultdict(list)
    
    for labs, level in zip(labels, levels):
        arrays = _map_labels_to_level(arrays, labs, level, comp_ids, ngroups)
    
    # Convert the dictionary values to a list of tuples for the keys
    return [tuple(array) for array in arrays.values()]

def _map_labels_to_level(arrays, labs, level, comp_ids, ngroups):
    """
    Helper function to map labels to the corresponding level based on
    the compressed group IDs.
    
    :param arrays: The current mapping of group indices to values
    :param labs: The labels of the current level
    :param level: The current Index level
    :param comp_ids: A numpy array of compressed IDs
    :param ngroups: The total number of groups
    :return: An updated arrays dictionary containing the mapped values
    """
    
    # Initialize a hash table for mapping groups to labels
    table = hashtable.Int64HashTable(ngroups)

    # Map compressed IDs to their corresponding labels
    table.map_keys_to_values(comp_ids, labs.astype(np.int64, copy=False))
    
    # Update the arrays dictionary with values from the current level
    for group_index in range(ngroups):
        level_value = level[table.get_item(group_index)]
        arrays[group_index].append(level_value)
    
    return arrays



def get_indexer_dict(
    labels: list[np.ndarray], keys: list[Index]
) -> dict[Hashable, npt.NDArray[np.intp]]:
    """
    Obtain a dictionary mapping labels to indexers for given keys.

    Parameters
    ----------
    labels : list of numpy.ndarray
        A list of label arrays.
    keys : list of Index
        A list of Index objects.

    Returns
    -------
    dict
        A dictionary where the keys are from `labels` and the values are
        index arrays corresponding to the indexer position.
    """
    # Compute the shape based on the lengths of the provided keys
    shape: Tuple[int, ...] = tuple(len(x) for x in keys)

    # Create the group index array
    index_array: np.ndarray = get_group_index(labels, shape, sort=True, xnull=True)
    
    # Check if the entire index_array is marked as -1 indicating missing values
    if np.all(index_array == -1):
        # If so, there is nothing to index, return an empty dictionary
        return {}

    # Calculate the number of groups
    ngroups: int = _calculate_ngroups(index_array, shape)

    # Get the sorter array that will sort labels and the index array
    sorter: np.ndarray = get_group_index_sorter(index_array, ngroups)

    # Sort all labels using the sorter
    sorted_labels: list[np.ndarray] = [lab.take(sorter) for lab in labels]

    # Sort the index array itself
    index_array: np.ndarray = index_array.take(sorter)

    # Generate the fast indices using a supporting library function
    return lib.indices_fast(sorter, index_array, keys, sorted_labels)


def _calculate_ngroups(index_array: np.ndarray, shape: tuple[int, ...]) -> int:
    """
    Calculate the number of groups for an index array.

    Parameters
    ----------
    index_array : numpy.ndarray
        The index array for which to calculate the number of groups.
    shape : Tuple[int, ...]
        The shape tuple representing dimensions for each level.

    Returns
    -------
    int
        The calculated number of groups.
    """
    # Calculate ngroups based on possible overflow scenario
    if is_int64_overflow_possible(shape):
        ngroups: int = (index_array.size and index_array.max()) + 1
    else:
        ngroups: int = np.prod(shape, dtype="i8")
    
    return ngroups



# ----------------------------------------------------------------------
# sorting levels...cleverly?


def get_group_index_sorter(
    group_index: npt.NDArray[np.intp], ngroups: int | None = None
) -> npt.NDArray[np.intp]:
    """
    Get a sorter array that can be used to sort the group_index.

    This function decides whether to use a group sort algorithm based on the
    size of the group_index and a cost comparison between the group sort and 
    argsort (mergesort). It is optimized for performance in multi-key groupby 
    operations.

    Parameters
    ----------
    group_index : np.ndarray[np.intp]
        Array containing indices of groups.
    ngroups : int, optional
        Number of groups. If None, it will be calculated as one plus the maximum value in group_index.

    Returns
    -------
    np.ndarray[np.intp]
        Array of indices that sorts the group_index array.
    """
    # Determine the number of groups if not provided
    if ngroups is None:
        ngroups = 1 + group_index.max()

    # Compute the length of the group_index array
    count = len(group_index)

    # Threshold constants for deciding between sorting methods
    threshold_alpha = 0.0
    threshold_beta = 1.0

    # Calculate if group sorting should be applied based on cost comparison
    use_groupsort = should_use_groupsort(count, ngroups, threshold_alpha, threshold_beta)

    # Perform the appropriate sorting method
    if use_groupsort:
        sorter, _ = algos.groupsort_indexer(
            ensure_platform_int(group_index), ngroups)
        # Ensuring the sorter array has the correct data type (platform-specific integer)
    else:
        sorter = group_index.argsort(kind="mergesort")
        # A stable sort is important to maintain the order of same-key items

    # Return the sorter array, ensuring it has the correct data type
    return ensure_platform_int(sorter)

def should_use_groupsort(count, ngroups, alpha, beta):
    """
    Determine whether to use group sort based on count of items and number of groups.

    We compare the complexity of the group sort to the complexity of using mergesort
    on the group_index array. This is a simplistic heuristic based on the count and
    ngroups.

    Parameters
    ----------
    count : int
        The count of items in the group_index array.
    ngroups : int
        The number of groups.
    alpha : float
        Threshold constant for count in the complexity comparison.
    beta : float
        Threshold constant for ngroups in the complexity comparison.

    Returns
    -------
    bool
        True if group sort should be used, False otherwise.
    """
    # Consider group sort if we have at least one item to sort
    if count > 0:
        # Compare the complexity heuristics to decide on the sorting method
        complexity_comparison = (alpha + beta * ngroups) < (count * np.log(count))
        return complexity_comparison
    else:
        # Do not use group sort if there's nothing to sort
        return False



def compress_group_index(
    group_index: npt.NDArray[np.int64], sort: bool = True
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """
    Compresses a group index array into a tuple of compressed IDs and observed group IDs.

    Args:
    - group_index: An array of integer group identifiers, which can be large.
    - sort: A boolean indicating whether the output should be sorted.

    Returns:
    - A tuple with two elements:
        - The compressed IDs corresponding to unique elements in `group_index`.
        - The unique observed group IDs from `group_index`.
    """
    if is_sorted(group_index):
        return compress_sorted_group_index(group_index)
    else:
        return compress_unsorted_group_index(group_index, sort)


def is_sorted(group_index: npt.NDArray[np.int64]) -> bool:
    """Check if the group_index array is already sorted in a non-descending order."""
    return np.all(group_index[1:] >= group_index[:-1])


def compress_sorted_group_index(group_index: npt.NDArray[np.int64]) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """Compress a sorted group index array."""
    # Mark the beginning of a new group
    unique_mask = np.concatenate([group_index[:1] > -1, group_index[1:] != group_index[:-1]])
    # Cumulative sum assigns a new ID to each unique element
    comp_ids = unique_mask.cumsum() - 1
    # Extract the original IDs of the unique elements
    obs_group_ids = group_index[unique_mask]
    return ensure_int64(comp_ids), ensure_int64(obs_group_ids)


def compress_unsorted_group_index(group_index: npt.NDArray[np.int64], sort: bool) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """Compress an unsorted group index array, optionally sorting the output."""
    table = hashtable.Int64HashTable(len(group_index))
    group_index = ensure_int64(group_index)
    comp_ids, obs_group_ids = table.get_labels_groupby(group_index)

    if sort and obs_group_ids.size > 0:
        obs_group_ids, comp_ids = _reorder_by_uniques(obs_group_ids, comp_ids)

    return ensure_int64(comp_ids), ensure_int64(obs_group_ids)



def _reorder_by_uniques(
    uniques: npt.NDArray[np.int64], labels: npt.NDArray[np.intp]
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.intp]]:
    """
    Reorders `uniques` and `labels` based on the sorted order of `uniques`.

    Parameters
    ----------
    uniques : np.ndarray[np.int64]
        Array of unique integers to be sorted.
    labels : np.ndarray[np.intp]
        Array of labels corresponding to the uniques array.

    Returns
    -------
    tuple[np.ndarray[np.int64], np.ndarray[np.intp]]
        A tuple containing sorted `uniques` array and reordered `labels` array such that the labels follow the sorted order of uniques.
    """
    # Sort the uniques array and get the sorted indices.
    sorter = uniques.argsort()
    
    # Create an array that will map the sorted indices back to the original indices.
    reverse_mapper = np.empty_like(sorter)
    reverse_mapper[sorter] = np.arange(len(sorter))
    
    # Identify labels that are negative and should not be moved.
    negative_label_mask = labels < 0
    
    # Use the reverse mapper to reorder the labels array.
    reordered_labels = reverse_mapper[labels]
    
    # Set negative labels back to -1, preserving their original value after reordering.
    reordered_labels[negative_label_mask] = -1
    
    # Take the sorted unique values according to the sorter indices.
    sorted_uniques = uniques[sorter]

    return sorted_uniques, reordered_labels

