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


def get_indexer_indexer(
    target: Index,
    level: Level | list[Level] | None,
    ascending: list[bool] | bool,
    kind: SortKind,
    na_position: NaPosition,
    sort_remaining: bool,
    key: IndexKeyFunc,
) -> npt.NDArray[np.intp] | None:
    """
    This method returns the indexer according to input parameters for the sort_index
    method of DataFrame and Series.
    """

    # Ensure key is mapped correctly according to levels
    # type: ignore[assignment]
    target = ensure_key_mapped(target, key, levels=level)

    # Sort the levels of the target (if any) in a monotonic order
    target = target._sort_levels_monotonic()

    # If levels exist in target, sort the target using sortlevel method.
    if level is not None:
        _, indexer = target.sortlevel(
            level,
            ascending=ascending,
            sort_remaining=sort_remaining,
            na_position=na_position,
        )

    # If target is already sorted according to the 'ascending' order, no need
    # to do anything
    elif _is_monotonic(target, ascending):
        return None

    # If target is a MultiIndex, get codes for sorting and sort based on these
    # codes
    elif isinstance(target, ABCMultiIndex):
        codes = [lev.codes for lev in target._get_codes_for_sorting()]
        indexer = lexsort_indexer(
            codes, orders=ascending, na_position=na_position, codes_given=True
        )

    # In other cases (non-MultiIndex) sort based on the values in target
    else:
        indexer = nargsort(
            target,
            kind=kind,
            ascending=cast(bool, ascending),
            na_position=na_position,
        )

    return indexer


def _is_monotonic(target: Index, ascending: list[bool] | bool) -> bool:
    return (np.all(ascending) and target.is_monotonic_increasing) or (
        not np.any(ascending) and target.is_monotonic_decreasing
    )


def get_group_index(
    labels, shape: Shape, sort: bool, xnull: bool
) -> npt.NDArray[np.int64]:
    """
    Get the indices based on unique combinations of labels.

    Parameters
    ----------
    labels : sequence of arrays
        Integers identifying levels at each location
    shape : tuple[int, ...]
        Number of unique levels at each location
    sort : bool
        If true, the returned indices will match the lexical ranks of labels
    xnull : bool
        If true, nulls (-1 labels) will be excluded

    Returns
    -------
    An array of int64 type where two elements are equal if their corresponding
    labels are equal at all location.
    """

    def calculate_cutoff(shape) -> int:
        """
        Calculate the cutoff point for int64 bounds.

        Parameters
        ----------
        shape : tuple
            The shape to evaluate

        Returns
        -------
        int
            The index where the int64 bounds are exceeded.
        """
        product = 1
        for i, val in enumerate(shape):
            product *= int(val)
            if product >= lib.i8max:
                return i
        return len(shape)

    def promote_nan_values(
            labels_array, array_size: int) -> tuple[np.ndarray, int]:
        """
        Promote nan values.

        Parameters
        ----------
        labels_array : np.ndarray
            The labels array to be checked
        array_size : int
            Size of the labels array

        Returns
        -------
        tuple
            Labels array with promoted values and new size
        """
        if (labels_array == -1).any():
            return labels_array + 1, array_size + 1
        else:
            return labels_array, array_size

    def cast_to_int64(array) -> np.ndarray:
        """
        Cast array element's dtype to int64.

        Parameters
        ----------
        array : np.ndarray
            Array to cast

        Returns
        -------
        np.ndarray
            Array with int64 dtype.
        """
        if array.dtype != np.int64:
            return array.astype(np.int64)
        return array

    labels = [cast_to_int64(label) for label in labels]
    shape_list = list(shape)

    if not xnull:
        for idx, (label, size) in enumerate(zip(labels, shape)):
            labels[idx], shape_list[idx] = promote_nan_values(label, size)

    while True:
        cutoff = calculate_cutoff(shape_list)
        stride = np.prod(shape_list[1:cutoff], dtype="i8")
        output = stride * labels[0].astype("i8", subok=False, copy=False)

        for i in range(1, cutoff):
            stride = stride // shape_list[i] if shape_list[i] != 0 else np.int64(
                0)
            output += labels[i] * stride

        if xnull:  # Exclude nulls
            mask = labels[0] == -1
            for label in labels[1:cutoff]:
                mask |= label == -1
            output[mask] = -1

        if cutoff == len(shape_list):  # all levels done!
            break

        # Order ids to retain lexical ranks and avoid overflow
        compression_ids, obs_ids = compress_group_index(output, sort=sort)
        labels = [compression_ids] + labels[cutoff:]
        shape_list = [len(obs_ids)] + shape_list[cutoff:]

    return output


def get_compressed_ids(
    labels, sizes: Shape
) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.int64]]:
    """
    Group_index is offsets into cartesian product of all possible labels. This
    space can be huge, so this function compresses it, by computing offsets
    (comp_ids) into the list of unique labels (obs_group_ids).

    Parameters
    ----------
    labels : list of label arrays
    sizes : tuple[int] of size of the levels

    Returns
    -------
    np.ndarray[np.intp]
        comp_ids
    np.ndarray[np.int64]
        obs_group_ids
    """
    ids = get_group_index(labels, sizes, sort=True, xnull=False)
    return compress_group_index(ids, sort=True)


def is_int64_overflow_possible(shape: Shape) -> bool:
    the_prod = 1
    for x in shape:
        the_prod *= int(x)

    return the_prod >= lib.i8max


def _decons_group_index(
        comp_labels: npt.NDArray[np.intp], shape: Shape) -> list[npt.NDArray[np.intp]]:
    """Deconstruct group indices into labels.

    Args:
        comp_labels (npt.NDArray[np.intp]): Component labels of the group indices.
        shape (Shape): Shape of the indices.

    Raises:
        ValueError: If factorized group indices cannot be deconstructed.

    Returns:
        list[npt.NDArray[np.intp]]: List of labels.
    """

    # Check if an Integer Overflow is possible with given shape
    if is_int64_overflow_possible(shape):
        raise ValueError("cannot deconstruct factorized group indices!")

    label_list = []
    factor = 1
    y_value = np.array(0)
    x_value = comp_labels

    # Iterate over the range in reverse order
    for i in reversed(range(len(shape))):
        # Compute label, mask and append to list
        label, factor, y_value = compute_label_and_update_values(
            x_value, factor, y_value, shape[i], comp_labels)
        label_list.append(label)

    # Return the list in reverse order
    return label_list[::-1]


def compute_label_and_update_values(x_value: npt.NDArray[np.intp],
                                    factor: int,
                                    y_value: npt.NDArray[np.intp],
                                    shape_component: int,
                                    comp_labels: npt.NDArray[np.intp]) -> Tuple[npt.NDArray[np.intp],
                                                                                int,
                                                                                npt.NDArray[np.intp]]:
    """Compute the label and update factor and y values.

    Args:
        x_value (npt.NDArray[np.intp]): X value from the previous method.
        factor (int): Factor value from the previous method.
        y_value (npt.NDArray[np.intp]): Y value from the previous method.
        shape_component (int): A component of the shape variable.
        comp_labels (npt.NDArray[np.intp]): Component labels of the group indices.

    Returns:
        tuple: A tuple with updated label, factor and y value.
    """

    # Calculating label
    label = (x_value - y_value) % (factor * shape_component) // factor
    # put in mask where comp_labels < 0 is True
    np.putmask(label, comp_labels < 0, -1)

    # Updating y value with modified label and factor
    y_value = label * factor

    # Updating the factor
    factor *= shape_component

    # Return updated label, factor and y_value
    return label, factor, y_value


def decons_obs_group_ids(
    group_ids: npt.NDArray[np.intp],
    observed_ids: npt.NDArray[np.intp],
    array_shape: Shape,
    group_labels: Sequence[npt.NDArray[np.signedinteger]],
    exclude_null: bool,
) -> list[npt.NDArray[np.intp]]:
    """
    This method tries to reconstruct group labels from observed group ids.

    Parameters
    ----------
    group_ids : np.ndarray[np.intp]
        The ids of the groups.
    observed_ids: np.ndarray[np.intp]
        The ids that are observed from the groups.
    array_shape : tuple[int]
        The shape of the array that includes the group id.
    group_labels : Sequence[np.ndarray[np.signedinteger]]
        The labels associated with the groups.
    exclude_null : bool
        If True, exclude nulls; i.e., -1 labels are passed through. If False, include nulls.
    """

    # If nulls are not excluded, modify 'array_shape' by 'lifting' where
    # labels are -1
    if not exclude_null:
        lift = np.fromiter(((label == -1).any()
                           for label in group_labels), dtype=np.intp)
        updated_shape = np.asarray(array_shape, dtype=np.intp) + lift
        array_shape = tuple(updated_shape)

    # If the overflow is not possible, take the fast route of reconstructing
    # using '_decons_group_index'
    if not is_int64_overflow_possible(array_shape):
        group_index_reconstructed = _decons_group_index(
            observed_ids, array_shape)
        if exclude_null or not lift.any():
            return group_index_reconstructed
        else:
            return [
                index - lift_value for index,
                lift_value in zip(
                    group_index_reconstructed,
                    lift)]

    else:
        # If overflow is possible, use 'unique_label_indices' for indexing
        unique_indices = unique_label_indices(group_ids)
        return [
            label[unique_indices].astype(
                np.intp,
                subok=False,
                copy=True) for label in group_labels]


def lexsort_indexer(
    keys: Sequence[ArrayLike | Index | Series],
    orders=None,
    na_position: str = "last",
    key: Callable | None = None,
    codes_given: bool = False,
) -> npt.NDArray[np.intp]:
    """
    This function performs lexical sorting on a set of keys

    Parameters
    ----------
    keys : Sequence[ArrayLike | Index | Series]
        Sequence of Series/enumerable items to be sorted by the indexer
    orders : bool or list of booleans, optional
        Bool defining sorting order: True for ascending, False for descending
        Can also be applied to each key individually by providing a list
    na_position : {'first', 'last'}, default 'last'
        The placement of NA elements in the sorted keys. Either at the end ('last'), or at the start ('first').
    key : Callable, optional
        A function applied to every element in keys before sorting
    codes_given: bool, optional, default False
        Indicator to skip conversion of keys to categorical if codes are provided in advance

    Returns
    -------
    np.ndarray
        A numpy array providing the sorted indexer for keys
    """
    from pandas.core.arrays import Categorical

    # Ensure valid na_position value
    if na_position not in ["last", "first"]:
        raise ValueError(f"invalid na_position: {na_position}")

    orders = normalize_orders(keys, orders)

    labels = process_keys(keys, orders, key, codes_given, na_position)

    # Reverse the labels and use numpy's lexsort function to return the sorted
    # indexer
    return np.lexsort(labels[::-1])


def normalize_orders(keys, orders):
    """
    Check and normalize the orders parameter to
    have a definite format of a list with len matching keys []
    """
    if isinstance(orders, bool):
        return [orders] * len(keys)
    elif orders is None:
        return [True] * len(keys)
    else:
        return orders


def process_keys(keys, orders, key, codes_given, na_position):
    """
    Map and process the keys according to the sort orders,
    key function, na_position, and whether the ordered codes are already given.
    """
    from pandas.core.arrays import Categorical
    labels = []

    for key_elem, order in zip(keys, orders):
        key_elem = ensure_key_mapped(key_elem, key)

        if codes_given:
            ordered_codes = cast(np.ndarray, key_elem)
            code_length = ordered_codes.max() + 1 if len(ordered_codes) else 0
        else:
            cat = Categorical(key_elem, ordered=True)
            ordered_codes = cat.codes
            code_length = len(cat.categories)

        mask = ordered_codes == -1

        if na_position == "last" and mask.any():
            ordered_codes = np.where(mask, code_length, ordered_codes)

        # not order means descending
        if not order:
            ordered_codes = np.where(
                mask, ordered_codes, code_length - ordered_codes - 1)

        labels.append(ordered_codes)

    return labels


def nargsort(
    items,
    kind="quicksort",
    ascending=True,
    na_position="last",
    key=None,
    mask=None,
):
    """
    This function sorts given items while handling NaN values. It can sort in ascending or descending order,
    and it can place NaN values either at the beginning or at the end. This function is envisioned as a direct replacement for np.argsort.

    parameters:
    items: np.ndarray, ExtensionArray, Index, or Series
    kind : { 'quicksort', 'mergesort', 'heapsort', 'stable'}, default 'quicksort' -> sorting algorithm
    ascending : bool, default=True -> defines the sorting order
    na_position : {'first', 'last'}, default 'last' -> Position for NaN values
    key : Callable, default= None -> key to sort by
    mask: np.ndarray[bool], default= None -> mask to put on items.

    Returns:
    np.ndarray[int]: The sorted items as an array
    """

    # if a key is given map it to items
    if key:
        items = key_mapped_item(items, key)
        return nargsort(items, kind, ascending, na_position, None, mask)

    # if item is a range index, return sorted items. If it is not a
    # multi-index, convert to an array
    if isinstance(items, ABCRangeIndex):
        return items.argsort(ascending)
    elif not isinstance(items, ABCMultiIndex):
        items = array_from_items(items)

    # raise an error for multi-index
    else:
        raise TypeError(
            "nargsort does not support MultiIndex. Use index.sort_values instead.")

    # apply a mask for NaN values
    if mask is None:
        mask = create_nan_mask(items)

    # sort and return items if not an array
    if not isinstance(items, np.ndarray):
        return items.argsort(ascending, kind, na_position)

    idx = create_index(len(items))
    non_nans, non_nan_idx = get_non_nan_items(items, mask)

    # handle NaN values
    nan_idx = get_nan_idx(mask)
    if not ascending:
        non_nans, non_nan_idx = reverse_items(non_nans, non_nan_idx)

    indexer = get_indexer(non_nans, non_nan_idx, kind)

    # reverse the order if items are not in ascending order
    if not ascending:
        indexer = reverse_items(indexer)

    # place NaNs at the beginning or end based on na_position
    if na_position == "last":
        indexer = concatenate_indexes(indexer, nan_idx)
    elif na_position == "first":
        indexer = concatenate_indexes(nan_idx, indexer)
    else:
        raise ValueError(f"invalid na_position: {na_position}")
    return platform_int_conversion(indexer)


def nargminmax(values: ExtensionArray, method: str, axis: AxisInt = 0):
    """
    Implementation of np.argmin/argmax but for ExtensionArray and handles missing values.

    Parameters
    ----------
    values : ExtensionArray
        Input array of values to compute argmin/argmax.
    method : str
        Method to use for computation. Either {"argmax", "argmin"}.
    axis : int, default 0
        Axis along which to compute argmin/argmax.

    Returns
    -------
    int
        Index of minimum or maximum value, depending on the 'method' parameter.
    """
    # Verify the method parameter to ensure it's the correct value
    assert method in {"argmax", "argmin"}

    # Choose correct function (np.argmax or np.argmin) depending on the input
    # method
    minmax_func = np.argmax if method == "argmax" else np.argmin

    # Prepare the array values for argument sorting
    prepared_values = values._values_for_argsort()

    # Get a boolean mask indicating missing values in the input array
    missing_values_mask = np.asarray(isna(values))

    if prepared_values.ndim > 1:
        return handle_multidimensional_array(
            prepared_values, missing_values_mask, minmax_func, axis)
    else:
        return handle_single_dimension_array(
            prepared_values, missing_values_mask, minmax_func)


def handle_multidimensional_array(
        values,
        missing_values_mask,
        minmax_func,
        axis):
    """
    Handles the argmin/argmax computation for multi-dimensional arrays.
    """
    if missing_values_mask.any():
        return handle_missing_values(
            values, missing_values_mask, minmax_func, axis)
    else:
        # Directly apply the minmax function along the specified axis
        return minmax_func(values, axis=axis)


def handle_single_dimension_array(values, missing_values_mask, minmax_func):
    """
    Handles the argmin/argmax computation for single-dimensional arrays.
    """
    # Apply the _nanargminmax function to the values and mask
    return _nanargminmax(values, missing_values_mask, minmax_func)


def handle_missing_values(values, missing_values_mask, minmax_func, axis):
    """
    Handles the computation when missing values are present, returning the minimum/maximum index neglecting missing values.
    """
    values_mask_pairs = zip(
        values.T, missing_values_mask.T) if axis == 1 else zip(
        values, missing_values_mask)
    return np.array([_nanargminmax(val_mask[0], val_mask[1], minmax_func)
                    for val_mask in values_mask_pairs])


def _nanargminmax(values: np.ndarray,
                  mask: npt.NDArray[np.bool_],
                  func) -> int:
    """
    Returns the index of the minimum or maximum value in a numpy array, ignoring any NaN values.

    Args:
        values (np.ndarray): A numpy array containing numerical data.
        mask (npt.NDArray[np.bool_]): A boolean mask, in the same shape as 'values', where True indicates a NaN value in the 'values' array.
        func: A function used to find either the minimum or the maximum. This can be np.nanargmin or np.nanargmax.

    Returns:
        int: The index of the minimum or maximum value in the 'values' array, ignoring NaN values.
    """
    # Generate an array with elements from 0 up to the length of 'values'
    indices = np.arange(values.shape[0])

    # Remove NaNs from 'values' using the 'mask'
    non_nan_values = values[~mask]

    # Get the corresponding indices of the non-NaN values in 'values'
    non_nan_indices = indices[~mask]

    # Use the input function to find the index of the min/max non-NaN value
    return non_nan_indices[func(non_nan_values)]


def _ensure_key_mapped_multiindex(
        index: MultiIndex,
        key_function: Callable,
        level=None) -> MultiIndex:
    """
    Returns a new MultiIndex in which the key function has been applied
    to all levels specified in 'level' (or all levels if 'level' is None).
    Used for key sorting for MultiIndex.

    Parameters
    ----------
    index : MultiIndex. The index to which the key function is to be applied.
    key_function : Callable. The function that takes an Index and returns an Index of
                   the same shape. This key is applied to each level separately.
                   The name of the level can be used to distinguish different levels for application.
    level : list-like, int or str, default None. Level or list of levels to apply the key function to.
            If None, key function is applied to all levels. Other levels are left unchanged.

    Returns
    -------
    changed_multi_index : MultiIndex. The resulting MultiIndex with modified levels.
    """

    # Get levels to apply the key function to
    sort_levels = [level] if isinstance(
        level, (str, int)) else list(
        range(
            index.nlevels)) if level is None else level

    # Apply key function to specified levels
    mapped_indexes = [
        ensure_key_mapped(
            index._get_level_values(level),
            key_function) if level in sort_levels else index._get_level_values(level) for level in range(
            index.nlevels)]

    # Return a new MultiIndex with modified levels
    changed_multi_index = type(index).from_arrays(mapped_indexes)

    return changed_multi_index


def map_key_to_values(values, key, levels=None):
    """
    Apply a callable key function to the values and ensure that the resulting array has the same shape.

    Parameters:
    ----------
    values : array-like, Series, or Index
        Input values to apply the key function on.
    key : callable, optional
        The function to apply to the values. None means return original values.
    levels : list, default None
        If the values are a MultiIndex, specifies levels to apply key.

    Returns:
    -------
    array-like, Series, or Index:
        Resulting values after applying the key function.
    """

    if not key:
        return values

    if isinstance(values, ABCMultiIndex):
        return apply_key_to_multiindex(values, key, levels)

    result = key(values.copy())

    if len(result) != len(values):
        raise ValueError(
            "User-provided `key` function must not change array shape.")

    try:
        if isinstance(values, Index):
            result = Index(result)
        else:
            values_class = type(values)
            result = values_class(result)
    except TypeError:
        raise TypeError(
            f"User-provided `key` function returned incompatible type {type(result)}.\
                        Cannot convert to original type `{type(values)}`.")

    return result


def get_flattened_list(
    comp_ids: npt.NDArray[np.intp],
    ngroups: int,
    levels: Iterable[Index],
    labels: Iterable[np.ndarray],
) -> list[tuple]:
    """
    This function maps compressed group ids to key tuples.

    Parameters:
    comp_ids: An array containing the group ids.
    ngroups: The number of groups.
    levels: An iterable containing the levels.
    labels: An iterable containing the labels.

    Returns:
    A list of tuples representing the flattened list.
    """
    comp_ids = comp_ids.astype(np.int64, copy=False)
    arrays: DefaultDict[int, list[int]] = defaultdict(list)

    # Mapping the compressed group ids to key tuples
    for labs, level in zip(labels, levels):
        table = hashtable.Int64HashTable(ngroups)
        table.map_keys_to_values(comp_ids, labs.astype(np.int64, copy=False))
        for i in range(ngroups):
            arrays[i].append(level[table.get_item(i)])

    # Converting the arrays dictionary to a list of tuples
    return [tuple(array) for array in arrays.values()]


def get_indexer_dict(label_list: list[np.ndarray],
                     keys: list[Index]) -> dict[Hashable,
                                                npt.NDArray[np.intp]]:
    """
    This function returns a dictionary that maps labels to indexers.

    Parameters:
    label_list: list of labels
    keys: list of index

    Returns:
    dictionary: mapping labels to indexers
    """
    # calculating shape of indexers
    shape_of_indexers = tuple(len(key) for key in keys)

    # getting the group_index
    group_index_data = get_group_index(
        label_list, shape_of_indexers, sort=True, xnull=True)

    # if all elements are -1, return an empty dictionary
    if np.all(group_index_data == -1):
        return {}

    # checking if int64_overflow is possible and calculating number of groups
    # accordingly
    no_of_groups = (
        (group_index_data.size and group_index_data.max()) +
        1) if is_int64_overflow_possible(shape_of_indexers) else np.prod(
        shape_of_indexers,
        dtype="i8")

    # getting the sorter to sort the labels and group index
    sorter_value = get_group_index_sorter(group_index_data, no_of_groups)

    # sorting the labels and group index
    sorted_labels_list = [label.take(sorter_value) for label in label_list]
    sorted_group_index = group_index_data.take(sorter_value)

    # returning a dictionary mapping labels to indexers
    return lib.indices_fast(
        sorter_value,
        sorted_group_index,
        keys,
        sorted_labels_list)


# ----------------------------------------------------------------------
# sorting levels...cleverly?


def get_group_index_sorter(
        group_index: np.ndarray[np.intp], total_groups: int | None = None) -> np.ndarray[np.intp]:
    """
    Implements `counting sort` and `np.argsort(kind='mergesort')` algorithms for sorting group indices.
    Both algorithms are stable sorts, necessary for correctness of groupby operations.

    Parameters
    ----------
    group_index : np.ndarray[np.intp]
        The group index to be sorted.
    total_groups : int or None, default None
        The total number of groups. If None, it is calculated as 1 + group_index.max().

    Returns
    -------
    np.ndarray[np.intp]
        The sorted group index.
    """
    if total_groups is None:
        total_groups = 1 + group_index.max()

    index_length = len(group_index)
    alpha_value = 0.0  # taking complexities literally; there may be
    beta_value = 1.0   # some room for fine-tuning these parameters

    # Check if counting sort should be used
    use_counting_sort = index_length > 0 and (
        (alpha_value +
         beta_value *
         total_groups) < (
            index_length *
            np.log(index_length)))

    if use_counting_sort:
        # Use group sort if conditions meet
        sorted_index, _ = algos.groupsort_indexer(
            ensure_platform_int(group_index),
            total_groups
        )
    else:
        # Use np.argsort with mergesort if counting sort is not required
        sorted_index = group_index.argsort(kind="mergesort")

    return ensure_platform_int(sorted_index)


def _handle_sorted_group_index(group_index):
    """
    Handles sorted group index
    It identifies unique groups and calculates corresponding compact indices.
    :param group_index: Sorted group index
    :return: tuple of compact indices (comp_ids) and the list of unique labels (obs_group_ids)
    """
    unique_mask = np.concatenate(
        [group_index[:1] > -1, group_index[1:] != group_index[:-1]])
    comp_ids = unique_mask.cumsum() - 1
    obs_group_ids = group_index[unique_mask]
    return comp_ids, obs_group_ids


def compress_group_index(
    group_index: npt.NDArray[np.int64], sort: bool = True
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """
    Compresses the group_index by computing offsets (comp_ids) into the list of unique labels (obs_group_ids).

    Parameters:
    - group_index: The group index as offsets into cartesian product of all possible labels.
    - sort: Boolean indicating whether to sort the returned values or not.

    Returns:
    A tuple containing compressed comp_ids and obs_group_ids.
    """
    if len(group_index) and np.all(group_index[1:] >= group_index[:-1]):
        # Fast path for sorted group_index
        comp_ids, obs_group_ids = _handle_sorted_group_index(group_index)
    else:
        size_hint = len(group_index)
        table = hashtable.Int64HashTable(size_hint)
        group_index = ensure_int64(group_index)

        # Group labels come out ascending (i.e., 1, 2, 3 etc)
        comp_ids, obs_group_ids = table.get_labels_groupby(group_index)

        if sort and len(obs_group_ids) > 0:
            obs_group_ids, comp_ids = _reorder_by_uniques(
                obs_group_ids, comp_ids)

    return ensure_int64(comp_ids), ensure_int64(obs_group_ids)


def _reorder_by_uniques(uniques: npt.NDArray[np.int64],
                        labels: npt.NDArray[np.intp]) -> tuple[npt.NDArray[np.int64],
                                                               npt.NDArray[np.intp]]:
    """
    Rearrange array of labels based on order of unique values.

    Parameters
    ----------
    uniques : np.ndarray[np.int64]
        Array of unique values.
    labels : np.ndarray[np.intp]
        Array of labels

    Returns
    -------
    tuple
        rearranged array of unique values and array of labels
    """
    # Getting the indexes that would sort the unique values
    indexes_of_sorted_uniques = uniques.argsort()

    # Create new array to map uniques back to original order
    reverse_indexer = np.empty(len(indexes_of_sorted_uniques), dtype=np.intp)
    # Map each element to its index in the sorted list
    reverse_indexer.put(
        indexes_of_sorted_uniques, np.arange(
            len(indexes_of_sorted_uniques)))

    # Identify the indexes of negative labels
    negative_label_locations = labels < 0
    # Apply the new order (reverse_indexer) to labels while preserving
    # original locations of negative labels
    reordered_labels = reverse_indexer.take(labels)
    # Replace negative labels with -1 in the reordered labels
    np.putmask(reordered_labels, negative_label_locations, -1)

    # Reorder the unique values based on the indexes of sorted uniques
    reordered_uniques = uniques.take(indexes_of_sorted_uniques)

    return reordered_uniques, reordered_labels
