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
    Helper method that returns the indexer according to input parameters
    for the sort_index method of DataFrame and Series.

    Parameters
    ----------
    target : Index
        The target index.
    level : int or level name or list of ints or list of level names
        The level to sort on.
    ascending : bool or list of bools, default True
        The sort order.
    kind : {'quicksort', 'mergesort', 'heapsort', 'stable'}
        The sorting algorithm to use.
    na_position : {'first', 'last'}
        The position of NaN values in the sorting.
    sort_remaining : bool
        Whether to sort the remaining levels.
    key : callable, optional
        The key function to apply before sorting.

    Returns
    -------
    Optional[ndarray[intp]]
        The indexer for the new index.
    """

    # Ensure the target is mapped using the given key function
    target = ensure_key_mapped(target, key, levels=level)  # type: ignore[assignment]

    # Sort the levels of the target index
    target = target._sort_levels_monotonic()

    if level is not None:
        # Sort by the specified level
        _, indexer = target.sortlevel(
            level,
            ascending=ascending,
            sort_remaining=sort_remaining,
            na_position=na_position,
        )
    elif _is_monotonic(target, ascending):
        # If the target is already monotonic, return None
        return None
    elif isinstance(target, ABCMultiIndex):
        # Sort a MultiIndex using lexsort_indexer
        codes = [lev.codes for lev in target._get_codes_for_sorting()]
        indexer = lexsort_indexer(
            codes, orders=ascending, na_position=na_position, codes_given=True
        )
    else:
        # Sort a regular Index using nargsort
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



import numpy as np
from numpy import ndarray
from typing import List, Tuple

def get_group_index(labels: List[np.ndarray], shape: Tuple[int, ...], sort: bool, xnull: bool) -> ndarray:
    """
    For the particular label_list, gets the offsets into the hypothetical list
    representing the totally ordered cartesian product of all possible label
    combinations, *as long as* this space fits within int64 bounds;
    otherwise, though group indices identify unique combinations of
    labels, they cannot be deconstructed.
    - If `sort`, rank of returned ids preserve lexical ranks of labels.
      i.e. returned id's can be used to do lexical sort on labels;
    - If `xnull` nulls (-1 labels) are passed through.

    Parameters
    ----------
    labels : sequence of arrays
        Integers identifying levels at each location
    shape : tuple[int, ...]
        Number of unique levels at each location
    sort : bool
        If the ranks of returned ids should match lexical ranks of labels
    xnull : bool
        If true nulls are excluded. i.e. -1 values in the labels are
        passed through.

    Returns
    -------
    An array of type int64 where two elements are equal if their corresponding
    labels are equal at all location.

    Notes
    -----
    The length of `labels` and `shape` must be identical.
    """

    def _int64_cut_off(shape: Tuple[int, ...]) -> int:
        """
        Calculate the cutoff point within the shape at which int64 bounds will be exceeded.

        Parameters
        ----------
        shape : tuple
            The shape to be evaluated

        Returns
        -------
        int
            The index point at which int64 bounds will be exceeded.
        """
        acc = 1
        for i, mul in enumerate(shape):
            acc *= int(mul)
            if not acc < lib.i8max:
                return i
        return len(shape)

    def maybe_lift(labels_array: np.ndarray, array_size: int) -> Tuple[np.ndarray, int]:
        """
        Promote nan values (assigned -1 label in lab array) so that all
        output values are non-negative.

        Parameters
        ----------
        labels_array : np.ndarray
            The labels array to be evaluated
        array_size : int
            The size of the labels array

        Returns
        -------
        tuple
            Tuple of the promoted labels array and the size
        """
        if (labels_array == -1).any():
            return (labels_array + 1), (array_size + 1)
        return labels_array, array_size

    def ensure_int64(array: np.ndarray) -> np.ndarray:
        """
        Ensures array is of type int64

        Parameters
        ----------
        array : np.ndarray
            Array to evaluate type

        Returns
        -------
        np.ndarray
            Array of type int64
        """
        if array.dtype != np.int64:
            return array.astype(np.int64)
        return array

    labels = [ensure_int64(x) for x in labels]
    lshape = list(shape)
    if not xnull:
        for i, (labels_array, array_size) in enumerate(zip(labels, shape)):
            labels[i], lshape[i] = maybe_lift(labels_array, array_size)

    labels = list(labels)

    # Iteratively process all the labels in chunks sized so less
    # than lib.i8max unique int ids will be required for each chunk
    while True:
        # how many levels can be done without overflow:
        nlev = _int64_cut_off(lshape)

        # compute flat ids for the first `nlev` levels
        stride = np.prod(lshape[1:nlev], dtype="i8")
        out = stride * labels[0].astype("i8", subok=False, copy=False)

        for i in range(1, nlev):
            if lshape[i] == 0:
                stride = np.int64(0)
            else:
                stride //= lshape[i]
            out += labels[i] * stride

        if xnull:  # exclude nulls
            mask = labels[0] == -1
            for labels_array in labels[1:nlev]:
                mask |= labels_array == -1
            out[mask] = -1

        if nlev == len(lshape):  # all levels done!
            break

        # compress what has been done so far in order to avoid overflow
        # to retain lexical ranks, obs_ids should be sorted
        comp_ids, obs_ids = compress_group_index(out, sort=sort)

        labels = [comp_ids] + labels[nlev:]
        lshape = [len(obs_ids)] + lshape[nlev:]

    return out




def get_compressed_ids(labels, sizes: Shape) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.int64]]:
    """
    Return compressed ids and group ids.

    Parameters
    ----------
    labels : list
        List of label arrays.
    sizes : tuple[int]
        Size of the levels.

    Returns
    -------
    tuple[npt.NDArray[np.intp], npt.NDArray[np.int64]]
        Compressed ids and group ids.
    """
    ids = get_group_index(labels, sizes, sort=True, xnull=False)
    return compress_group_index(ids, sort=True)



def is_int64_overflow_possible(shape: Shape) -> bool:
    the_prod = 1
    for x in shape:
        the_prod *= int(x)

    return the_prod >= lib.i8max


def _decons_group_index(comp_labels: np.ndarray, shape: Shape) -> List[np.ndarray]:
    """
    Deconstruct group indices into labels.

    Args:
        comp_labels (np.ndarray): Component labels of the group indices.
        shape (Shape): Shape of the indices.

    Raises:
        ValueError: If factorized group indices cannot be deconstructed.

    Returns:
        List[np.ndarray]: List of labels.
    """

    if is_int64_overflow_possible(shape):
        raise ValueError("Cannot deconstruct factorized group indices!")

    label_list = []
    factor = 1
    y = np.array(0)
    x = comp_labels

    for i in reversed(range(len(shape))):
        # Get labels based on deconstructing
        labels = (x - y) % (factor * shape[i]) // factor
        # Replace invalid labels with -1
        np.putmask(labels, comp_labels < 0, -1)
        # Add labels to label_list
        label_list.append(labels)
        # Update y for next iteration
        y = labels * factor
        # Update factor for next iteration
        factor *= shape[i]

    return label_list[::-1]




def decons_obs_group_ids(
    comp_ids: npt.NDArray[np.intp],
    obs_ids: npt.NDArray[np.intp],
    shape: Shape,
    labels: Sequence[npt.NDArray[np.signedinteger]],
    xnull: bool,
) -> list[npt.NDArray[np.intp]]:
    """
    Reconstruct labels from observed group ids.

    Parameters
    ----------
    comp_ids : np.ndarray[np.intp]
    obs_ids: np.ndarray[np.intp]
    shape : tuple[int]
    labels : Sequence[np.ndarray[np.signedinteger]]
    xnull : bool
        If nulls are excluded; i.e. -1 labels are passed through.
    """
    if not xnull:
        lift = np.fromiter(((a == -1).any() for a in labels), dtype=np.intp)
        arr_shape = np.asarray(shape, dtype=np.intp) + lift
        shape = tuple(arr_shape)

    # Check if int64 overflow is possible
    if not is_int64_overflow_possible(shape):
        # obs ids are deconstructable! take the fast route!
        out = _decons_group_index(obs_ids, shape)
        if xnull or not lift.any():
            return out
        else:
            # Subtract lift from each value in out
            return [x - y for x, y in zip(out, lift)]
    else:
        # Create an indexer using unique label indices
        indexer = unique_label_indices(comp_ids)
        # Return the labels after applying the indexer
        return [lab[indexer].astype(np.intp, subok=False, copy=True) for lab in labels]




def lexsort_indexer(
    keys: Sequence[ArrayLike | Index | Series],
    orders=None,
    na_position: str = "last",
    key: Callable | None = None,
    codes_given: bool = False,
) -> npt.NDArray[np.intp]:
    """
    Performs lexical sorting on a set of keys

    Parameters
    ----------
    keys : Sequence[ArrayLike | Index | Series]
        Sequence of arrays to be sorted by the indexer
        Sequence[Series] is only if key is not None.
    orders : bool or list of booleans, optional
        Determines the sorting order for each element in keys. If a list,
        it must be the same length as keys. This determines whether the
        corresponding element in keys should be sorted in ascending
        (True) or descending (False) order. if bool, applied to all
        elements as above. if None, defaults to True.
    na_position : {'first', 'last'}, default 'last'
        Determines placement of NA elements in the sorted list ("last" or "first")
    key : Callable, optional
        Callable key function applied to every element in keys before sorting
    codes_given: bool, False
        Avoid categorical materialization if codes are already provided.

    Returns
    -------
    np.ndarray[np.intp]
    """
    from pandas.core.arrays import Categorical

    if na_position not in ["last", "first"]:
        raise ValueError(f"invalid na_position: {na_position}")

    if isinstance(orders, bool):
        orders = [orders] * len(keys)
    elif orders is None:
        orders = [True] * len(keys)

    # Convert keys to a list of labels
    labels = []
    for k, order in zip(keys, orders):
        mapped_key = ensure_key_mapped(k, key)
        if codes_given:
            codes = cast(np.ndarray, mapped_key)
            n = codes.max() + 1 if len(codes) else 0
        else:
            cat = Categorical(mapped_key, ordered=True)
            codes = cat.codes
            n = len(cat.categories)

        mask = codes == -1

        # Modify codes depending on the value of na_position
        if na_position == "last" and mask.any():
            codes = np.where(mask, n, codes)
        
        modified_codes = np.where(mask, codes, n - codes - 1)
        
        labels.append(modified_codes)
    
    # Reverse the list of labels
    reversed_labels = labels[::-1]
    
    return np.lexsort(reversed_labels)




def nargsort(
    items: ArrayLike | Index | Series,
    kind: SortKind = "quicksort",
    ascending: bool = True,
    na_position: str = "last",
    key: Callable | None = None,
    mask: npt.NDArray[np.bool_] | None = None,
) -> npt.NDArray[np.intp]:
    """
    Intended to be a drop-in replacement for np.argsort which handles NaNs.

    Adds ascending, na_position, and key parameters.

    (GH #6399, #5231, #27237)

    Parameters
    ----------
    items : np.ndarray, ExtensionArray, Index, or Series
    kind : {'quicksort', 'mergesort', 'heapsort', 'stable'}, default 'quicksort'
    ascending : bool, default True
    na_position : {'first', 'last'}, default 'last'
    key : Optional[Callable], default None
    mask : Optional[np.ndarray[bool]], default None
        Passed when called by ExtensionArray.argsort.

    Returns
    -------
    np.ndarray[np.intp]
    """

    if key is not None:
        # see TestDataFrameSortKey, TestRangeIndex::test_sort_values_key
        items = ensure_key_mapped(items, key)
        return nargsort(
            items,
            kind=kind,
            ascending=ascending,
            na_position=na_position,
            key=None,
            mask=mask,
        )

    if isinstance(items, ABCRangeIndex):
        return items.argsort(ascending=ascending)
    elif not isinstance(items, ABCMultiIndex):
        items = extract_array(items)
    else:
        raise TypeError(
            "nargsort does not support MultiIndex. Use index.sort_values instead."
        )

    if mask is None:
        mask = np.asarray(isna(items))

    if not isinstance(items, np.ndarray):
        # i.e. ExtensionArray
        return items.argsort(
            ascending=ascending,
            kind=kind,
            na_position=na_position,
        )

    idx = np.arange(len(items))
    non_nans = items[~mask]
    non_nan_idx = idx[~mask]

    nan_idx = np.nonzero(mask)[0]
    if not ascending:
        non_nans = non_nans[::-1]
        non_nan_idx = non_nan_idx[::-1]
    indexer = non_nan_idx[non_nans.argsort(kind=kind)]
    if not ascending:
        indexer = indexer[::-1]
    # Finally, place the NaNs at the end or the beginning according to
    # na_position
    if na_position == "last":
        indexer = np.concatenate([indexer, nan_idx])
    elif na_position == "first":
        indexer = np.concatenate([nan_idx, indexer])
    else:
        raise ValueError(f"invalid na_position: {na_position}")
    return ensure_platform_int(indexer)



def nargminmax(values: ExtensionArray, method: str, axis: AxisInt = 0):
    """
    Implementation of np.argmin/argmax but for ExtensionArray and which
    handles missing values.

    Parameters
    ----------
    values : ExtensionArray
    method : {"argmax", "argmin"}
    axis : int, default 0

    Returns
    -------
    int
    """
    assert method in {"argmax", "argmin"}

    # Set the appropriate function based on the method parameter
    func = np.argmax if method == "argmax" else np.argmin

    # Check the dimensions of the values array
    arr_values = values._values_for_argsort()
    mask = np.asarray(isna(values))

    if arr_values.ndim > 1:
        if mask.any():
            if axis == 1:
                zipped = zip(arr_values, mask)
            else:
                zipped = zip(arr_values.T, mask.T)
            # Apply _nanargminmax function to each pair of values and masks
            return np.array([_nanargminmax(v, m, func) for v, m in zipped])
        else:
            # Apply func along the axis
            return func(arr_values, axis=axis)

    # Apply the _nanargminmax function to the single pair of values and mask
    return _nanargminmax(arr_values, mask, func)




def _nanargminmax(values: np.ndarray, mask: npt.NDArray[np.bool_], func) -> int:
    """
    Returns the index of the minimum or maximum value in an array, ignoring any NaN values.

    Args:
        values: numpy array of values.
        mask: boolean mask indicating which values are NaN.
        func: function to calculate the minimum or maximum value, such as `np.nanargmin` or `np.nanargmax`.

    Returns:
        The index of the minimum or maximum value in the array, after ignoring any NaN values.
    """
    idx = np.arange(values.shape[0])
    non_nan_values = values[~mask]
    non_nan_indices = idx[~mask]

    return non_nan_indices[func(non_nan_values)]




def _ensure_key_mapped_multiindex(index: MultiIndex, key: Callable, level=None) -> MultiIndex:
    """
    Returns a new MultiIndex in which key has been applied
    to all levels specified in level (or all levels if level
    is None). Used for key sorting for MultiIndex.

    Parameters
    ----------
    index : MultiIndex
        Index to which to apply the key function on the
        specified levels.
    key : Callable
        Function that takes an Index and returns an Index of
        the same shape. This key is applied to each level
        separately. The name of the level can be used to
        distinguish different levels for application.
    level : list-like, int or str, default None
        Level or list of levels to apply the key function to.
        If None, key function is applied to all levels. Other
        levels are left unchanged.

    Returns
    -------
    labels : MultiIndex
        Resulting MultiIndex with modified levels.
    """

    # Determine the levels to apply the key function to
    sort_levels = determine_sort_levels(level, index)  

    # Apply the key function to the specified levels
    mapped = apply_key_function_to_levels(sort_levels, index, key)

    # Return a new MultiIndex with modified levels
    return create_new_multiindex(mapped, index)

def determine_sort_levels(level, index):
    if level is not None:
        if isinstance(level, (str, int)):
            return [level]
        else:
            return level
    else:
        return list(range(index.nlevels))  # satisfies mypy

def apply_key_function_to_levels(sort_levels, index, key):
    return [
        ensure_key_mapped(
            index._get_level_values(level), key
        )
        if level in sort_levels
        else index._get_level_values(level)
        for level in range(index.nlevels)
    ]

def create_new_multiindex(mapped, index):
    return type(index).from_arrays(mapped)




def ensure_key_mapped(values: Union[ArrayLike, Index, Series], key: Optional[Callable], levels: Optional[List] = None) -> Union[ArrayLike, Index, Series]:
    """
    Applies a callable key function to the values function and checks
    that the resulting value has the same shape. Can be called on Index
    subclasses, Series, DataFrames, or ndarrays.

    Parameters
    ----------
    values : Series, DataFrame, Index subclass, or ndarray
    key : Optional[Callable], key to be called on the values array
    levels : Optional[List], if values is a MultiIndex, list of levels to
    apply the key to.
    """
    from pandas.core.indexes.api import Index

    if not key:
        return values

    if isinstance(values, ABCMultiIndex):
        return _ensure_key_mapped_multiindex(values, key, level=levels)

    result = key(values.copy())
    
    # Check if the resulting value has the same shape as the original value
    if len(result) != len(values):
        raise ValueError(
            "User-provided `key` function must not change the shape of the array."
        )

    try:
        if isinstance(values, Index):
            # Convert to a new Index subclass, not necessarily the same
            result = Index(result)
        else:
            # Try to revert to original type otherwise
            type_of_values = type(values)
            result = type_of_values(result)
    except TypeError:
        raise TypeError(
            f"User-provided `key` function returned an invalid type {type(result)} \
            which could not be converted to {type(values)}."
        )

    return result




from collections import defaultdict
from typing import DefaultDict, Iterable, List, Tuple
import numpy as np
import numpy.typing as npt
from hashtable import Int64HashTable


def get_flattened_list(
    comp_ids: npt.NDArray[np.intp],
    ngroups: int,
    levels: Iterable[Index],
    labels: Iterable[np.ndarray],
) -> List[Tuple]:
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
    arrays: DefaultDict[int, List[int]] = defaultdict(list)

    # Mapping the compressed group ids to key tuples
    for labs, level in zip(labels, levels):
        table = Int64HashTable(ngroups)
        table.map_keys_to_values(comp_ids, labs.astype(np.int64, copy=False))
        for i in range(ngroups):
            arrays[i].append(level[table.get_item(i)])
    
    # Converting the arrays dictionary to a list of tuples
    return [tuple(array) for array in arrays.values()]




def get_indexer_dict(label_list: list[np.ndarray], keys: list[Index]) -> dict[Hashable, npt.NDArray[np.intp]]:
    """
    Returns a dictionary mapping labels to indexers.
    """
    # Calculate the shape of the indexers
    shape = tuple(len(x) for x in keys)
    
    # Get the group index
    group_index = get_group_index(label_list, shape, sort=True, xnull=True)
    
    # If all elements are -1, return an empty dictionary
    if np.all(group_index == -1):
        return {}

    # Calculate the number of groups
    ngroups = (
        ((group_index.size and group_index.max()) + 1)
        if is_int64_overflow_possible(shape)
        else np.prod(shape, dtype="i8")
    )

    # Sort the labels and group index using the sorter
    sorter = get_group_index_sorter(group_index, ngroups)
    sorted_labels = [lab.take(sorter) for lab in label_list]
    group_index = group_index.take(sorter)

    # Return the result from lib.indices_fast
    return lib.indices_fast(sorter, group_index, keys, sorted_labels)




# ----------------------------------------------------------------------
# sorting levels...cleverly?


import numpy as np


def get_group_index_sorter(group_index: np.ndarray[np.intp], ngroups: int | None = None) -> np.ndarray[np.intp]:
    """
    Implements `counting sort` and `np.argsort(kind='mergesort')` algorithms for sorting group indices.
    Both algorithms are stable sorts, necessary for correctness of groupby operations.

    Parameters
    ----------
    group_index : np.ndarray[np.intp]
        The group index to be sorted.
    ngroups : int or None, default None
        The number of groups. If None, it is calculated as 1 + group_index.max().

    Returns
    -------
    np.ndarray[np.intp]
        The sorted group index.
    """
    if ngroups is None:
        ngroups = 1 + group_index.max()

    count = len(group_index)
    alpha = 0.0  # taking complexities literally; there may be
    beta = 1.0   # some room for fine-tuning these parameters

    # Check if counting sort should be used
    do_groupsort = count > 0 and ((alpha + beta * ngroups) < (count * np.log(count)))

    if do_groupsort:
        sorter, _ = algos.groupsort_indexer(
            ensure_platform_int(group_index),
            ngroups
        )
    else:
        # Use np.argsort with mergesort if counting sort is not required
        sorter = group_index.argsort(kind="mergesort")

    return ensure_platform_int(sorter)



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
    if len(group_index) > 0 and np.all(group_index[1:] >= group_index[:-1]):
        # Fast path for sorted group_index
        unique_mask = np.concatenate([group_index[:1] > -1, group_index[1:] != group_index[:-1]])
        comp_ids = unique_mask.cumsum() - 1
        obs_group_ids = group_index[unique_mask]
        
    else:
        size_hint = len(group_index)
        table = hashtable.Int64HashTable(size_hint)

        group_index = ensure_int64(group_index)

        # Group labels come out ascending (i.e., 1, 2, 3 etc)
        comp_ids, obs_group_ids = table.get_labels_groupby(group_index)

        if sort and len(obs_group_ids) > 0:
            obs_group_ids, comp_ids = _reorder_by_uniques(obs_group_ids, comp_ids)

    return ensure_int64(comp_ids), ensure_int64(obs_group_ids)




def _reorder_by_uniques(
    uniques: npt.NDArray[np.int64], labels: npt.NDArray[np.intp]
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.intp]]:
    """
    Rearrange the array of labels based on the order of unique values.

    Parameters
    ----------
    uniques : np.ndarray[np.int64]
        The array of unique values.
    labels : np.ndarray[np.intp]
        The array of labels.

    Returns
    -------
    np.ndarray[np.int64]
        The rearranged array of unique values.
    np.ndarray[np.intp]
        The rearranged array of labels.
    """
    # Sort the unique values
    sorter = uniques.argsort()

    # Create reverse indexer to keep track of where elements came from
    reverse_indexer = np.empty(len(sorter), dtype=np.intp)
    reverse_indexer.put(sorter, np.arange(len(sorter)))

    # Create mask to identify negative labels
    mask = labels < 0

    # Rearrange labels based on the reverse indexer
    labels = reverse_indexer.take(labels)
    np.putmask(labels, mask, -1)

    # Rearrange the unique values
    uniques = uniques.take(sorter)

    return uniques, labels

