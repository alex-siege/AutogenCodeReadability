from __future__ import annotations

import functools
from typing import (
    TYPE_CHECKING,
    cast,
    overload,
)

import numpy as np

from pandas._libs import (
    algos as libalgos,
    lib,
)

from pandas.core.dtypes.cast import maybe_promote
from pandas.core.dtypes.common import (
    ensure_platform_int,
    is_1d_only_ea_dtype,
)
from pandas.core.dtypes.missing import na_value_for_dtype

from pandas.core.construction import ensure_wrapped_if_datetimelike

if TYPE_CHECKING:
    from pandas._typing import (
        ArrayLike,
        AxisInt,
        npt,
    )

    from pandas.core.arrays._mixins import NDArrayBackedExtensionArray
    from pandas.core.arrays.base import ExtensionArray


@overload
def take_nd(
    arr: np.ndarray,
    indexer,
    axis: AxisInt = ...,
    fill_value=...,
    allow_fill: bool = ...
) -> np.ndarray:
    """
    Takes values from an input array along the specified axis at specified indices or slices.

    Parameters:
        arr (np.ndarray): The input array.
        indexer : The indices or slices used to take values from the input array.
        axis (AxisInt): The axis along which to take values. Default is None.
        fill_value: The value used to fill in for invalid indices or slices. Default is None.
        allow_fill (bool): Whether to allow the fill_value to be used. Default is False.

    Returns:
        np.ndarray: Array containing the taken values.
    """

...



@overload
def take_nd(
        arr: ExtensionArray,
        indexer,
        axis: AxisInt = ...,
        fill_value=...,
        allow_fill: bool = ...
) -> ArrayLike:
    """
    Perform a take operation along a specified axis on the given ExtensionArray.

    Parameters:
        arr (ExtensionArray): The input array.
        indexer : The indexer specifying the elements to be taken.
        axis (AxisInt, optional): The axis along which to perform the take operation. Default is None.
        fill_value (optional): The value to fill in when there are missing elements in the indexer. Default is None.
        allow_fill (bool, optional): Flag indicating whether to allow filling in missing elements using the fill_value. 
                                    Default is True.

    Returns:
        ArrayLike: The result of the take operation.

    """

    ...



def take_nd(arr: ArrayLike, indexer, axis: AxisInt = 0, fill_value=lib.no_default, allow_fill: bool = True) -> ArrayLike:
    """
    Specialized Cython take which sets NaN values in one pass

    This dispatches to ``take`` defined on ExtensionArrays.

    Note: this function assumes that the indexer is a valid(ated) indexer with
    no out of bound indices.

    Parameters
    ----------
    arr : np.ndarray or ExtensionArray
        Input array.
    indexer : ndarray
        1-D array of indices to take, subarrays corresponding to -1 value
        indices are filed with fill_value
    axis : int, default 0
        Axis to take from
    fill_value : any, default np.nan
        Fill value to replace -1 values with
    allow_fill : bool, default True
        If False, indexer is assumed to contain no -1 values so no filling
        will be done.  This short-circuits computation of a mask.  Result is
        undefined if allow_fill == False and -1 is present in indexer.

    Returns
    -------
    subarray : np.ndarray or ExtensionArray
        May be the same type as the input, or cast to an ndarray.
    """
    
    # Set the fill_value if not provided
    if fill_value is lib.no_default:
        fill_value = na_value_for_dtype(arr.dtype, compat=False)
    # Promote the fill_value to match the dtype if arr is a datetime or timedelta array
    elif lib.is_np_dtype(arr.dtype, "mM"):
        dtype, promoted_fill_value = maybe_promote(arr.dtype, fill_value)
        if arr.dtype != dtype:
            arr = arr.astype(dtype)
        fill_value = promoted_fill_value
    
    if not isinstance(arr, np.ndarray):
        if not is_1d_only_ea_dtype(arr.dtype):
            arr = cast("NDArrayBackedExtensionArray", arr)
        return arr.take(indexer, fill_value=fill_value, allow_fill=allow_fill, axis=axis)
    
    arr = np.asarray(arr)
    return _take_nd_ndarray(arr, indexer, axis, fill_value, allow_fill)




def _take_nd_ndarray(
    arr: np.ndarray,
    indexer: npt.NDArray[np.intp] | None,
    axis: AxisInt,
    fill_value,
    allow_fill: bool,
) -> np.ndarray:
    """
    Take elements from an array along a specified axis.

    Args:
        arr: The input array.
        indexer: The indices used to take elements from the array.
        axis: The axis along which to take elements.
        fill_value: The value to use for missing indices when allow_fill is True.
        allow_fill: Indicates whether to fill missing indices or not.

    Returns:
        The array containing the taken elements.

    """

    # If indexer is None, create a default indexer with range of shape[axis]
    if indexer is None:
        indexer = np.arange(arr.shape[axis], dtype=np.intp)
        dtype, fill_value = arr.dtype, arr.dtype.type()
    else:
        indexer = ensure_platform_int(indexer)

    # Preprocess the indexer and fill_value
    dtype, fill_value, mask_info = _take_preprocess_indexer_and_fill_value(
        arr, indexer, fill_value, allow_fill
    )

    # Flip the order if arr is 2D and C-contiguous
    flip_order = False
    if arr.ndim == 2 and arr.flags.f_contiguous:
        flip_order = True

    if flip_order:
        arr = arr.T
        axis = arr.ndim - axis - 1

    # Calculate the shape of the output array
    out_shape_ = list(arr.shape)
    out_shape_[axis] = len(indexer)
    out_shape = tuple(out_shape_)

    # Create the output array
    if arr.flags.f_contiguous and axis == arr.ndim - 1:
        out = np.empty(out_shape, dtype=dtype, order="F")
    else:
        out = np.empty(out_shape, dtype=dtype)

    # Get the appropriate function for taking elements and apply it
    func = _get_take_nd_function(
        arr.ndim, arr.dtype, out.dtype, axis=axis, mask_info=mask_info
    )
    func(arr, indexer, out, fill_value)

    if flip_order:
        out = out.T
    return out



def take_1d(
    arr: ArrayLike,
    indexer: npt.NDArray[np.intp],
    fill_value=None,
    allow_fill: bool = True,
    mask: npt.NDArray[np.bool_] | None = None,
) -> ArrayLike:
    """
    Specialized version for 1D arrays. Differences compared to `take_nd`:

    - Assumes input array has already been converted to numpy array / EA
    - Assumes indexer is already guaranteed to be intp dtype ndarray
    - Only works for 1D arrays

    To ensure the lowest possible overhead.

    Note: similarly to `take_nd`, this function assumes that the indexer is
    a valid(ated) indexer with no out of bound indices.

    Parameters
    ----------
    arr : np.ndarray or ExtensionArray
        Input array.
    indexer : ndarray
        1-D array of indices to take (validated indices, intp dtype).
    fill_value : any, default np.nan
        Fill value to replace -1 values with
    allow_fill : bool, default True
        If False, indexer is assumed to contain no -1 values so no filling
        will be done.  This short-circuits computation of a mask. Result is
        undefined if allow_fill == False and -1 is present in indexer.
    mask : np.ndarray, optional, default None
        If `allow_fill` is True, and the mask (where indexer == -1) is already
        known, it can be passed to avoid recomputation.
    """
    # Dispatch ExtensionArray to their method
    if not isinstance(arr, np.ndarray):
        return arr.take(indexer, fill_value=fill_value, allow_fill=allow_fill)

    # If filling is not allowed, return the array without taking
    if not allow_fill:
        return arr.take(indexer)

    dtype, fill_value, mask_info = _take_preprocess_indexer_and_fill_value(
        arr, indexer, fill_value, True, mask
    )

    # At this point, it's guaranteed that dtype can hold both the arr values and the fill_value
    out = np.empty(indexer.shape, dtype=dtype)

    # Get the take_nd function
    func = _get_take_nd_function(
        arr.ndim, arr.dtype, out.dtype, axis=0, mask_info=mask_info
    )

    # Perform the take operation
    func(arr, indexer, out, fill_value)

    return out



def take_2d_multi(
    arr: np.ndarray,
    indexer: tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]],
    fill_value=np.nan,
) -> np.ndarray:
    """
    Specialized Cython take which sets NaN values in one pass.
    """

    # Check for well-behaved indexer
    assert indexer[0] is not None and indexer[1] is not None, "Indexer cannot be None"
    row_idx, col_idx = indexer

    # Ensure row_idx and col_idx are integer arrays
    row_idx = ensure_platform_int(row_idx)
    col_idx = ensure_platform_int(col_idx)

    # Update indexer with revised row_idx and col_idx
    indexer = row_idx, col_idx

    # Initialize mask_info with None
    mask_info = None

    # Check for promotion based on types only
    dtype, fill_value = maybe_promote(arr.dtype, fill_value)

    # Check if promotion is actually required based on indexer
    row_mask = row_idx == -1
    col_mask = col_idx == -1
    row_needs = row_mask.any()
    col_needs = col_mask.any()
    mask_info = (row_mask, col_mask), (row_needs, col_needs)

    if dtype != arr.dtype:
        # Depromote and set fill_value to dummy if promotion is not required
        if not (row_needs or col_needs):
            dtype, fill_value = arr.dtype, arr.dtype.type()

    # Initialize out_shape with length of row_idx and col_idx
    out_shape = len(row_idx), len(col_idx)
    
    # Create an empty array with dtype
    out = np.empty(out_shape, dtype=dtype)

    # Get appropriate function from _take_2d_multi_dict
    func = _take_2d_multi_dict.get((arr.dtype.name, out.dtype.name), None)
    if func is None and arr.dtype != out.dtype:
        func = _take_2d_multi_dict.get((out.dtype.name, out.dtype.name), None)
        if func is not None:
            func = _convert_wrapper(func, out.dtype)

    # Call appropriate function if func is not None
    if func is not None:
        func(arr, indexer, out=out, fill_value=fill_value)
    else:
        # Call _take_2d_multi_object if func is None
        _take_2d_multi_object(arr, indexer, out, fill_value=fill_value, mask_info=mask_info)

    # Return output array
    return out



# Refactor to reduce complexity of nested conditions
@functools.lru_cache
def _get_take_nd_function_cached(ndim: int, arr_dtype: np.dtype, out_dtype: np.dtype, axis: AxisInt):
    """
    Part of _get_take_nd_function below that doesn't need `mask_info` and thus
    can be cached (mask_info potentially contains a numpy ndarray which is not
    hashable and thus cannot be used as argument for cached function).
    """
    # find the appropriate function based on ndim and axis
    if ndim == 1:
        func = _take_1d_dict.get((arr_dtype.name, out_dtype.name), None)
    elif ndim == 2:
        if axis == 0:
            func = _take_2d_axis0_dict.get((arr_dtype.name, out_dtype.name), None)
        else:
            func = _take_2d_axis1_dict.get((arr_dtype.name, out_dtype.name), None)

    # if a function is found, return it
    if func is not None:
        return func

    # handle specific dtypes and convert to the appropriate function
    if out_dtype.name in ['string', 'uint', 'float16', 'complex'] or out_dtype.name.startswith('M8[ns]') or out_dtype.name.startswith('m8[ns]'):
        if ndim == 1:
            func = _take_1d_dict.get((out_dtype.name, out_dtype.name), None)
        elif ndim == 2:
            if axis == 0:
                func = _take_2d_axis0_dict.get((out_dtype.name, out_dtype.name), None)
            else:
                func = _take_2d_axis1_dict.get((out_dtype.name, out_dtype.name), None)
        
        # if a function is found, convert it and return it
        if func is not None:
            func = _convert_wrapper(func, out_dtype)
            return func

    return None




def _get_take_nd_function(ndim, arr_dtype, out_dtype, axis=0, mask_info=None):
    """
    Get the appropriate "take" implementation for the given dimension, axis
    and dtypes.
    """

    # Try to retrieve cached function if ndim is less than or equal to 2
    func = _get_take_nd_function_cached(ndim, arr_dtype, out_dtype, axis) if ndim <= 2 else None

    if func is None:
        # Define a nested function to be used as implementation
        def func(arr, indexer, out, fill_value=np.nan) -> None:
            indexer = ensure_platform_int(indexer)
            _take_nd_object(arr, indexer, out, axis=axis, fill_value=fill_value, mask_info=mask_info)

    return func




def _view_wrapper(f, arr_dtype=None, out_dtype=None, fill_wrap=None):
    """
    A wrapper function that applies views and wraps fill value before calling
    the wrapped function.

    Parameters:
    - f: a function to be wrapped
    - arr_dtype: the data type to view the input array as (default: None)
    - out_dtype: the data type to view the output array as (default: None)
    - fill_wrap: a function to wrap the fill value (default: None)

    Returns:
    - wrapper: the wrapped function
    """

    def wrapper(
        arr: np.ndarray, indexer: np.ndarray, out: np.ndarray, fill_value=np.nan
    ) -> None:
        """
        A wrapper function that applies views and wraps fill value before calling
        the wrapped function.

        Parameters:
        - arr: input array
        - indexer: array of indices
        - out: output array
        - fill_value: value to fill missing data (default: np.nan)
        """
        # Apply arr_dtype if provided
        if arr_dtype is not None:
            arr = arr.view(arr_dtype)
        
        # Apply out_dtype if provided
        if out_dtype is not None:
            out = out.view(out_dtype)
        
        # Apply fill_wrap if provided
        if fill_wrap is not None:
            # FIXME: if we get here with dt64/td64 we need to be sure we have matching resos
            if fill_value.dtype.kind == "m":
                fill_value = fill_value.astype("m8[ns]")
            else:
                fill_value = fill_value.astype("M8[ns]")
            fill_value = fill_wrap(fill_value)

        f(arr, indexer, out, fill_value=fill_value)

    return wrapper




def _convert_wrapper(f, conv_dtype):
    """
    Wrapper function that converts arr to conv_dtype and calls function f
    with the converted arr.

    Args:
        f: The function to call with the converted arr.
        conv_dtype: The dtype to convert arr to.

    Returns:
        The wrapper function.
    """
    def wrapper(arr: np.ndarray, indexer: np.ndarray, out: np.ndarray, fill_value=np.nan) -> None:
        """
        Wrapper function that converts arr to conv_dtype and calls function f
        with the converted arr.

        Args:
            arr: The array to convert.
            indexer: The indexer to apply on arr.
            out: The output array.
            fill_value: The fill value to use.

        Returns:
            None
        """
        # Avoid casting dt64/td64 to integers
        if conv_dtype == object:
            arr = ensure_wrapped_if_datetimelike(arr)
        # Convert arr to conv_dtype
        arr = arr.astype(conv_dtype)
        # Call function f with the converted arr
        f(arr, indexer, out, fill_value=fill_value)

    return wrapper




_take_1d_dict = {
    ("int8", "int8"): libalgos.take_1d_int8_int8,
    ("int8", "int32"): libalgos.take_1d_int8_int32,
    ("int8", "int64"): libalgos.take_1d_int8_int64,
    ("int8", "float64"): libalgos.take_1d_int8_float64,
    ("int16", "int16"): libalgos.take_1d_int16_int16,
    ("int16", "int32"): libalgos.take_1d_int16_int32,
    ("int16", "int64"): libalgos.take_1d_int16_int64,
    ("int16", "float64"): libalgos.take_1d_int16_float64,
    ("int32", "int32"): libalgos.take_1d_int32_int32,
    ("int32", "int64"): libalgos.take_1d_int32_int64,
    ("int32", "float64"): libalgos.take_1d_int32_float64,
    ("int64", "int64"): libalgos.take_1d_int64_int64,
    ("int64", "float64"): libalgos.take_1d_int64_float64,
    ("float32", "float32"): libalgos.take_1d_float32_float32,
    ("float32", "float64"): libalgos.take_1d_float32_float64,
    ("float64", "float64"): libalgos.take_1d_float64_float64,
    ("object", "object"): libalgos.take_1d_object_object,
    ("bool", "bool"): _view_wrapper(libalgos.take_1d_bool_bool, np.uint8, np.uint8),
    ("bool", "object"): _view_wrapper(libalgos.take_1d_bool_object, np.uint8, None),
    ("datetime64[ns]", "datetime64[ns]"): _view_wrapper(
        libalgos.take_1d_int64_int64, np.int64, np.int64, np.int64
    ),
    ("timedelta64[ns]", "timedelta64[ns]"): _view_wrapper(
        libalgos.take_1d_int64_int64, np.int64, np.int64, np.int64
    ),
}

_take_2d_axis0_dict = {
    ("int8", "int8"): libalgos.take_2d_axis0_int8_int8,
    ("int8", "int32"): libalgos.take_2d_axis0_int8_int32,
    ("int8", "int64"): libalgos.take_2d_axis0_int8_int64,
    ("int8", "float64"): libalgos.take_2d_axis0_int8_float64,
    ("int16", "int16"): libalgos.take_2d_axis0_int16_int16,
    ("int16", "int32"): libalgos.take_2d_axis0_int16_int32,
    ("int16", "int64"): libalgos.take_2d_axis0_int16_int64,
    ("int16", "float64"): libalgos.take_2d_axis0_int16_float64,
    ("int32", "int32"): libalgos.take_2d_axis0_int32_int32,
    ("int32", "int64"): libalgos.take_2d_axis0_int32_int64,
    ("int32", "float64"): libalgos.take_2d_axis0_int32_float64,
    ("int64", "int64"): libalgos.take_2d_axis0_int64_int64,
    ("int64", "float64"): libalgos.take_2d_axis0_int64_float64,
    ("float32", "float32"): libalgos.take_2d_axis0_float32_float32,
    ("float32", "float64"): libalgos.take_2d_axis0_float32_float64,
    ("float64", "float64"): libalgos.take_2d_axis0_float64_float64,
    ("object", "object"): libalgos.take_2d_axis0_object_object,
    ("bool", "bool"): _view_wrapper(
        libalgos.take_2d_axis0_bool_bool, np.uint8, np.uint8
    ),
    ("bool", "object"): _view_wrapper(
        libalgos.take_2d_axis0_bool_object, np.uint8, None
    ),
    ("datetime64[ns]", "datetime64[ns]"): _view_wrapper(
        libalgos.take_2d_axis0_int64_int64, np.int64, np.int64, fill_wrap=np.int64
    ),
    ("timedelta64[ns]", "timedelta64[ns]"): _view_wrapper(
        libalgos.take_2d_axis0_int64_int64, np.int64, np.int64, fill_wrap=np.int64
    ),
}

_take_2d_axis1_dict = {
    ("int8", "int8"): libalgos.take_2d_axis1_int8_int8,
    ("int8", "int32"): libalgos.take_2d_axis1_int8_int32,
    ("int8", "int64"): libalgos.take_2d_axis1_int8_int64,
    ("int8", "float64"): libalgos.take_2d_axis1_int8_float64,
    ("int16", "int16"): libalgos.take_2d_axis1_int16_int16,
    ("int16", "int32"): libalgos.take_2d_axis1_int16_int32,
    ("int16", "int64"): libalgos.take_2d_axis1_int16_int64,
    ("int16", "float64"): libalgos.take_2d_axis1_int16_float64,
    ("int32", "int32"): libalgos.take_2d_axis1_int32_int32,
    ("int32", "int64"): libalgos.take_2d_axis1_int32_int64,
    ("int32", "float64"): libalgos.take_2d_axis1_int32_float64,
    ("int64", "int64"): libalgos.take_2d_axis1_int64_int64,
    ("int64", "float64"): libalgos.take_2d_axis1_int64_float64,
    ("float32", "float32"): libalgos.take_2d_axis1_float32_float32,
    ("float32", "float64"): libalgos.take_2d_axis1_float32_float64,
    ("float64", "float64"): libalgos.take_2d_axis1_float64_float64,
    ("object", "object"): libalgos.take_2d_axis1_object_object,
    ("bool", "bool"): _view_wrapper(
        libalgos.take_2d_axis1_bool_bool, np.uint8, np.uint8
    ),
    ("bool", "object"): _view_wrapper(
        libalgos.take_2d_axis1_bool_object, np.uint8, None
    ),
    ("datetime64[ns]", "datetime64[ns]"): _view_wrapper(
        libalgos.take_2d_axis1_int64_int64, np.int64, np.int64, fill_wrap=np.int64
    ),
    ("timedelta64[ns]", "timedelta64[ns]"): _view_wrapper(
        libalgos.take_2d_axis1_int64_int64, np.int64, np.int64, fill_wrap=np.int64
    ),
}

_take_2d_multi_dict = {
    ("int8", "int8"): libalgos.take_2d_multi_int8_int8,
    ("int8", "int32"): libalgos.take_2d_multi_int8_int32,
    ("int8", "int64"): libalgos.take_2d_multi_int8_int64,
    ("int8", "float64"): libalgos.take_2d_multi_int8_float64,
    ("int16", "int16"): libalgos.take_2d_multi_int16_int16,
    ("int16", "int32"): libalgos.take_2d_multi_int16_int32,
    ("int16", "int64"): libalgos.take_2d_multi_int16_int64,
    ("int16", "float64"): libalgos.take_2d_multi_int16_float64,
    ("int32", "int32"): libalgos.take_2d_multi_int32_int32,
    ("int32", "int64"): libalgos.take_2d_multi_int32_int64,
    ("int32", "float64"): libalgos.take_2d_multi_int32_float64,
    ("int64", "int64"): libalgos.take_2d_multi_int64_int64,
    ("int64", "float64"): libalgos.take_2d_multi_int64_float64,
    ("float32", "float32"): libalgos.take_2d_multi_float32_float32,
    ("float32", "float64"): libalgos.take_2d_multi_float32_float64,
    ("float64", "float64"): libalgos.take_2d_multi_float64_float64,
    ("object", "object"): libalgos.take_2d_multi_object_object,
    ("bool", "bool"): _view_wrapper(
        libalgos.take_2d_multi_bool_bool, np.uint8, np.uint8
    ),
    ("bool", "object"): _view_wrapper(
        libalgos.take_2d_multi_bool_object, np.uint8, None
    ),
    ("datetime64[ns]", "datetime64[ns]"): _view_wrapper(
        libalgos.take_2d_multi_int64_int64, np.int64, np.int64, fill_wrap=np.int64
    ),
    ("timedelta64[ns]", "timedelta64[ns]"): _view_wrapper(
        libalgos.take_2d_multi_int64_int64, np.int64, np.int64, fill_wrap=np.int64
    ),
}


import numpy as np
import numpy.typing as npt

from typing import Any, Optional, Tuple

AxisInt = int

def _take_nd_object(
    arr: np.ndarray,
    indexer: npt.NDArray[np.intp],
    out: np.ndarray,
    axis: AxisInt,
    fill_value: Any,
    mask_info: Optional[Tuple[np.ndarray, bool]]
) -> None:
    """
    Take elements from an array along an axis.

    Args:
        arr: The input array.
        indexer: An array of indices.
        out: The output array.
        axis: The axis along which to take elements.
        fill_value: The value to fill any masked elements in the output array with.
        mask_info: A tuple consisting of the mask array and a boolean specifying if the array needs masking.
    """

    # Unpack the mask_info tuple
    if mask_info is not None:
        mask, needs_masking = mask_info
    else:
        mask = indexer == -1
        needs_masking = mask.any()

    # Convert arr to the same dtype as out
    if arr.dtype != out.dtype:
        arr = arr.astype(out.dtype)

    # Take elements from arr along the specified axis
    if arr.shape[axis] > 0:
        arr.take(indexer, axis=axis, out=out)

    # Fill any masked elements in the output array with fill_value
    if needs_masking:
        outindexer = [slice(None)] * arr.ndim
        outindexer[axis] = mask
        out[tuple(outindexer)] = fill_value




def _take_2d_multi_object(
    arr: np.ndarray,
    indexer: tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]],
    out: np.ndarray,
    fill_value,
    mask_info,
) -> None:
    """
    Take values from a 2-dimensional array based on row and column indices.

    Args:
        arr (np.ndarray): The 2-dimensional array.
        indexer (tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]]): The row and column indices.
        out (np.ndarray): The output array.
        fill_value: The value to fill missing indices with.
        mask_info: Additional information for masking.

    Returns:
        None
    """

    row_idx, col_idx = indexer  # both np.intp

    row_mask = np.zeros_like(row_idx, dtype=bool)
    col_mask = np.zeros_like(col_idx, dtype=bool)
    row_needs = False
    col_needs = False

    # check if mask_info is not None and update row_mask, col_mask, row_needs and col_needs
    if mask_info is not None:
        (row_mask, col_mask), (row_needs, col_needs) = mask_info
    else:
        row_mask = row_idx == -1
        col_mask = col_idx == -1
        row_needs = row_mask.any()
        col_needs = col_mask.any()

    # fill missing values with fill_value if fill_value is not None
    if fill_value is not None:
        if row_needs:
            out[row_mask, :] = fill_value
        if col_needs:
            out[:, col_mask] = fill_value

    # take values from arr based on row and column indices
    for i, u_ in enumerate(row_idx):
        if u_ != -1:
            for j, v in enumerate(col_idx):
                if v != -1:
                    out[i, j] = arr[u_, v]




def _take_preprocess_indexer_and_fill_value(arr: np.ndarray,
                                           indexer: npt.NDArray[np.intp],
                                           fill_value,
                                           allow_fill: bool,
                                           mask: npt.NDArray[np.bool_] | None = None,
                                           ) -> tuple[np.dtype, any, tuple[np.ndarray | None, bool] | None]:
    """Take preprocess indexer and fill value."""
   
    if not allow_fill:
        dtype = arr.dtype
        fill_value = arr.dtype.type()
        mask_info = None, False
    else:
        dtype, fill_value = maybe_promote(arr.dtype, fill_value)

        if dtype != arr.dtype:
            if mask is not None:
                needs_masking = True
            else:
                mask = indexer == -1
                needs_masking = mask.any()
                
            mask_info = mask, needs_masking
            
            if not needs_masking:
                dtype = arr.dtype
                fill_value = arr.dtype.type()

    return dtype, fill_value, mask_info


