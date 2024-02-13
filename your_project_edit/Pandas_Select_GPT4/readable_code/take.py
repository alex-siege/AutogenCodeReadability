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


from typing import overload

@overload
def take_nd(
    arr: np.ndarray,  # Array from which elements will be taken
    indexer,          # Indexer (e.g., list of indices) that specifies which elements to take
    axis: AxisInt = ...,  # Axis along which to take elements from the array
    fill_value=...,        # Fill value to use if `allow_fill` is True and indexer is out-of-bounds
    allow_fill: bool = ...,    # Whether to allow filling with `fill_value` for out-of-bounds indices
) -> np.ndarray:
    """
    Overload signature declaration for 'take_nd' function.
    
    This function will take elements from `arr` along a specified `axis` according to `indexer`.
    If `allow_fill` is True, it allows filling indices that are out-of-bounds with `fill_value`.
    The actual implementation of the function will use this signature to type check the inputs.
    
    Parameters:
    - arr (np.ndarray): The input array from which the elements are taken.
    - indexer: Indexes to take along the `axis` of `arr`.
    - axis (AxisInt, optional): The axis along which elements should be taken.
    - fill_value (optional): The value to use for out-of-bound indices if `allow_fill` is True.
    - allow_fill (bool, optional): True if filling is allowed; otherwise, False.
    
    Returns:
    - np.ndarray: An array consisting of the elements taken from `arr` along the specified axis.
    """
    ...



@overload
def take_nd(
    arr: ExtensionArray, 
    indexer,
    axis: AxisInt = ...,
    fill_value=...,
    allow_fill: bool = ...,
) -> ArrayLike:
    """
    Method signature for `take_nd` function which selects elements from an `ExtensionArray`
    according to the specified `indexer` and along a given axis. 

    Args:
    - arr (ExtensionArray): The input array from which to take elements.
    - indexer: The indices of the elements to be selected.
    - axis (AxisInt, optional): The axis along which to select elements. Default is undefined.
    - fill_value (optional): The value to use for missing indices when allow_fill is True. Default is undefined.
    - allow_fill (bool, optional): Whether to allow filling missing indices with the fill_value. Default is undefined.

    Returns:
    - ArrayLike: An array containing the selected elements.
    """
    ...



def take_nd(
    array: ArrayLike,
    indexer,
    axis: AxisInt = 0,
    fill_value=lib.no_default,
    allow_fill: bool = True,
) -> ArrayLike:
    """
    Take elements from an array along the specified axis using an indexer.

    This function handles ExtensionArrays and standard numpy ndarrays, automatically
    dealing with -1 values in the indexer by replacing them with the fill_value.

    Parameters
    ----------
    array : np.ndarray or ExtensionArray
        Input array from which to take elements.
    indexer : ndarray
        1-D array of integer indices to take. Positions with -1 will be filled 
        with fill_value if allow_fill is True.
    axis : int, default 0
        Axis along which the elements are taken.
    fill_value : any, default np.nan
        Value used to fill in the elements corresponding to -1 in indexer.
    allow_fill : bool, default True
        Whether to fill positions in the result corresponding to -1 in the
        indexer with fill_value.

    Returns
    -------
    subarray : np.ndarray or ExtensionArray
        An array of the taken elements, potentially with a different dtype.
    """
    # Determine the fill value based on the input dtype when no default is provided
    fill_value = _fill_value_for_dtype(array.dtype, fill_value)
    
    # Handle np.ndarrays and ExtensionArrays differently
    if not isinstance(array, np.ndarray):
        return _take_from_extension_array(array, indexer, axis, fill_value, allow_fill)
    else:
        # Convert input array to np.ndarray explicitly, if it isn't already one
        array = np.asarray(array)
        return _take_nd_ndarray(array, indexer, axis, fill_value, allow_fill)


def _fill_value_for_dtype(dtype, fill_value):
    """
    Obtain the fill value for an array based on its dtype.

    Parameters
    ----------
    dtype : dtype
        The dtype of the array for which to determine the fill value.
    fill_value : any
        The provided fill value, which may be a default marker indicating
        that a specific fill value should be inferred from the dtype.

    Returns
    -------
    fill_value : any
        The fill value to use for the array.
    """
    if fill_value is lib.no_default:
        return na_value_for_dtype(dtype, compat=False)
    elif lib.is_np_dtype(dtype, "mM"):
        dtype, fill_value = maybe_promote(dtype, fill_value)
        if dtype != array.dtype:
            # Cast to the promoted dtype
            return dtype, fill_value
    return fill_value


def _take_from_extension_array(array, indexer, axis, fill_value, allow_fill):
    """
    Handle taking elements from an ExtensionArray.

    Parameters
    ----------
    array : ExtensionArray
        The input array to take elements from.
    indexer : ndarray
        The array of integer indices indicating which elements to take.
    axis : int
        The axis along which elements are taken.
    fill_value : any
        The fill value to use for missing positions (indicated by -1 in the indexer).
    allow_fill : bool
        Whether filling of missing positions is allowed.

    Returns
    -------
    subarray : ExtensionArray
        The resulting ExtensionArray containing the taken elements.
    """
    # Handle specific cases for 1D-only or general ExtensionArrays
    if is_1d_only_ea_dtype(array.dtype):
        # Cast to NDArrayBackedExtensionArray if needed
        array = cast("NDArrayBackedExtensionArray", array)
        return array.take(indexer, fill_value=fill_value, allow_fill=allow_fill, axis=axis)
    else:
        return array.take(indexer, fill_value=fill_value, allow_fill=allow_fill)



def _take_nd_ndarray(
    arr: np.ndarray,
    indexer: npt.NDArray[np.intp] | None,
    axis: AxisInt,
    fill_value,
    allow_fill: bool,
) -> np.ndarray:
    """
    Take values from an NDArray according to the given indexer and fill them if necessary.
    
    :param arr: numpy ndarray from which values are taken
    :param indexer: array of indices to be taken; if None, indices along the axis are used
    :param axis: the axis over which to select values
    :param fill_value: value to use for missing values
    :param allow_fill: whether to allow filling with the fill_value
    :return: numpy ndarray with values taken from the input array along the given axis
    """
    indexer, dtype, fill_value = _prepare_indexer_dtype_and_fill_value(arr, indexer, axis, fill_value, allow_fill)

    flip_order = _check_if_flip_order_required(arr)

    if flip_order:
        arr, axis = _transpose_arr_and_update_axis(arr, axis)

    out = _allocate_output_array(arr, dtype, indexer, axis)
    
    func = _get_take_nd_function(
        arr.ndim, arr.dtype, out.dtype, axis=axis, mask_info=mask_info
    )
    func(arr, indexer, out, fill_value)

    if flip_order:
        out = out.T
    
    return out

def _prepare_indexer_dtype_and_fill_value(arr, indexer, axis, fill_value, allow_fill):
    """
    Prepare the indexer, dtype and fill value for the take operation.
    
    :param arr: numpy ndarray from which values are taken
    :param indexer: array of indices to be taken or None
    :param axis: the axis over which to select values
    :param fill_value: value to use for missing values
    :param allow_fill: whether to allow filling with the fill_value
    :return: tuple of (indexer, dtype, fill_value)
    """
    if indexer is None:
        indexer = np.arange(arr.shape[axis], dtype=np.intp)
        return indexer, arr.dtype, arr.dtype.type()
    else:
        indexer = ensure_platform_int(indexer)

    dtype, fill_value, mask_info = _take_preprocess_indexer_and_fill_value(
        arr, indexer, fill_value, allow_fill
    )
    return indexer, dtype, fill_value

def _check_if_flip_order_required(arr):
    """
    Check if we need to flip the order of the given array.
    
    :param arr: numpy ndarray that is checked
    :return: boolean indicating if flipping is required
    """
    return arr.ndim == 2 and arr.flags.f_contiguous

def _transpose_arr_and_update_axis(arr, axis):
    """
    Transpose the array and update the axis accordingly.
    
    :param arr: numpy ndarray to transpose
    :param axis: axis to update after transpose
    :return: tuple of transposed array and updated axis
    """
    arr = arr.T
    axis = arr.ndim - axis - 1
    return arr, axis

def _allocate_output_array(arr, dtype, indexer, axis):
    """
    Allocate output array with the correct shape and dtype.
    
    :param arr: input numpy ndarray
    :param dtype: data type for the output array
    :param indexer: array of indices for selection
    :param axis: the axis over which to select values
    :return: numpy ndarray allocated for output
    """
    out_shape = list(arr.shape)
    out_shape[axis] = len(indexer)
    if arr.flags.f_contiguous and axis == arr.ndim - 1:
        return np.empty(tuple(out_shape), dtype=dtype, order="F")
    else:
        return np.empty(tuple(out_shape), dtype=dtype)



def take_1d(
    arr: ArrayLike,
    indexer: npt.NDArray[np.intp],
    fill_value=None,
    allow_fill: bool = True,
    mask: npt.NDArray[np.bool_] | None = None,
) -> ArrayLike:
    """
    Take elements from a 1D array (either a numpy array or an ExtensionArray) using an indexer.

    This specialized function is optimized for 1D arrays and should be used only when necessary 
    preconditions are met, such as the converted input array and validated indexer of intp dtype.

    Parameters
    ----------
    arr: ArrayLike
        An array-like structure from which to take elements. Could be a numpy array or an ExtensionArray.
    indexer: npt.NDArray[np.intp]
        A 1D numpy array of indices that specifies which elements to take.
    fill_value: any, default None
        The value to replace entries corresponding to `-1` in the indexer. Defaults to numpy's NaN.
    allow_fill: bool, default True
        If False, no filling will take place, assuming the indexer contains no `-1` values.
    mask: npt.NDArray[np.bool_] or None, default None
        An optional boolean mask that indicates where the indexer is equal to `-1`.

    Returns
    -------
    ArrayLike
        An array of the same type as `arr`, with elements taken according to `indexer`.
    """
    # Dispatch to dedicated method if input is not a numpy ndarray (e.g., if it's an ExtensionArray)
    if not isinstance(arr, np.ndarray):
        return arr.take(indexer, fill_value=fill_value, allow_fill=allow_fill)

    # If fill is not allowed, assume that indexer has no -1 values and use numpy's take directly
    if not allow_fill:
        return arr.take(indexer)

    # Preprocessing to adjust dtype, mask, and fill_value according to input requirements
    dtype, adjusted_fill_value, mask_info = _preprocess_inputs(
        arr, indexer, fill_value, mask
    )

    # Create an output array of the correct dtype
    out_arr = np.empty(indexer.shape, dtype=dtype)

    # Retrieve the appropriate underlying function for taking elements
    take_func = _get_take_function(arr.ndim, dtype, mask_info)

    # Perform the take action and populate the output array
    take_func(arr, indexer, out_arr, adjusted_fill_value)

    return out_arr

def _preprocess_inputs(arr, indexer, fill_value, mask):
    """
    Preprocess the inputs for the `take_1d` function, adjusting dtype and resolving mask and fill value.
    
    Parameters
    ----------
    arr: The array from which to take values.
    indexer: A 1D numpy array of indices that specifies which elements to take.
    fill_value: Value to replace entries corresponding to `-1` in the indexer.
    mask: An optional boolean mask indicating where the indexer equals `-1`.

    Returns
    -------
    Tuple of (dtype, fill_value, mask_info) to be used by the `take_1d` function.
    """
    return _take_preprocess_indexer_and_fill_value(
        arr, indexer, fill_value, True, mask
    )

def _get_take_function(ndim, arr_dtype, mask_info):
    """
    Retrieve a function suitable for taking elements according to the array's dimensions and the mask information.
    
    Parameters
    ----------
    ndim: Number of dimensions of the input array.
    arr_dtype: Data type of the input array.
    mask_info: Processed mask information based on fill requirements and indexer values.

    Returns
    -------
    Function that performs the take operation.
    """
    return _get_take_nd_function(
        ndim, arr_dtype, arr_dtype, axis=0, mask_info=mask_info
    )



def assert_valid_indexer(indexer):
    """
    Ensures that the indexer tuple and its contents are not None.

    Parameters:
    - indexer: A tuple containing two Numpy arrays for row and column indexing.
    """
    # Assert that the indexer is not None.
    assert indexer is not None, "Indexer cannot be None"
    # Assert that the first element (row indices) of the indexer is not None.
    assert indexer[0] is not None, "Row indices within indexer cannot be None"
    # Assert that the second element (column indices) of the indexer is not None.
    assert indexer[1] is not None, "Column indices within indexer cannot be None"

def determine_dtype_and_fill_value(arr, fill_value, row_idx, col_idx):
    """
    Determine and possibly promote the dtype for the output array based on given fill_value and indexer.

    Parameters:
    - arr: 2D NumPy array from which elements will be taken.
    - fill_value: Value used to fill missing data.
    - row_idx: Numpy array containing row indices.
    - col_idx: Numpy array containing column indices.

    Returns:
    Tuple containing the determined dtype and fill value.
    """
    # Attempt to promote the dtype based on the fill_value provided.
    dtype, fill_value = maybe_promote(arr.dtype, fill_value)

    # If the dtype is promoted and different from the array's dtype, check if promotion is required.
    if dtype != arr.dtype:
        # Construct masks for missing row and column indices.
        row_mask, col_mask = row_idx == -1, col_idx == -1
        # Depromote the dtype if missing data is not actually referenced by the indexer.
        if not (row_mask.any() or col_mask.any()):
            # Revert to the original array's dtype and corresponding fill value.
            dtype, fill_value = arr.dtype, arr.dtype.type()

    return dtype, fill_value

def get_take_function(arr_dtype_name, out_dtype_name):
    """
    Look up a specialized take function by dtype names.

    Parameters:
    - arr_dtype_name: The name of the dtype of the array from which elements are taken.
    - out_dtype_name: The name of the dtype for the output array.

    Returns:
    The function for the take operation, or None if no specialized function is available.
    """
    # Attempt to retrieve a take function based on the dtypes involved.
    func = _take_2d_multi_dict.get((arr_dtype_name, out_dtype_name))
    # If a specific function is not found and dtypes differ, try getting a generic take function.
    if func is None and arr_dtype_name != out_dtype_name:
        # Fallback to a generic function for the output dtype.
        func = _take_2d_multi_dict.get((out_dtype_name, out_dtype_name))
        # If found, convert the function using a wrapper for compatibility with the output dtype.
        if func is not None:
            func = _convert_wrapper(func, out_dtype_name)
    return func

def apply_take_function(func, arr, indexer, out, fill_value):
    """
    Apply the take operation using the available function or fall back to a generic handler.

    Parameters:
    - func: Function to be used for the take operation.
    - arr: 2D Numpy array from which elements will be taken.
    - indexer: Tuple containing row and column index arrays.
    - out: Output 2D Numpy array to store the taken elements.
    - fill_value: Value used to fill missing data.
    """
    # If there is a specialized function available, use it to perform the take operation.
    if func is not None:
        func(arr, indexer, out=out, fill_value=fill_value)
    else:
        # If the specialized function is not available, use a generic handler.
        # The mask_info is not needed here because dtype alignment was handled earlier.
        _take_2d_multi_object(arr, indexer, out, fill_value=fill_value, mask_info=None)



import functools

@functools.lru_cache
def _get_take_nd_function_cached(ndim, arr_dtype, out_dtype, axis):
    """
    Fetch a cached 'take' function based on array dimensions, data types, and axis.
    
    This function checks if there is an existing optimized function for performing
    'take' operation on an array of given dimensions and data types. If a suitable
    cached function is found, it is returned; otherwise, None is returned.
    
    Parameters:
    - ndim: The number of dimensions of the array.
    - arr_dtype: The data type of the source array.
    - out_dtype: The data type of the output array.
    - axis: The axis along which to take elements.
    """
    def get_func_for_dim(take_func_dict_1d, take_func_dict_2d):
        """
        Helper function to fetch from the appropriate function dictionary
        based on the dimensionality and axis.
        """
        if ndim == 1:
            return take_func_dict_1d.get((arr_dtype.name, out_dtype.name), None)
        elif ndim == 2:
            if axis == 0:
                return take_func_dict_2d.get((arr_dtype.name, out_dtype.name), None)
            else:
                return take_func_dict_2d.get((arr_dtype.name, out_dtype.name), None)
        return None

    # Attempt to find the function in the pre-existing dictionary for 1D or 2D cases
    func = get_func_for_dim(_take_1d_dict, _take_2d_axis0_dict if axis == 0 else _take_2d_axis1_dict)
    
    # If an appropriate function is found, return it
    if func:
        return func

    # An alternative attempt to find a function for converting the data type
    func = get_func_for_dim(_take_1d_dict, _take_2d_axis0_dict if axis == 0 else _take_2d_axis1_dict)
    
    # If found, wrap and return a function that additionally converts the data type
    if func:
        return _convert_wrapper(func, out_dtype)

    # If no suitable function is found, return None
    return None



def _get_take_nd_function(
    ndim: int,
    arr_dtype: np.dtype,
    out_dtype: np.dtype,
    axis: AxisInt = 0,
    mask_info=None,
):
    """
    Get the appropriate "take" implementation for the given dimension, axis,
    and dtypes.

    Parameters:
        ndim: int - The number of dimensions of the array.
        arr_dtype: np.dtype - The data type of the input array.
        out_dtype: np.dtype - The desired data type of the output array.
        axis: AxisInt - The axis along which to take the elements.
        mask_info: Optional information about the mask being applied.

    Returns:
        A function that implements the "take" operation for the specified
        conditions.
    """

    # If array dimension is 2 or less, use cached approach for performance
    if ndim <= 2:
        take_nd_func = _get_take_nd_function_cached(ndim, arr_dtype, out_dtype, axis)
    else:
        # Define a nested function for taking with mask information
        take_nd_func = _take_nd_with_mask(mask_info)

    return take_nd_func

def _take_nd_with_mask(mask_info):
    """
    Create a take function that considers mask information.

    Parameters:
        mask_info: Information about the mask being applied.

    Returns:
        A function that can be used to perform a masked take operation.
    """
    def take_nd_func(arr, indexer, out, fill_value=np.nan) -> None:
        """
        Perform a masked take operation on an array.

        Parameters:
            arr: Input array on which take operation is being performed.
            indexer: Indices to take along the specified axis.
            out: Output array where results will be placed.
            fill_value: Value to use for missing positions.
        """
        # Ensure the indexer is in the platform-specific integer format
        indexer = ensure_platform_int(indexer)
        # Perform the object dtype take with the mask info
        _take_nd_object(
            arr, indexer, out, axis=axis, fill_value=fill_value, mask_info=mask_info
        )

    return take_nd_func



def _view_wrapper(func, arr_dtype=None, out_dtype=None, fill_wrap=None):
    """
    Wraps the function 'func' to provide additional processing of input and output arrays.
    
    Parameters:
    - func: The function to wrap, which operates on arrays.
    - arr_dtype: The data type to which the input array should be cast before processing.
    - out_dtype: The data type to which the output array should be cast before processing.
    - fill_wrap: A function to modify the fill value if provided.
    """
    
    def wrapper(arr: np.ndarray, indexer: np.ndarray, out: np.ndarray, fill_value=np.nan) -> None:
        """
        The actual wrapper function that casts arrays and modifies the fill value before calling 'func'.
        
        Parameters:
        - arr: The input array to be processed.
        - indexer: An index array indicating which elements of 'arr' to process.
        - out: The output array where results are stored.
        - fill_value: The fill value to use where data is missing.
        """
        # Cast the input array to the specified data type if provided.
        if arr_dtype is not None:
            arr = cast_array_dtype(arr, arr_dtype)
        
        # Cast the output array to the specified data type if provided.
        if out_dtype is not None:
            out = cast_array_dtype(out, out_dtype)
        
        # Modify the fill_value as needed before processing
        if fill_wrap is not None:
            fill_value = modify_fill_value(fill_value, fill_wrap)
        
        # Call the original function with modified arguments.
        func(arr, indexer, out, fill_value=fill_value)
    
    return wrapper

def cast_array_dtype(array: np.ndarray, dtype) -> np.ndarray:
    """
    Casts an array to a specified data type.
    
    Parameters:
    - array: The array to cast.
    - dtype: The target data type.
    
    Returns:
    - The array cast to 'dtype'.
    """
    return array.view(dtype)

def modify_fill_value(fill_value, fill_wrap) -> np.dtype:
    """
    Modifies the fill value for data types that require specific resolutions.
    
    Parameters:
    - fill_value: The original fill value.
    - fill_wrap: The function to modify the fill value.
    
    Returns:
    - The modified fill value.
    """
    # Handle datetime64 or timedelta64 types specifically.
    if fill_value.dtype.kind in ["m", "M"]:
        new_fill_value_dtype = "m8[ns]" if fill_value.dtype.kind == "m" else "M8[ns]"
        fill_value = fill_value.astype(new_fill_value_dtype)
    return fill_wrap(fill_value)



def _convert_wrapper(f, target_dtype):
    """
    Create a wrapper function that converts an array to the target dtype,
    then calls the provided function with the converted array and additional arguments.
    
    Parameters:
    f (callable): The function to be called with the converted array.
    target_dtype (dtype): The target numpy data type to convert the array into.
    
    Returns:
    callable: A wrapper function that can be called with an array, indexer, output array, and fill value.
    """
    
    def wrapper(arr: np.ndarray, indexer: np.ndarray, out: np.ndarray, fill_value=np.nan) -> None:
        """
        The wrapper function that converts the array to the target data type,
        and then calls the provided function with the converted array and additional arguments.
        
        Parameters:
        arr (np.ndarray): The input array to convert and process.
        indexer (np.ndarray): The indexer array.
        out (np.ndarray): The output array to store the result.
        fill_value (optional): The fill value to use for missing data. Defaults to np.nan.
        
        Returns:
        None
        """
        
        # If the target data type is object, ensure wrapping for datetimelike types to avoid wrong casting
        if target_dtype == object:
            arr = ensure_wrapped_if_datetimelike(arr)
        
        # Convert the input array to the target data type
        converted_arr = arr.astype(target_dtype)
        
        # Call the provided function with the converted array and the additional arguments
        f(converted_arr, indexer, out, fill_value=fill_value)

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


def _take_nd_object(
    arr: np.ndarray,
    indexer: npt.NDArray[np.intp],
    out: np.ndarray,
    axis: AxisInt,
    fill_value,
    mask_info,
) -> None:
    """
    Places the elements of an ndarray 'arr' into another array 'out' 
    according to a given indexer. Applies a fill_value where necessary.

    :param arr: Source array.
    :param indexer: Defines the indices of elements to be taken from 'arr'.
    :param out: Output array that will contain the taken elements.
    :param axis: The axis along which to take elements.
    :param fill_value: The value to use for 'out' where 'indexer' is -1.
    :param mask_info: Optional mask info to use instead of using the indexer.
    """
    # Initialize or retrieve mask and needs_masking
    mask, needs_masking = _get_mask_info(indexer, mask_info)
    # Cast array to the target dtype if different
    arr = _cast_arr_dtype(arr, out)
    # Take elements along the specified axis if the dimension is not zero
    _take_elements(arr, indexer, out, axis)
    # Apply fill values where masking is required
    if needs_masking:
        _apply_fill_value(out, axis, mask, fill_value)

def _get_mask_info(indexer, mask_info):
    """
    Retrieves or computes mask info based on the provided indexer.
    
    :param indexer: Array of indices.
    :param mask_info: Precomputed mask info.
    :return: A tuple containing the mask array and a boolean indicating if needs masking.
    """
    if mask_info is not None:
        mask, needs_masking = mask_info
    else:
        mask = indexer == -1
        needs_masking = mask.any()
    return mask, needs_masking

def _cast_arr_dtype(arr, out):
    """
    Casts an array to the dtype of the output array if they differ.
    
    :param arr: Input array.
    :param out: Target array for the operation.
    :return: The casted array or the original if the dtypes match.
    """
    if arr.dtype != out.dtype:
        arr = arr.astype(out.dtype)
    return arr

def _take_elements(arr, indexer, out, axis):
    """
    Takes elements from the array 'arr' placing them into the 'out' array.
    
    :param arr: Input array from which to take elements.
    :param indexer: Defines the indices of elements to take from 'arr'.
    :param out: Target output array.
    :param axis: Axis along which to take elements.
    """
    if arr.shape[axis] > 0:
        arr.take(indexer, axis=axis, out=out)

def _apply_fill_value(out, axis, mask, fill_value):
    """
    Applies a fill value to the specified portion of the 'out' array.
    
    :param out: The target array to fill.
    :param axis: The axis along which the filler will be applied.
    :param mask: A Boolean mask where the fill value needs to be applied.
    :param fill_value: The value used to fill the 'out' array.
    """
    outindexer = [slice(None)] * out.ndim
    outindexer[axis] = mask
    out[tuple(outindexer)] = fill_value



def _set_fill_value_for_needs(out, fill_value, row_mask, col_mask, row_needs, col_needs):
    """Set the fill value for rows and columns that need it."""
    if row_needs:
        out[row_mask, :] = fill_value
    if col_needs:
        out[:, col_mask] = fill_value

def _populate_output_array(out, arr, row_idx, col_idx):
    """Populate the output array with elements from the input array based on the indexes."""
    for i, row_index in enumerate(row_idx):
        if row_index != -1:  # Filter out invalid row indexes
            for j, col_index in enumerate(col_idx):
                if col_index != -1:  # Filter out invalid column indexes
                    out[i, j] = arr[row_index, col_index]

def _take_2d_multi_object(
    arr: np.ndarray,
    indexer: tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]],
    out: np.ndarray,
    fill_value,
    mask_info,
) -> None:
    """
    Take a 2D slice of a multiple-object array while accounting for missing data.

    Parameters
    ----------
    arr : np.ndarray
        The input array.
    indexer : tuple of npt.NDArray
        Tuple of row and column index arrays.
    out : np.ndarray
        The output array that will receive the sliced data.
    fill_value :
        Value to use for 'missing' elements.
    mask_info :
        Information about which elements are considered 'missing'.
    """
    row_idx, col_idx = indexer  # Extract row and column index arrays
    if mask_info is not None:
        (row_mask, col_mask), (row_needs, col_needs) = mask_info
    else:
        # Determine mask and needs for missing rows and columns
        row_mask = row_idx == -1
        col_mask = col_idx == -1
        row_needs = row_mask.any()
        col_needs = col_mask.any()

    if fill_value is not None:
        # If a fill value is provided, apply it where needed
        _set_fill_value_for_needs(out, fill_value, row_mask, col_mask, row_needs, col_needs)
    
    # Populate the output array with the appropriate data from the input array
    _populate_output_array(out, arr, row_idx, col_idx)



def _take_preprocess_indexer_and_fill_value(
    arr: np.ndarray,
    indexer: npt.NDArray[np.intp],
    fill_value,
    allow_fill: bool,
    mask: npt.NDArray[np.bool_] | None = None,
):
    """
    Process the indexer and fill_value for taking elements from an array.
    
    Parameters:
    arr (np.ndarray): The array from which elements are taken.
    indexer (npt.NDArray[np.intp]): An array of indices indicating which elements to take.
    fill_value: The value to use for missing or invalid positions.
    allow_fill (bool): Flag indicating whether filling is allowed.
    mask (npt.NDArray[np.bool_] | None, optional): An optional boolean mask for the indexer.

    Returns:
    tuple: Tuple containing the resulting dtype, the fill_value, and mask_info.
    """
    def get_mask_info(mask, indexer, arr):
        """
        Determines the need for masking and the mask information.
        """
        if mask is not None:
            return mask, True
        else:
            computed_mask = indexer == -1
            return computed_mask, computed_mask.any()

    # If fill is not allowed, use the array's dtype as fill_value and no masking is needed
    if not allow_fill:
        return arr.dtype, arr.dtype.type(), (None, False)

    # Determine if promotion is needed based on fill_value
    promoted_dtype, promoted_fill_value = maybe_promote(arr.dtype, fill_value)

    # If promotion is needed, calculate mask info
    if promoted_dtype != arr.dtype:
        mask, needs_masking = get_mask_info(mask, indexer, arr)
        mask_info = mask, needs_masking
        # If masking is not needed, revert to original dtype and fill_value
        if not needs_masking:
            return arr.dtype, arr.dtype.type(), mask_info

        return promoted_dtype, promoted_fill_value, mask_info

    # No promotion needed, return original dtype and fill_value
    return arr.dtype, fill_value, (None, False)

