"""
requests.utils
~~~~~~~~~~~~~~

This module provides utility functions that are used within Requests
that are also useful for external consumption.
"""

import codecs
import contextlib
import io
import os
import re
import socket
import struct
import sys
import tempfile
import warnings
import zipfile
from collections import OrderedDict

from urllib3.util import make_headers, parse_url

from . import certs
from .__version__ import __version__

# The import of to_native_string is maintained for backwards
# compatibility, though unused here.
from ._internal_utils import (
    _HEADER_VALIDATORS_BYTE,
    _HEADER_VALIDATORS_STR,
    HEADER_VALIDATORS,
    to_native_string,
)

from .compat import (
    Mapping,
    basestring,
    bytes,
    getproxies,
    getproxies_environment,
    integer_types,
)

from .compat import parse_http_list as _parse_list_header

from .compat import (
    proxy_bypass as _proxy_bypass_environment,
    proxy_bypass_environment,
    quote,
    str,
    unquote,
    urlparse,
    urlunparse,
)

from .cookies import cookiejar_from_dict

from .exceptions import (
    FileModeWarning,
    InvalidHeader,
    InvalidURL,
    UnrewindableBodyError,
)

from .structures import CaseInsensitiveDict

NETRC_FILES = (".netrc", "_netrc")

DEFAULT_CA_BUNDLE_PATH = certs.where()

DEFAULT_PORTS = {"http": 80, "https": 443}

# To ensure ', ' is used as delimiter, preserving the previous behavior.
DEFAULT_ACCEPT_ENCODING = ", ".join(
    re.split(r",\s*", make_headers(accept_encoding=True)["accept-encoding"])
)


if sys.platform == "win32":
    # Provide a proxy_bypass version on Windows without DNS lookups
    def proxy_bypass_registry(host):
        """Determines if the given host should bypass the proxy on Windows, by checking registry settings."""
        try:
            import winreg
        except ImportError:
            return False

        try:
            internet_settings_key = winreg.OpenKey(
                winreg.HKEY_CURRENT_USER,
                r"Software\Microsoft\Windows\CurrentVersion\Internet Settings",
            )
            proxy_enable_flag = int(
                winreg.QueryValueEx(
                    internet_settings_key,
                    "ProxyEnable")[0])
            proxy_override_list = winreg.QueryValueEx(
                internet_settings_key, "ProxyOverride")[0]
        except (OSError, ValueError):
            return False
        if not proxy_enable_flag or not proxy_override_list:
            return False

        proxy_override_list = proxy_override_list.split(";")
        for pattern in proxy_override_list:
            if should_bypass_proxy_for_host(pattern, host):
                return True
        return False

    def should_bypass_proxy_for_host(pattern, host):
        """Checks if the given host matches the bypass pattern defined in registry."""
        if pattern == "<local>":
            if "." not in host:
                return True
        pattern = pattern.replace(".", r"\.")  # Escape dots
        pattern = pattern.replace("*", r".*")  # Convert wildcard to regex
        # Convert single char wildcard to regex
        pattern = pattern.replace("?", r".")
        return re.match(pattern, host, re.I) is not None

    def proxy_bypass(host):
        """Returns True if the host should be bypassed, checking environment or registry settings."""
        if getproxies_environment():
            return _proxy_bypass_environment(host)
        else:
            return proxy_bypass_registry(host)


def dict_to_sequence(d):
    """
    Converts a dictionary into a sequence (list of tuples) if the input
    is dictionary-like, otherwise returns the input as-is.

    Parameters:
    - d (dict or iterable): Input dictionary-like object or any iterable.

    Returns:
    - Iterable: A sequence representing the dictionary items or the original input.
    """

    # Check if the input has the 'items' attribute (common in dictionaries)
    if hasattr(d, "items"):
        # If so, convert to a sequence of (key, value) tuples
        d = d.items()

    return d


def super_len(o):
    total_length = None
    current_position = 0

    if hasattr(o, "__len__"):
        total_length = len(o)

    elif hasattr(o, "len"):
        total_length = o.len

    elif hasattr(o, "fileno"):
        try:
            fileno = o.fileno()
        except (io.UnsupportedOperation, AttributeError):
            # AttributeError is a surprising exception, seeing as how we've just checked
            # that `hasattr(o, 'fileno')`.  It happens for objects obtained via
            # `Tarfile.extractfile()`, per issue 5229.
            pass
        else:
            total_length = os.fstat(fileno).st_size

            # Having used fstat to determine the file length, we need to
            # confirm that this file was opened up in binary mode.
            if "b" not in o.mode:
                warnings.warn(
                    ("Requests has determined the content-length for this "
                     "request using the binary size of the file: however, the "
                     "file has been opened in text mode (i.e. without the 'b' "
                     "flag in the mode). This may lead to an incorrect "
                     "content-length. In Requests 3.0, support will be removed "
                     "for files in text mode."), FileModeWarning, )

    if hasattr(o, "tell"):
        try:
            current_position = o.tell()
        except OSError:
            # This can happen in some weird situations, such as when the file
            # is actually a special file descriptor like stdin. In this
            # instance, we don't know what the length is, so set it to zero and
            # let requests chunk it instead.
            if total_length is not None:
                current_position = total_length
        else:
            if hasattr(o, "seek") and total_length is None:
                # StringIO and BytesIO have seek but no usable fileno
                try:
                    # seek to end of file
                    o.seek(0, 2)
                    total_length = o.tell()

                    # seek back to current position to support
                    # partially read file-like objects
                    o.seek(current_position or 0)
                except OSError:
                    total_length = 0

    if total_length is None:
        total_length = 0

    return max(0, total_length - current_position)


def get_netrc_auth(url, raise_errors=False):
    """
    Extracts and returns authentication details for a given URL from the .netrc file if available.

    Parameters:
    - url (str): The URL for which to retrieve the authentication details.
    - raise_errors (bool, optional): If True, raises parsing errors instead of silently passing. Defaults to False.

    Returns:
    - tuple: A tuple containing the username and password, or None if no authentication details were found.
    """
    netrc_file = get_netrc_file_location()
    if not netrc_file:
        return  # Early return if .netrc file location could not be determined

    try:
        from netrc import netrc, NetrcParseError
        host = get_host_from_url(url)
        if host:
            return get_credentials(netrc_file, host, raise_errors)
    # Handling ImportError and AttributeError for App Engine compatibility
    except (ImportError, AttributeError):
        pass


def get_netrc_file_location():
    """
    Determines the .netrc file location based on the NETRC environment variable or default locations.

    Returns:
    - str or None: The path to the .netrc file if found, otherwise None.
    """
    netrc_file = os.environ.get("NETRC")
    netrc_locations = (netrc_file,) if netrc_file else (
        os.path.expanduser(f"~/{f}") for f in NETRC_FILES)

    for location in netrc_locations:
        try:
            expanded_location = os.path.expanduser(location)
            if os.path.exists(expanded_location):
                return expanded_location
        except KeyError:  # Occurs when $HOME is undefined and getpwuid fails
            pass
    return None


def get_host_from_url(url):
    """
    Extracts the host from a given URL.

    Parameters:
    - url (str): The URL from which to extract the host.

    Returns:
    - str: The host part of the URL.
    """
    from urllib.parse import urlparse
    ri = urlparse(url)
    splitstr = b":" if not isinstance(url, str) else ":"
    return ri.netloc.split(splitstr)[0]


def get_credentials(netrc_path, host, raise_errors):
    """
    Retrieves the credentials for a given host from the .netrc file.

    Parameters:
    - netrc_path (str): The path to the .netrc file.
    - host (str): The host for which to retrieve credentials.
    - raise_errors (bool): Flag indicating whether to raise errors or not.

    Returns:
    - tuple: A tuple containing the username and password, or None if no credentials were found.
    """
    from netrc import NetrcParseError, netrc
    try:
        _netrc = netrc(netrc_path).authenticators(host)
        if _netrc:
            # Determines the correct login index
            login_index = 0 if _netrc[0] else 1
            return (_netrc[login_index], _netrc[2])
    except (NetrcParseError, OSError) as e:
        if raise_errors:
            raise e


def guess_filename(obj):
    """Tries to guess the filename of the given object."""
    name = getattr(obj, "name", None)
    if name and isinstance(name,
                           basestring) and name[0] != "<" and name[-1] != ">":
        return os.path.basename(name)


def extract_zipped_paths(path):
    """
    Check if a path is valid, and if not, checks if it refers to a file inside a zip archive.
    If it does, extracts that file to a temporary location and returns the new path.
    Otherwise, returns the original path.
    """
    if os.path.exists(path):
        # The given path exists, return as is
        return path

    archive, member = split_path_until_archive_found(path)

    if not is_valid_zip_archive(archive):
        # If the archive part of the path is not a valid zip file, return the
        # original path
        return path

    return extract_member_from_archive(archive, member)


def split_path_until_archive_found(path):
    """
    Split the given path into the first existing archive part and the member part.
    """
    archive, member = os.path.split(path)
    while archive and not os.path.exists(archive):
        archive, prefix = os.path.split(archive)
        if not prefix:
            break  # Prevent infinite loop
        member = "/".join([prefix, member])
    return archive, member


def is_valid_zip_archive(archive):
    """
    Check if the given archive path points to a valid zip file.
    """
    return zipfile.is_zipfile(archive)


def extract_member_from_archive(archive, member):
    """
    Extract the member from the provided archive into a temporary directory
    and return the path to this extracted file.
    """
    zip_file = zipfile.ZipFile(archive)
    if member not in zip_file.namelist():
        # If member is not in the zip file, return the original archive path
        return archive

    return write_member_to_temporary(member, zip_file)


def write_member_to_temporary(member, zip_file):
    """
    Write the specified member from the zip file to a temporary directory.
    """
    # Define the path for the extracted file
    tmp = tempfile.gettempdir()
    extracted_path = os.path.join(tmp, member.split("/")[-1])

    # Extract and write only if it doesn't already exist
    if not os.path.exists(extracted_path):
        with atomic_open(extracted_path) as file_handler:
            file_handler.write(zip_file.read(member))

    return extracted_path


@contextlib.contextmanager
def atomic_open(filename):
    """
    Securely open a file for writing. It ensures that the file writing process
    is atomic by using a temporary file. If the write is successful, the
    temporary file is renamed to the target filename. If an exception occurs,
    the temporary file is removed.

    Args:
    filename (str): The path to the file that needs to be written to, in an atomic fashion.

    Yields:
    file object: A temporary file handler that can be used to write to the file.
    """
    # Create a temporary file in the same directory as the target file
    tmp_descriptor, tmp_name = tempfile.mkstemp(dir=os.path.dirname(filename))
    try:
        # Open the temporary file descriptor for writing in binary mode
        with os.fdopen(tmp_descriptor, "wb") as tmp_handler:
            yield tmp_handler
        # If writing was successful, atomically replace the target file with
        # the temporary file
        os.replace(tmp_name, filename)
    except BaseException:
        # In case of any exception, remove the temporary file and re-raise the
        # exception
        os.remove(tmp_name)
        raise


def from_key_val_list(value):
    """
    Convert an object into an OrderedDict if it can be represented as such.
    Specifically, this function is designed to work with objects that can directly
    be converted to a dictionary-like structure, such as a list of 2-tuples.
    If the provided object cannot be represented as a dictionary, a ValueError is raised.

    Example Usage:
    -------------
    >>> from_key_val_list([('key', 'val')])
    OrderedDict([('key', 'val')])

    >>> from_key_val_list('string')
    ValueError: cannot encode objects that are not 2-tuples

    >>> from_key_val_list({'key': 'val'})
    OrderedDict([('key', 'val')])

    :param value: The object to convert into an OrderedDict.
    :type value: list, tuple, dict, str, bytes, bool, int
    :raises ValueError: If the object cannot be represented as a dictionary.
    :return: An OrderedDict representation of the input object, or None if the input is None.
    :rtype: OrderedDict or None
    """
    # Check if the input value is None and return None if true.
    if value is None:
        return None

    # Raise an error if the value is of a type that cannot be converted into a
    # dictionary.
    if isinstance(value, (str, bytes, bool, int)):
        raise ValueError("cannot encode objects that are not 2-tuples")

    # Attempt to convert the input value into an OrderedDict and return it.
    return OrderedDict(value)


def to_key_val_list(value):
    """
    Converts a given value to a list of key-value tuples if the value can be represented as a dictionary.

    This function can handle inputs that are already lists of tuples or dictionaries. Other data types,
    such as strings, bytes, booleans, and integers, will lead to a ValueError being raised as they cannot
    be directly represented as key-value pairs.

    Args:
        value: The input value to be converted.

    Returns:
        A list of tuples if the conversion is possible.

    Raises:
        ValueError: If the input value is of a type that cannot be represented as key-value pairs.

    Examples:
        >>> to_key_val_list([('key', 'val')])
        [('key', 'val')]
        >>> to_key_val_list({'key': 'val'})
        [('key', 'val')]
        >>> to_key_val_list('string')
        Traceback (most recent call last):
        ...
        ValueError: cannot encode objects that are not 2-tuples
    """

    # Return None immediately if the input is None
    if value is None:
        return None

    # Raise an error for types that cannot be converted to key-value pairs
    if isinstance(value, (str, bytes, bool, int)):
        raise ValueError("cannot encode objects that are not 2-tuples")

    # If value is a dictionary-like object, get its items
    if isinstance(value, Mapping):
        value = value.items()

    # Return the value as a list, which works for both list of tuples and
    # dictionary items
    return list(value)


# From mitsuhiko/werkzeug (used with permission).
def parse_list_header(value):
    """
    Parse lists as described by RFC 2068 Section 2.

    This method parses comma-separated lists where elements may include quoted-strings.
    It handles quoted-strings containing commas and non-quoted strings with mid-quotes, removing
    quotes after parsing. It is similar to parse_set_header but allows duplicate items and
    preserves case sensitivity. The function produces a list derived from the input string value.

    Example:
    >>> parse_list_header('token, "quoted value"')
    ['token', 'quoted value']

    To create a header from the list, use the dump_header function.

    Parameters:
    - value: A string containing the list header.

    Returns:
    A list parsed from the input string.
    """

    def remove_enclosing_quotes(item):
        """Remove quotes that enclose an item if present."""
        if item.startswith('"') and item.endswith('"'):
            return unquote_header_value(item[1:-1])
        return item

    result = [_process_item(item) for item in _parse_list_header(value)]
    return result


def _process_item(item):
    """
    Process individual items found by parsing the list header. Removes enclosing quotes if present.

    Parameters:
    - item: A string item from the list header.

    Returns:
    The item with enclosing quotes removed if they were present.
    """
    return remove_enclosing_quotes(item)


# From mitsuhiko/werkzeug (used with permission).
def parse_dict_header(header_str):
    """
    Parse a string containing header-like key-value pairs into a dictionary.

    This function supports headers as described by RFC 2068 Section 2, allowing
    for values to be optionally quoted strings.

    Args:
        header_str (str): A string representation of header key-value pairs.

    Returns:
        dict: A dictionary with keys and values parsed from header_str.
    """
    result = {}
    # Split the header string into individual key-value pair strings
    items = _parse_list_header(header_str)

    for item in items:
        # If there's no equal sign, the value is None
        if "=" not in item:
            result[item] = None
        else:
            key, value = _split_key_value(item)
            result[key] = _process_value(value)

    return result


def _split_key_value(item):
    """
    Splits an item into a key and value at the first equals sign.

    Args:
        item (str): The item to split.

    Returns:
        tuple: A tuple containing the key and value as two elements.
    """
    # Split the item into key and value parts at the first '='
    return item.split("=", 1)


def _process_value(value):
    """
    Process the value part of a key-value pair, unquoting if necessary.

    Args:
        value (str): The value to process.

    Returns:
        str: The processed value, unquoted if it was quoted.
    """
    # If the value is enclosed in quotes, remove them, else return as is
    if value.startswith('"') and value.endswith('"'):
        return unquote_header_value(value[1:-1])
    return value


# From mitsuhiko/werkzeug (used with permission).
def unquote_header_value(value, is_filename=False):
    """
    Unquotes a header value, specifically for handling cases with filenames
    and peculiar browser behaviors, like Internet Explorer's handling of file paths.

    This method aims to reverse the process of quoting header values but
    adheres to the quirks of browser implementations rather than strict
    RFC compliance.

    Parameters:
    - value (str): The quoted header value to be unquoted.
    - is_filename (bool): Specifies if the header value is a filename.

    Returns:
    - str: The unquoted header value.
    """
    if _is_quoted(value):
        value = _strip_quotes(value)
        value = _handle_filenames(value, is_filename)
    return value


def _is_quoted(value):
    """Check if the given value is quoted at the beginning and end."""
    # Check if the first and last characters of the value are quotes
    return value and value[0] == value[-1] == '"'


def _strip_quotes(value):
    """Remove the first and last characters, typically quotes, from the value."""
    # Remove the enclosing quotes from the value
    return value[1:-1]


def _handle_filenames(value, is_filename):
    """
    Handles special cases for filenames, including UNC paths, to ensure
    correct formatting and unquoting.

    Parameters:
    - value (str): The header value potentially representing a filename.
    - is_filename (bool): Indicates if the value is expected to be a filename.

    Returns:
    - str: The properly unquoted and formatted header value.
    """
    # Check if the value does not start with a UNC path indicator or it's not
    # a filename
    if not is_filename or not value.startswith("\\\\"):
        # Replace escaped backslashes and escaped quotes with their literals
        return value.replace("\\\\", "\\").replace('\\"', '"')
    return value


def dict_from_cookiejar(cj):
    """
    Converts cookies from a CookieJar into a dictionary.

    This function iterates through all cookies in a CookieJar
    and stores their names and values in a dictionary. The resulting
    dictionary maps each cookie's name to its value.

    :param cj: The CookieJar object containing cookies.
    :rtype: dict
    """
    # Initialize an empty dictionary to store cookie names and values
    cookie_dict = {}

    # Iterate through each cookie in the CookieJar
    for cookie in cj:
        # Assign the cookie's name as the key and its value as the value in
        # cookie_dict
        cookie_dict[cookie.name] = cookie.value

    # Return the dictionary containing all cookies' names and values
    return cookie_dict


def add_dict_to_cookiejar(cj, cookie_dict):
    """
    Returns a CookieJar from a key/value dictionary.

    This function takes a dictionary of cookies and adds them to a specified CookieJar object.

    :param cj: CookieJar to insert cookies into.
    :param cookie_dict: Dict of key/values to insert into CookieJar.
    :rtype: CookieJar
    """
    # Convert cookie dictionary into CookieJar and return it
    return cookiejar_from_dict(cookie_dict, cj)


def get_encodings_from_content(content):
    """
    Retrieves a list of encoding declarations found within an HTML or XML document.

    This function searches for character encoding declarations within the provided content string
    by parsing and finding matches for the 'charset' attribute in meta tags and XML declarations.
    The search is not exhaustive and is limited to commonly used patterns.

    Args:
        content (bytes): The content byte string from which encodings will be extracted.

    Returns:
        list: A list of all found encodings as strings. The list could be empty if no encodings are found.

    Note:
        This function is deprecated and will be removed in a future version.
    """

    # Display a deprecation warning when this function is used.
    warnings.warn(
        (
            "In requests 3.0, get_encodings_from_content will be removed. For "
            "more information, please see the discussion on issue #2266. (This"
            " warning should only appear once.)"
        ),
        DeprecationWarning,
    )

    # Compile regular expressions for finding charset encodings in different contexts.
    # For <meta> tags with charset attribute.
    charset_re = re.compile(r'<meta.*?charset=["\']*(.+?)["\'>]', flags=re.I)
    # For <meta> tags with HTTP-Equiv="Content-Type" and a charset declaration.
    pragma_re = re.compile(
        r'<meta.*?content=["\']*;?charset=(.+?)["\'>]',
        flags=re.I)
    # For XML declarations that include an encoding attribute.
    xml_re = re.compile(r'^<\?xml.*?encoding=["\']*(.+?)["\'>]')

    # Find all matches for each compiled regex within the provided content.
    charset_matches = charset_re.findall(content)
    pragma_matches = pragma_re.findall(content)
    xml_matches = xml_re.findall(content)

    # Combine all found matches into a single list and return it.
    all_matches = charset_matches + pragma_matches + xml_matches
    return all_matches


def _parse_content_type_header(header):
    """
    Parses the Content-Type header to extract the content type and parameters.

    :param header: A string representing the Content-Type header value.
    :return: A tuple containing the content type as a string and a dictionary of parameters.
    """
    content_type, parameters = _extract_content_type_and_params(header)
    params_dict = _parse_parameters(parameters)
    return content_type, params_dict


def _extract_content_type_and_params(header):
    """
    Splits the header into content type and parameters.

    :param header: The full Content-Type header string.
    :return: A tuple of the content type and a list of parameter strings.
    """
    tokens = header.split(";")
    content_type = tokens[0].strip()
    params = tokens[1:]
    return content_type, params


def _parse_parameters(params):
    """
    Parses the list of parameter strings into a dictionary.

    :param params: A list of parameter strings.
    :return: A dictionary with parameter names as keys and their values.
    """
    params_dict = {}
    for param in params:
        param = param.strip()
        if param:
            key, value = _parse_single_param(param)
            params_dict[key.lower()] = value
    return params_dict


def _parse_single_param(param):
    """
    Parses a single parameter string into its key and value.

    :param param: A single parameter string.
    :return: A tuple of the parameter's key and value.
    """
    items_to_strip = "\"' "
    default_value = True
    if "=" in param:
        key, value = param.split("=", 1)
        key = key.strip(items_to_strip).lower()
        value = value.strip(items_to_strip)
    else:
        key = param.strip(items_to_strip).lower()
        value = default_value
    return key, value


def get_encoding_from_headers(headers):
    """Returns encodings from given HTTP Header Dict.

    This function parses the 'Content-Type' header to extract and return the charset encoding if specified.
    It falls back to default charsets for 'text' and 'application/json' types if charset is not explicitly mentioned.

    :param headers: Dictionary containing the HTTP headers.
    :rtype: str or None
    """
    content_type = headers.get("content-type")

    # Return None if 'Content-Type' header is missing or empty
    if not content_type:
        return None

    content_type, params = _parse_content_type_header(content_type)

    return _get_charset_from_params(content_type, params)


def _get_charset_from_params(content_type, params):
    """Extracts charset from the parameters of the Content-Type header or assigns default value based on the media type.

    :param content_type: The media type from the 'Content-Type' header
    :param params: A dictionary of all parameters specified in the 'Content-Type' header
    :rtype: str
    """
    # Return the charset value after stripping any leading/trailing single or
    # double quotes
    if "charset" in params:
        return params["charset"].strip("'\"")

    # Default charset for 'text' media type
    if "text" in content_type:
        return "ISO-8859-1"

    # Default charset for 'application/json' media type as per RFC 4627
    if "application/json" in content_type:
        return "utf-8"


def stream_decode_response_unicode(iterator, response):
    """
    Generator function to stream decode each chunk of the iterator using the response encoding.

    This function yields each decoded chunk of data. It handles the case where the encoding is not
    specified by yielding raw bytes from the iterator. If the encoding is specified, it decodes each chunk
    using an incremental decoder, which replaces undecodable bytes.

    Parameters:
    - iterator: Iterable yielding bytes, the content to be decoded.
    - response: Response object that may contain the 'encoding' attribute.

    Yields:
    - Decoded strings from the given bytes iterator, according to the response's encoding.
    """

    # Check if the response encoding is not specified and yield raw bytes
    # directly from the iterator.
    if response.encoding is None:
        yield from iterator
    else:
        # Create an incremental decoder for the specified encoding, replacing
        # errors.
        decoder = codecs.getincrementaldecoder(
            response.encoding)(errors="replace")

        # Decode each chunk from the iterator, yielding non-empty results.
        for chunk in iterator:
            decoded_chunk = decoder.decode(chunk)
            if decoded_chunk:
                yield decoded_chunk

        # Decode any remaining input to handle final bits, yielding if
        # non-empty.
        final_chunk = decoder.decode(b"", final=True)
        if final_chunk:
            yield final_chunk


def iter_slices(string, slice_length):
    """
    Generator that yields slices of a given string with the specified length.

    Args:
    - string (str): The string to be sliced.
    - slice_length (int): The length of each slice. If None or non-positive, the entire string is returned.
    """
    # Initialize the current position for slicing
    position = 0

    # Ensure valid slice length; default to entire string length if invalid
    if slice_length is None or slice_length <= 0:
        slice_length = len(string)

    # Loop to slice the string and yield slices until the end of the string is
    # reached
    while position < len(string):
        # Yield a slice of the string from the current position to the desired
        # slice length
        yield string[position: position + slice_length]
        # Move the current position forward by the slice length for the next
        # iteration
        position += slice_length


def get_unicode_content(response):
    """
    Returns the requested content back in unicode format.

    Tries to decode the response content using:
    1. The charset specified in the Content-Type header of the response.
    2. A fallback mechanism that replaces invalid unicode characters.

    Deprecation warning is issued as this function will be removed in future versions.

    :param response: The HTTP response object from which to extract unicode content.
    :rtype: str
    """
    issue_warning_for_deprecation()

    encoding = get_encoding_from_headers(response.headers)

    # Try to decode based on Content-Type charset
    unicode_content = decode_with_charset(response, encoding)
    if unicode_content is not None:
        return unicode_content

    # Fallback: decode replacing errors
    return decode_with_fallback(response)


def issue_warning_for_deprecation():
    """Issues a deprecation warning for the use of this function."""
    warnings.warn(
        (
            "In requests 3.0, get_unicode_content will be removed. For "
            "more information, please see the discussion on issue #2266. "
            "(This warning should only appear once.)"
        ),
        DeprecationWarning,
    )


def decode_with_charset(response, encoding):
    """
    Attempts to decode the response content using the specified charset.

    :param response: The HTTP response object.
    :param encoding: The charset encoding to use for decoding.
    :rtype: str or None
    """
    if encoding:
        try:
            return str(response.content, encoding)
        except UnicodeError:
            # Decoding failed, return None to indicate failure
            pass
    return None


def decode_with_fallback(response):
    """
    Attempts to decode the response content using a fallback mechanism
    that replaces undecodable characters.

    :param response: The HTTP response object.
    :rtype: str
    """
    try:
        return str(response.content, errors="replace")
    except TypeError:
        # In case of TypeError, return the raw content
        return response.content


# The unreserved URI characters (RFC 3986)
UNRESERVED_SET = frozenset(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz" + "0123456789-._~"
)


def unquote_unreserved(uri):
    """
    Un-escape any percent-escape sequences in a URI that are unreserved
    characters. This leaves all reserved, illegal, and non-ASCII bytes encoded.

    :param uri: The URI string to be processed.
    :rtype: str
    """
    parts = uri.split("%")  # Split the URI at every percent sign.
    for i in range(1, len(parts)):
        # Get the hex pair following the percent sign.
        hex_pair = parts[i][0:2]
        # Process and rebuild the URI part.
        parts[i] = process_hex_pair(hex_pair, parts[i])
    return "".join(parts)  # Reassemble the URI parts back into a complete URI.


def process_hex_pair(hex_pair, uri_part):
    """
    Process a hex pair and return the appropriate string.
    If the hex pair represents an unreserved character, it's un-escaped,
    otherwise it's left as is.

    :param hex_pair: The hex pair to be processed.
    :param uri_part: The remainder of the URI part after the hex pair.
    :rtype: str
    """
    if is_valid_hex_pair(hex_pair):  # Check if the hex pair is valid.
        char = hex_to_char(hex_pair)  # Convert hex pair to character.
        if char in UNRESERVED_SET:  # Check if the character is unreserved.
            # Un-escape by replacing hex pair with actual character.
            return char + uri_part[2:]
    return f"%{uri_part}"  # Leave the URI part unchanged.


def is_valid_hex_pair(hex_pair):
    """
    Check if the hex pair is valid (two characters long and alphanumeric).

    :param hex_pair: The hex pair to be checked.
    :rtype: bool
    """
    return len(hex_pair) == 2 and hex_pair.isalnum()  # Validate hex pair.


def hex_to_char(hex_pair):
    """
    Convert a hex pair to its corresponding character.

    :param hex_pair: The hex pair to be converted.
    :rtype: str
    """
    try:
        # Convert hex pair to integer then to character.
        return chr(int(hex_pair, 16))
    except ValueError:
        # Raise exception for invalid hex.
        raise InvalidURL(f"Invalid percent-escape sequence: '{hex_pair}'")


def requote_uri(uri):
    """
    Re-quote the given URI to ensure it is fully and consistently quoted.

    This function first attempts to unquote unreserved characters and then requote the URI,
    ensuring that illegal characters are quoted while reserved, unreserved, and '%' characters are not re-quoted.

    If the initial unquote/quote cycle fails due to an InvalidURL exception,
    the function then attempts to directly quote the URI while ensuring that any '%' characters are properly quoted.

    :param uri: The original URI to be requoted.
    :rtype: str
    :return: The requoted URI.
    """
    # Define a set of characters that are safe to include in a URI including
    # the percent sign
    safe_with_percent = "!#$%&'()*+,/:;=?@[]~"
    # Define another set of safe characters excluding the percent sign for
    # special case handling
    safe_without_percent = "!#$&'()*+,/:;=?@[]~"
    try:
        # Attempt to unquote reserved characters and requote the URI
        # This is the primary method of ensuring the URI is correctly quoted
        return quote(unquote_unreserved(uri), safe=safe_with_percent)
    except InvalidURL:
        # If there's an error with unquoting, directly quote the URI
        # This ensures that any stray '%' characters are correctly handled
        return quote(uri, safe=safe_without_percent)


def dotted_netmask(mask_bits):
    """Generate a dotted decimal netmask from mask bits.

    Example: given 24, it will return 255.255.255.0.
    """
    # Calculate the binary netmask based on the bits provided
    netmask_bin = '1' * mask_bits + '0' * (32 - mask_bits)
    # Split the binary netmask into octets for conversion
    octets = [netmask_bin[i:i + 8] for i in range(0, 32, 8)]
    # Convert each binary octet into decimal and format it as dotted decimal
    return '.'.join(str(int(octet, 2)) for octet in octets)


def address_to_long(ip_address):
    """Converts an IP address from its dotted decimal form to a long integer."""
    return struct.unpack("=L", socket.inet_aton(ip_address))[0]


def network_and_mask(net_description):
    """Extract the network address and the netmask from a CIDR notation."""
    network_address, bits = net_description.split("/")
    # Convert netmask to dotted decimal format
    netmask = dotted_netmask(int(bits))
    return network_address, netmask


def calculate_network_address(ip_long, netmask_long):
    """Calculates the network address by applying the netmask on the IP."""
    return ip_long & netmask_long


def address_in_network(ip, net):
    """Check if an IP address belongs to a given network subnet.

    Args:
        ip (str): The IP address to check.
        net (str): The network subnet in CIDR notation.

    Returns:
        bool: True if the IP address belongs to the subnet, False otherwise.
    """
    # Convert IP address from dotted decimal to long
    ip_long = address_to_long(ip)
    # Extract network address and netmask from subnet description
    netaddr, netmask = network_and_mask(net)
    # Convert netmask and network address to long
    netmask_long = address_to_long(netmask)
    network_long = address_to_long(netaddr)
    # Calculate the network address of the given IP
    ip_network_long = calculate_network_address(ip_long, netmask_long)
    # Compare the network address of the IP with that of the subnet
    return ip_network_long == calculate_network_address(
        network_long, netmask_long)


def dotted_netmask(mask):
    """
    Converts mask from /xx format to xxx.xxx.xxx.xxx format.
    Example: if mask is 24, function returns 255.255.255.0

    Args:
        mask (int): The subnet mask in /xx format.

    Returns:
        str: The subnet mask in dotted decimal format.
    """
    # Calculate the binary representation of the mask, with the `mask` number
    # of leading 1 bits and the rest filled with 0 bits.
    netmask_binary = 0xFFFFFFFF ^ (1 << 32 - mask) - 1

    # Convert the binary representation of the mask to a 4-byte packed binary
    # format.
    netmask_packed = struct.pack(">I", netmask_binary)

    # Convert the packed binary format of the mask to a dotted decimal string.
    netmask_dotted_decimal = socket.inet_ntoa(netmask_packed)

    return netmask_dotted_decimal


def is_ipv4_address(string_ip):
    """
    Check if the given string is a valid IPv4 address.

    Args:
        string_ip (str): The string representation of an IP address.

    Returns:
        bool: True if the string is a valid IPv4 address, False otherwise.
    """
    try:
        # Try to convert the string IP to its binary form. If fails, it's not a
        # valid IPv4.
        socket.inet_aton(string_ip)
    except OSError:
        # If conversion fails, it means the string_ip is not a valid IPv4
        # address.
        return False
    # If the try block succeeds, it means the string_ip is a valid IPv4
    # address.
    return True


def is_valid_cidr(string_network):
    """
    Checks if the given string is a valid CIDR network format.

    :param string_network: CIDR format string to be validated.
    :rtype: bool
    """
    def is_valid_mask(mask):
        """
        Checks if the subnet mask is within the valid range.

        :param mask: subnet mask as an integer.
        :rtype: bool
        """
        return 1 <= mask <= 32

    def is_valid_ip(ip_address):
        """
        Validates the IP address part of the CIDR.

        :param ip_address: String containing an IP address.
        :rtype: bool
        """
        try:
            socket.inet_aton(ip_address)
            return True
        except OSError:
            return False

    if string_network.count("/") != 1:
        return False

    ip_part, mask_part = string_network.split("/")
    try:
        mask = int(mask_part)
    except ValueError:
        return False

    if not is_valid_mask(mask):
        return False

    return is_valid_ip(ip_part)


@contextlib.contextmanager
def set_environ(env_name, value):
    """
    Temporarily set the environment variable 'env_name' to a new 'value',
    restore it to its original value after the context block.

    Args:
        env_name (str): The name of the environment variable to modify.
        value (str): The new value for the environment variable. If None, no action is taken.
    """

    # Check if a new value is provided to avoid unnecessary operations
    if value is not None:
        # Save the old value to restore it later
        old_value = os.environ.get(env_name)
        # Set the new value of the environment variable
        os.environ[env_name] = value
        value_changed = True
    else:
        value_changed = False

    try:
        # Yield control back to the context block
        yield
    finally:
        # If the value was changed, restore the original or delete if it didn't
        # exist before
        if value_changed:
            restore_environment_variable(env_name, old_value)


def restore_environment_variable(env_name, old_value):
    """
    Restore the environment variable 'env_name' to its 'old_value'.
    If 'old_value' is None, the variable is deleted.

    Args:
        env_name (str): The name of the environment variable to restore.
        old_value (str|None): The original value of the environment variable before it was changed.
                              If None, the variable is assumed to be new and is deleted.
    """
    if old_value is None:
        # Delete the environment variable if it didn't exist before
        del os.environ[env_name]
    else:
        # Restore the original value of the environment variable
        os.environ[env_name] = old_value


def should_bypass_proxies(url, no_proxy):
    """
    Returns whether we should bypass proxies or not.

    :rtype: bool
    """
    # Prioritize lowercase environment variables over uppercase
    # to keep a consistent behaviour with other http projects (curl, wget).
    def get_proxy(key):
        return os.environ.get(key) or os.environ.get(key.upper())

    # First check whether no_proxy is defined. If it is, check that the URL
    # we're getting isn't in the no_proxy list.
    no_proxy_arg = no_proxy
    if no_proxy is None:
        no_proxy = get_proxy("no_proxy")
    parsed = urlparse(url)

    if parsed.hostname is None:
        # URLs don't always have hostnames, e.g. file:/// urls.
        return True

    if no_proxy:
        # We need to check whether we match here. We need to see if we match
        # the end of the hostname, both with and without the port.
        no_proxy = (
            host for host in no_proxy.replace(
                " ", "").split(",") if host)

        if is_ipv4_address(parsed.hostname):
            for proxy_ip in no_proxy:
                if is_valid_cidr(proxy_ip):
                    if address_in_network(parsed.hostname, proxy_ip):
                        return True
                elif parsed.hostname == proxy_ip:
                    # If no_proxy ip was defined in plain IP notation instead of cidr notation &
                    # matches the IP of the index
                    return True
        else:
            host_with_port = parsed.hostname
            if parsed.port:
                host_with_port += f":{parsed.port}"

            for host in no_proxy:
                if parsed.hostname.endswith(
                        host) or host_with_port.endswith(host):
                    # The URL does match something in no_proxy, so we don't want
                    # to apply the proxies on this URL.
                    return True

    with set_environ("no_proxy", no_proxy_arg):
        # parsed.hostname can be `None` in cases such as a file URI.
        try:
            bypass = proxy_bypass(parsed.hostname)
        except (TypeError, socket.gaierror):
            bypass = False

    if bypass:
        return True

    return False


def get_environ_proxies(url, no_proxy=None):
    """
    Return a dictionary of environment proxies applicable to the given URL.

    This function checks if the URL should bypass the proxies based on
    a given no_proxy list. If the URL should bypass proxies,
    an empty dictionary is returned. Otherwise, the environment proxies are returned.

    :param url: The URL for which to get the proxies.
    :type url: str
    :param no_proxy: A list or string of URLs that should not go through any proxy.
    :type no_proxy: str or list
    :rtype: dict
    """
    # Check if the URL should bypass proxies, if so return an empty dictionary
    if should_bypass_proxies(url, no_proxy=no_proxy):
        return {}
    else:
        # Fetch and return the system-wide proxy settings
        return getproxies()


def select_proxy(url, proxies):
    """
    Select an appropriate proxy for the given URL from a provided dictionary.

    Args:
    - url: The URL for which the proxy needs to be selected.
    - proxies: A dictionary mapping scheme or scheme+hostname to their proxy URLs.

    Returns:
    - The proxy URL as a string if a suitable match is found, else None.
    """
    # Ensure proxies dict is not None
    proxies = proxies or {}
    # Parse the URL to extract its components
    urlparts = urlparse(url)

    # If URL without hostname, return proxy for its scheme or 'all'
    if urlparts.hostname is None:
        return proxies.get(urlparts.scheme, proxies.get("all"))

    # Potential proxy keys in order of preference
    proxy_keys = [
        urlparts.scheme + "://" + urlparts.hostname,  # Full scheme + hostname
        urlparts.scheme,  # Just scheme
        "all://" + urlparts.hostname,  # Any scheme with specific hostname
        "all",  # Any scheme, any hostname
    ]

    # Loop through potential keys and return the first match
    for proxy_key in proxy_keys:
        if proxy_key in proxies:
            return proxies[proxy_key]

    # If no proxy matches, return None
    return None


def resolve_proxies(request, proxies, trust_env=True):
    """
    Resolve the final proxies dictionary to be used in a request.

    If `trust_env` is True, this method will attempt to fetch proxies from the
    environment if they are not explicitly provided in the `proxies` parameter.

    :param request: The request object containing the URL.
    :param proxies: A dictionary mapping from scheme to proxy URL.
    :param trust_env: Whether to read proxy settings from the environment.

    :return: A dictionary of resolved proxies.
    """
    # Initialize proxies dictionary if not provided
    proxies = proxies or {}

    # Copy given proxies to avoid modifying the original
    resolved_proxies = proxies.copy()

    # Conditionally resolve environment proxies
    if trust_env:
        update_proxies_from_env(request.url, resolved_proxies)

    return resolved_proxies


def update_proxies_from_env(url, proxies):
    """
    Update the proxies dictionary based on environment settings.

    This function will update the given `proxies` dictionary in-place by adding or
    replacing proxy settings based on environment variables and the provided URL.

    :param url: The URL for which to resolve a proxy.
    :param proxies: The current dictionary of proxies to be updated.
    """
    scheme = urlparse(url).scheme  # Extract scheme from URL
    # Get no_proxy settings from current proxies
    no_proxy = proxies.get("no_proxy")
    environ_proxies = get_environ_proxies(
        url, no_proxy=no_proxy)  # Fetch environment proxies

    # Determine the appropriate proxy from environment, falling back to 'all'
    proxy = environ_proxies.get(scheme, environ_proxies.get("all"))

    # Only set the proxy if one was found
    if proxy:
        proxies.setdefault(scheme, proxy)


def default_user_agent(name="python-requests"):
    """
    Generates a default user-agent string.

    Constructs a default user-agent string by appending the version number to the given name.
    If no name is provided, "python-requests" is used by default.

    Parameters:
    - name (str): The base name to use for the user-agent, defaults to "python-requests".

    Returns:
    - str: The constructed user-agent string.
    """
    # Construct and return the user-agent string by combining the name with
    # the version number.
    return f"{name}/{__version__}"


def default_headers():
    """
    Creates and returns a dictionary of default HTTP headers for web requests.

    This includes a User-Agent, Accept-Encoding, Accept, and Connection header.
    The User-Agent is defined by the `default_user_agent()` function,
    and `DEFAULT_ACCEPT_ENCODING` is a predefined constant.

    :rtype: requests.structures.CaseInsensitiveDict
    """
    # Create a CaseInsensitiveDict of default HTTP headers
    return CaseInsensitiveDict(
        {
            # Set the user agent using a predefined function
            "User-Agent": default_user_agent(),
            # Use a constant predefined acceptable encoding
            "Accept-Encoding": DEFAULT_ACCEPT_ENCODING,
            "Accept": "*/*",  # Accept any content type
            "Connection": "keep-alive",  # Keep the connection alive
        }
    )


def parse_header_links(header_value):
    """
    Parses a string containing multiple link headers into a list of dicts.

    Each dict contains the URL and parameters of a link. For example, it converts the string:

    'Link: <http://.../front.jpeg>; rel="front"; type="image/jpeg",<http://.../back.jpeg>; rel="back";type="image/jpeg"'

    into:

    [
        {'url': 'http://.../front.jpeg', 'rel': 'front', 'type': 'image/jpeg'},
        {'url': 'http://.../back.jpeg', 'rel': 'back', 'type': 'image/jpeg'}
    ]

    :param header_value: A string containing the link header information.
    :rtype: list
    """
    # Characters to be replaced in the parsing process
    replace_chars = " '\""

    # Strip unwanted characters from the beginning and end
    header_value = header_value.strip(replace_chars)

    # Return an empty list if header_value is empty
    if not header_value:
        return []

    links = []  # Initialize list to hold parsed links
    for val in re.split(", *<", header_value):  # Split the value by , and <
        # Separate and parse the URL and parameters
        url, params = _parse_val(val)
        # Parse individual link into a dictionary
        link = _parse_link(url, params, replace_chars)
        links.append(link)  # Add the parsed link to links list

    return links


def _parse_val(val):
    """
    Parses a value into a URL and parameters.

    :param val: A part of the header string containing a URL and possibly parameters.
    :rtype: tuple
    """
    try:
        url, params = val.split(";", 1)
    except ValueError:  # Handle cases where there is no ';' to split on
        url, params = val, ""

    # Return URL and params with extra chars stripped from URL
    return url.strip("<> '\""), params


def _parse_link(url, params, replace_chars):
    """
    Parses URL and parameters string into a link dictionary.

    :param url: The URL string to be included in the link dict.
    :param params: The parameters string associated with the URL.
    :param replace_chars: Characters to be stripped from the URL and parameters.
    :rtype: dict
    """
    link = {"url": url}  # Initialize link dict with URL

    for param in params.split(";"):  # Split parameters by ';'
        # Parse key-value pair from param
        key, value = _parse_param(param, replace_chars)
        if key:  # If a valid key is returned
            link[key] = value  # Add key-value pair to link dict

    return link


def _parse_param(param, replace_chars):
    """
    Parses a parameter string into a key and value.

    :param param: The parameter string to parse.
    :param replace_chars: Characters to be stripped from key and value.
    :rtype: tuple
    """
    try:
        key, value = param.split("=")
        return key.strip(replace_chars), value.strip(
            replace_chars)  # Return stripped key and value
    except ValueError:  # Handle cases where there is no '=' to split on
        return None, None  # Return None for both key and value


# Null bytes; no need to recreate these on each call to guess_json_utf
_null = "\x00".encode("ascii")  # encoding to ASCII for Python 3
_null2 = _null * 2
_null3 = _null * 3


def guess_json_utf(data):
    """
    Guess the encoding of a JSON string by inspecting its byte order mark (BOM) or the
    presence and position of null bytes in the first four bytes.

    :param data: A byte string containing the JSON data.
    :rtype: str or None
    :returns: The guessed encoding format as a string, or None if the encoding could not be determined.
    """
    sample = data[:4]  # Extract the first four bytes for analysis.

    # First, check for BOM markers for different UTF encodings.
    encoding = check_for_bom(sample)
    if encoding:
        return encoding

    # If no BOM is found, inspect the presence and arrangement of null bytes.
    return check_for_null_bytes(sample)


def check_for_bom(sample):
    """
    Check the sample for a UTF Byte Order Mark (BOM).

    :param sample: The first four bytes of the data.
    :rtype: str or None
    :returns: The encoding if a BOM is found, otherwise None.
    """
    if sample in (codecs.BOM_UTF32_LE, codecs.BOM_UTF32_BE):
        return "utf-32"  # BOM for UTF-32
    if sample[:3] == codecs.BOM_UTF8:
        return "utf-8-sig"  # BOM for UTF-8 (with BOM, MS style)
    if sample[:2] in (codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE):
        return "utf-16"  # BOM for UTF-16
    return None  # No BOM found


def check_for_null_bytes(sample):
    """
    Determine encoding by analyzing the presence and distribution of null bytes in the sample.

    :param sample: The first four bytes of the data.
    :rtype: str or None
    :returns: The guessed encoding based on null byte analysis, or None if indeterminate.
    """
    null_count = sample.count(_null)  # Count the number of null bytes

    if null_count == 0:
        return "utf-8"  # No null bytes indicates UTF-8
    if null_count == 2:
        # Special handling for two null bytes
        return analyze_two_null_bytes(sample)
    if null_count == 3:
        # Special handling for three null bytes
        return analyze_three_null_bytes(sample)

    # Fall back if the pattern of null bytes doesn't match known encodings
    return None


def analyze_two_null_bytes(sample):
    """
    Determine the encoding when two null bytes are present in the sample.

    :param sample: The first four bytes of the data.
    :rtype: str or None
    :returns: The guessed encoding or None if indeterminate.
    """
    if sample[::2] == _null2:  # 1st and 3rd bytes are null
        return "utf-16-be"
    if sample[1::2] == _null2:  # 2nd and 4th bytes are null
        return "utf-16-le"
    return None  # Pattern not recognized


def analyze_three_null_bytes(sample):
    """
    Determine the encoding when three null bytes are present in the sample.

    :param sample: The first four bytes of the data.
    :rtype: str or None
    :returns: The guessed encoding or None if indeterminate.
    """
    if sample[:3] == _null3:
        return "utf-32-be"
    if sample[1:] == _null3:
        return "utf-32-le"
    return None  # Pattern not recognized


def prepend_scheme_if_needed(url, new_scheme):
    """
    Given a URL that may or may not have a scheme, prepend the given scheme.
    Does not replace a present scheme with the one provided as an argument.

    :param url: The URL to prepend the scheme to
    :param new_scheme: The scheme to prepend to the URL if it doesn't already have one
    :rtype: str
    """
    # Parse the given URL
    parsed = parse_url(url)

    # Extract parts of the URL
    scheme, auth, host, port, path, query, fragment = parsed

    # Fix potential parsing issue related to netloc and path
    netloc, path = fix_parsing_issue(parsed.netloc, path)

    # Include auth in netloc if auth exists
    if auth:
        netloc = include_auth_in_netloc(auth, netloc)

    # Prepend new scheme if the URL does not have one
    if scheme is None:
        scheme = new_scheme

    # Ensure path is not None
    path = ensure_path_is_not_none(path)

    # Reconstruct URL with potentially modified parts
    return urlunparse((scheme, netloc, path, "", query, fragment))


def fix_parsing_issue(netloc, original_path):
    """
    Fixes the parsing issue where netloc might be absent, swapping netloc and path if necessary.

    :param netloc: The net location part of the URL
    :param original_path: The path part of the URL
    :return: Corrected netloc and path
    """
    if not netloc:
        # Swap netloc and path if netloc is absent
        return original_path, netloc
    else:
        return netloc, original_path


def ensure_path_is_not_none(path):
    """
    Ensure the path is not None, returning an empty string if it is.

    :param path: The path to ensure is not None
    :return: Path or an empty string if the path was None
    """
    if path is None:
        return ""
    else:
        return path


def include_auth_in_netloc(auth, original_netloc):
    """
    Includes the auth part in the net location if auth exists.

    :param auth: The auth part of the URL
    :param original_netloc: The original net location of the URL
    :return: Net location with auth included if auth exists
    """
    # Concatenate auth and netloc with '@' if auth exists
    return "@".join([auth, original_netloc])


def get_auth_from_url(url):
    """
    Given a URL with authentication components, extract them into a tuple of
    username, password. If the URL doesn't contain authentication information,
    return an empty string for both username and password.

    Parameters:
    - url (str): The URL from which authentication information is to be extracted.

    Returns:
    - tuple: A tuple containing the username and password extracted from the URL.
    """
    # Parse the given URL into components
    parsed = urlparse(url)

    try:
        # Attempt to extract and decode the authentication information
        # Unquote is used to convert percent-encoded chars back to normal
        username = unquote(parsed.username)
        password = unquote(parsed.password)
        auth = (username, password)
    except (AttributeError, TypeError):
        # If parsing fails or authentication info is missing, return empty
        # strings
        auth = ("", "")

    return auth


def check_header_validity(header):
    """
    Verifies that each part of the header (name and value) does not contain
    leading whitespace, reserved characters, or return characters.

    Parameters:
    header (tuple): The header to check, in the format (name, value).
    """
    name, value = header  # Extracts name and value from header tuple.

    # Validates the name part of the header.
    _validate_header_part(header, name, part_index=0)

    # Validates the value part of the header.
    _validate_header_part(header, value, part_index=1)


def _validate_header_part(header, header_part, header_validator_index):
    """
    Validates a part of the header (either name or value) against predefined validators.

    Raises InvalidHeader exception if the header part is not a string or bytes,
    or if it doesn't match the corresponding validator (indicating invalid characters
    or leading whitespace).

    :param header: The full header being validated.
    :param header_part: The part of the header (name or value) being validated.
    :param header_validator_index: Index to determine which validator to use (0 for name, 1 for value).
    """
    # Determine the validator based on the type of header_part
    validator = _get_validator_for_header_part(
        header_part, header_validator_index)

    # Validate the header part against the chosen validator
    _perform_validation(header, header_part, validator, header_validator_index)


def _get_validator_for_header_part(header_part, header_validator_index):
    """
    Selects the appropriate validator based on the type of the header_part.

    :param header_part: The part of the header (name or value) being validated.
    :param header_validator_index: Index to determine which validator to use.
    :returns: A validator function.
    :raises InvalidHeader: If the header_part is neither a string nor bytes.
    """
    if isinstance(header_part, str):
        return _HEADER_VALIDATORS_STR[header_validator_index]
    elif isinstance(header_part, bytes):
        return _HEADER_VALIDATORS_BYTE[header_validator_index]
    else:
        # Raise an exception if header_part type is not recognized
        raise InvalidHeader(
            f"Header part ({header_part!r}) from {header} "
            f"must be of type str or bytes, not {type(header_part)}"
        )


def _perform_validation(
        header,
        header_part,
        validator,
        header_validator_index):
    """
    Validates the header part against the provided validator.

    :param header: The full header being validated.
    :param header_part: The header part (name or value) under validation.
    :param validator: The regex validator against which to validate the header_part.
    :param header_validator_index: Used to determine the kind of header part (name or value).
    :raises InvalidHeader: If the header_part does not match the validator.
    """
    if not validator.match(header_part):
        # Identifying whether the invalid part is the name or value of the
        # header
        header_kind = "name" if header_validator_index == 0 else "value"
        # Raise an exception with a detailed message if header_part fails to
        # validate
        raise InvalidHeader(
            f"Invalid leading whitespace, reserved character(s), or return"
            f" character(s) in header {header_kind}: {header_part!r}"
        )


def urldefragauth(url):
    """
    Remove the fragment and the authentication part from a given URL.

    This function takes a URL, parses it to remove any authentication credentials present, and also removes
    the fragment at the end. The modified URL is then reconstructed and returned without these components.

    :param url: The URL to be processed.
    :rtype: str
    :return: The URL stripped of authentication and fragment.
    """
    # Parse the URL into components
    scheme, netloc, path, params, query, fragment = urlparse(url)

    # Handle cases where the scheme is missing and prepend it if needed
    if not netloc:
        # When netloc is empty, path holds the netloc part due to URL parse
        # behavior
        netloc, path = path, netloc  # Swap to correct the parsing anomaly

    # Remove authentication credentials from netloc, if present
    # Split on '@' and take the last part to exclude credentials
    netloc = netloc.rsplit("@", 1)[-1]

    # Reconstruct the URL without the fragment and authentication parts
    clean_url = urlunparse((scheme, netloc, path, params, query, ""))

    return clean_url


def rewind_body(prepared_request):
    """
    Rewinds the body of a prepared request to a previously saved position.

    This function is typically used to reset the position of the file pointer in the request body,
    allowing the body to be read again from the beginning, such as in the case of a redirect.

    Parameters:
    - prepared_request: The prepped request object whose body needs to be rewound.

    Raises:
    - UnrewindableBodyError: If the body cannot be rewound due to either it not having a 'seek' method,
                             or it's not positioned correctly.
    """
    # Function to attempt rewinding the body of the request
    def attempt_rewind(body_seek, position):
        """
        Attempts to rewind the body of a request using the seek method.

        Parameters:
        - body_seek: The seek method of the request body.
        - position: The position to rewind to, which is initially recorded.

        Raises:
        - UnrewindableBodyError: If rewinding the body fails due to an OSError.
        """
        try:
            body_seek(position)
        except OSError:
            # Raising an error if rewinding fails due to an OSError.
            raise UnrewindableBodyError(
                "An error occurred when rewinding request body for redirect.")

    # Check if the body has a 'seek' method and if its position is correctly
    # defined.
    body_seek = getattr(prepared_request.body, "seek", None)
    if body_seek is not None and isinstance(
            prepared_request._body_position, integer_types):
        # Attempt to rewind if conditions are met.
        attempt_rewind(body_seek, prepared_request._body_position)
    else:
        # Raise an error if conditions for rewinding are not met.
        raise UnrewindableBodyError(
            "Unable to rewind request body for redirect.")
