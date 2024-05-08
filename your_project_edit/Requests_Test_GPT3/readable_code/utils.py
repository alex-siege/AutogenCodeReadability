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

# to_native_string is unused here, but imported here for backwards
# compatibility
from ._internal_utils import (  # noqa: F401
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
    proxy_bypass,
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

# Ensure that ', ' is used to preserve previous delimiter behavior.
DEFAULT_ACCEPT_ENCODING = ", ".join(
    re.split(r",\s*", make_headers(accept_encoding=True)["accept-encoding"])
)


if sys.platform == "win32":
    # provide a proxy_bypass version on Windows without DNS lookups

    def proxy_bypass_registry(host):
        try:
            import winreg
        except ImportError:
            return False

        try:
            internetSettings = winreg.OpenKey(
                winreg.HKEY_CURRENT_USER,
                r"Software\Microsoft\Windows\CurrentVersion\Internet Settings",
            )
            # ProxyEnable could be REG_SZ or REG_DWORD, normalizing it
            proxyEnable = int(
                winreg.QueryValueEx(
                    internetSettings,
                    "ProxyEnable")[0])
            # ProxyOverride is almost always a string
            proxyOverride = winreg.QueryValueEx(
                internetSettings, "ProxyOverride")[0]
        except (OSError, ValueError):
            return False
        if not proxyEnable or not proxyOverride:
            return False

        # make a check value list from the registry entry: replace the
        # '<local>' string by the localhost entry and the corresponding
        # canonical entry.
        proxyOverride = proxyOverride.split(";")
        # now check if we match one of the registry values.
        for test in proxyOverride:
            if test == "<local>":
                if "." not in host:
                    return True
            test = test.replace(".", r"\.")  # mask dots
            test = test.replace("*", r".*")  # change glob sequence
            test = test.replace("?", r".")  # change glob char
            if re.match(test, host, re.I):
                return True
        return False

    def proxy_bypass(host):  # noqa
        """Return True, if the host should be bypassed.

        Checks proxy settings gathered from the environment, if specified,
        or the registry.
        """
        if getproxies_environment():
            return proxy_bypass_environment(host)
        else:
            return proxy_bypass_registry(host)


def dict_to_sequence(input_dict):
    """Converts a dictionary to a sequence and returns the updated dictionary."""

    # If input_dict has the 'items' attribute, convert it to a sequence
    if hasattr(input_dict, "items"):
        input_dict = input_dict.items()

    return input_dict


def super_len(o):
    """
    Calculate the remaining length of an object to be read.

    Args:
    o: any object

    Returns:
    Remaining length to be read from the object.
    """

    # Initialize variables
    total_length = None
    current_position = 0

    # Check if the object has a built-in length function
    if hasattr(o, "__len__"):
        total_length = len(o)

    # Check if the object has a 'len' attribute
    elif hasattr(o, "len"):
        total_length = o.len

    # Check if the object has a 'fileno' attribute
    elif hasattr(o, "fileno"):
        try:
            fileno = o.fileno()
            total_length = os.fstat(fileno).st_size

            # Check if the file was opened in binary mode
            if "b" not in o.mode:
                warnings.warn(
                    ("Requests has determined the content-length for this "
                     "request using the binary size of the file: however, the "
                     "file has been opened in text mode (i.e. without the 'b' "
                     "flag in the mode). This may lead to an incorrect "
                     "content-length. In Requests 3.0, support will be removed "
                     "for files in text mode."), FileModeWarning, )

        except (io.UnsupportedOperation, AttributeError):
            # AttributeError can happen for objects obtained via
            # `Tarfile.extractfile()`
            pass

    # Check if the object has a 'tell' attribute
    if hasattr(o, "tell"):
        try:
            current_position = o.tell()
        except OSError:
            # Handle cases where 'tell' raises OSError
            if total_length is not None:
                current_position = total_length
        else:
            if hasattr(o, "seek") and total_length is None:
                try:
                    # Seek to end of file to get total_length
                    o.seek(0, 2)
                    total_length = o.tell()

                    # Seek back to current position to support partially read
                    # file-like objects
                    o.seek(current_position or 0)

                except OSError:
                    total_length = 0

    # Set total_length to 0 if it's still None
    if total_length is None:
        total_length = 0

    # Calculate remaining length to be read
    return max(0, total_length - current_position)


def get_netrc_auth(url, raise_errors=False):
    """
    Returns the Requests tuple auth for a given URL from the netrc file.

    Args:
    url (str): The URL for which authentication information is needed.
    raise_errors (bool): Flag indicating whether to raise errors if authentication fails.

    Returns:
    tuple: A tuple containing login and password for authentication.
    """

    # Get the path to the netrc file
    netrc_file = os.environ.get("NETRC")
    if netrc_file is not None:
        netrc_locations = (netrc_file,)
    else:
        netrc_locations = (f"~/{f}" for f in NETRC_FILES)

    # Find the netrc file
    netrc_path = None
    for file_path in netrc_locations:
        try:
            location = os.path.expanduser(file_path)
        except KeyError:
            # Handling error in case $HOME is undefined
            return

        if os.path.exists(location):
            netrc_path = location
            break

    # Return if netrc file is not found
    if netrc_path is None:
        return

    # Parse the URL to extract host
    ri = urlparse(url)
    split_str = b":" if isinstance(url, str) else b":".decode("ascii")
    host = ri.netloc.split(split_str)[0]

    try:
        # Get authentication info from netrc file
        _netrc = netrc(netrc_path).authenticators(host)
        if _netrc:
            # Return login and password
            login_index = 0 if _netrc[0] else 1
            return (_netrc[login_index], _netrc[2])
    except (NetrcParseError, OSError):
        # Skip netrc auth if there was an error parsing the file
        if raise_errors:
            raise
    except (ImportError, AttributeError):
        # Handling import errors
        pass


def guess_filename(obj):
    """Tries to guess the filename of the given object."""
    name = getattr(obj, "name", None)
    if name and isinstance(name,
                           basestring) and name[0] != "<" and name[-1] != ">":
        return os.path.basename(name)


def extract_zipped_paths(path):
    """Replace nonexistent paths that look like they refer to a member of a zip
    archive with the location of an extracted copy of the target, or else
    just return the provided path unchanged.
    """
    if os.path.exists(path):
        # This is already a valid path, no need to do anything further
        return path

    # Find the first valid part of the provided path and treat that as a zip archive
    # Assume the rest of the path is the name of a member in the archive
    archive, member = os.path.split(path)

    # Find the folder in which the zip file exists
    while archive and not os.path.exists(archive):
        archive, prefix = os.path.split(archive)
        if not prefix:
            # Avoid infinite loop
            break
        member = "/".join([prefix, member])

    if not zipfile.is_zipfile(archive):
        return path

    zip_file = zipfile.ZipFile(archive)

    if member not in zip_file.namelist():
        return path

    # We have a valid zip archive and a valid member of that archive
    tmp_dir = tempfile.gettempdir()
    extracted_file = os.path.join(tmp_dir, member.split("/")[-1])
    if not os.path.exists(extracted_file):
        # Use read + write to avoid creating nested folders, only the file is
        # wanted, avoids mkdir racing condition
        with atomic_open(extracted_file) as file_handler:
            file_handler.write(zip_file.read(member))
    return extracted_file


@contextlib.contextmanager
def atomic_open(filename):
    """Write a file to the disk in an atomic fashion"""

    # Create a temporary file
    tmp_descriptor, tmp_name = tempfile.mkstemp(dir=os.path.dirname(filename))

    try:
        with os.fdopen(tmp_descriptor, "wb") as tmp_handler:
            # yield the temporary file handler to the caller
            yield tmp_handler
        # Replace the original file with the temporary file
        os.replace(tmp_name, filename)

    except BaseException:
        # If an exception occurs, remove the temporary file
        os.remove(tmp_name)
        raise


def from_key_val_list(data):
    """Converts an object into an OrderedDict if it can be represented as a dictionary.

    :param data: An object to be converted to an OrderedDict
    :return: An OrderedDict representation of the input data

    Examples:
        >>> from_key_val_list([('key', 'val')])
        OrderedDict([('key', 'val')])
        >>> from_key_val_list('string')
        ValueError: cannot encode objects that are not 2-tuples
        >>> from_key_val_list({'key': 'val'})
        OrderedDict([('key', 'val')])
    """

    # Check if input is None
    if data is None:
        return None

    # Check if data is a string, bytes, bool, or int which cannot be converted
    # to OrderedDict
    if isinstance(data, (str, bytes, bool, int)):
        raise ValueError("cannot encode objects that are not 2-tuples")

    # Convert data to OrderedDict
    return OrderedDict(data)


def to_key_val_list(obj):
    """Converts an object to a list of key-value tuples if the object is representable as a dictionary.

    Args:
        obj: An object to convert into a list of key-value tuples.

    Returns:
        list: A list of key-value tuples representing the input object.
    """
    if obj is None:
        return None

    if isinstance(obj, (str, bytes, bool, int)):
        raise ValueError("cannot encode objects that are not 2-tuples")

    if isinstance(obj, Mapping):
        obj = obj.items()

    return list(obj)


# From mitsuhiko/werkzeug (used with permission).
def parse_list_header(value):
    """Parse lists as described by RFC 2068 Section 2.

    In particular, parse comma-separated lists where the elements of
    the list may include quoted-strings. A quoted-string could
    contain a comma. A non-quoted string could have quotes in the
    middle. Quotes are removed automatically after parsing.

    It basically works like :func:`parse_set_header` just that items
    may appear multiple times and case sensitivity is preserved.

    The return value is a standard :class:`list`:

    >>> parse_list_header('token, "quoted value"')
    ['token', 'quoted value']

    To create a header from the :class:`list` again, use the
    :func:`dump_header` function.

    :param value: a string with a list header.
    :return: :class:`list`
    :rtype: list
    """
    def _parse_list_header(value):
        """Helper function to parse the list header."""
        result = []
        for item in value.split(','):
            result.append(item.strip())
        return result

    def unquote_header_value(item):
        """Helper function to remove quotes from quoted string."""
        return item[1:-1]

    result = []
    for item in _parse_list_header(value):
        if item.startswith('"') and item.endswith('"'):
            item = unquote_header_value(item)
        result.append(item)
    return result


# From mitsuhiko/werkzeug (used with permission).
def parse_dict_header(value):
    """Parse lists of key, value pairs as described by RFC 2068 Section 2 and
    convert them into a python dict:

    >>> d = parse_dict_header('foo="is a fish", bar="as well"')
    >>> type(d) is dict
    True
    >>> sorted(d.items())
    [('bar', 'as well'), ('foo', 'is a fish')]

    If there is no value for a key it will be `None`:

    >>> parse_dict_header('key_without_value')
    {'key_without_value': None}

    To create a header from the :class:`dict` again, use the
    :func:`dump_header` function.

    :param value: a string with a dict header.
    :return: :class:`dict`
    :rtype: dict
    """
    result = {}

    # Iterate over parsed list header
    for item in _parse_list_header(value):
        if "=" not in item:
            result[item] = None
        else:
            name, value_str = item.split("=", 1)
            value = _parse_header_value(value_str)
            result[name] = value

    return result


def _parse_header_value(value_str):
    """Parse the value of a header item by unquoting if necessary.

    :param value_str: the string value to be parsed.
    :return: parsed value.
    """
    if value_str[:1] == value_str[-1:] == '"':
        return unquote_header_value(value_str[1:-1])
    return value_str


# From mitsuhiko/werkzeug (used with permission).
def unquote_header_value(value, is_filename=False):
    """Unquotes a header value (Reversal of :func:`quote_header_value`).

    This function does not perform real unquoting. It is designed to match
    what browsers are actually using for quoting.

    :param value: the header value to unquote.
    :param is_filename: a boolean indicating if the value is a filename.
    :return: unquoted header value as a string
    """
    if value and value[0] == value[-1] == '"':
        # Remove surrounding double quotes
        value = value[1:-1]

        if is_filename and value[:2] == "\\\\":
            # Return the value without quotes to preserve UNC paths
            return value
        else:
            # Replacing '\\' with '\' and '\\"' with '"' for normal values
            return value.replace("\\\\", "\\").replace('\\"', '"')

    return value


def dict_from_cookiejar(cookie_jar):
    """
    Returns a key/value dictionary from a CookieJar.

    :param cookie_jar: CookieJar object to extract cookies from.
    :rtype: dict
    """

    cookie_dict = {}

    for cookie in cookie_jar:
        cookie_dict[cookie.name] = cookie.value

    return cookie_dict


def add_dict_to_cookiejar(cj, cookie_dict):
    """
    Returns a CookieJar from a key/value dictionary.

    :param cj: CookieJar to insert cookies into.
    :param cookie_dict: Dict of key/values to insert into CookieJar.
    :rtype: CookieJar
    """

    # Call the helper function cookiejar_from_dict to add key-value pairs to
    # the CookieJar
    return cookiejar_from_dict(cookie_dict, cj)


def get_encodings_from_content(content):
    """Returns encodings from given content string.

    :param content: bytestring to extract encodings from.
    """
    # Display deprecation warning
    warnings.warn(
        (
            "In requests 3.0, get_encodings_from_content will be removed. For "
            "more information, please see the discussion on issue #2266. (This"
            " warning should only appear once.)"
        ),
        DeprecationWarning,
    )

    # Regular expressions to extract encodings
    charset_re = re.compile(r'<meta.*?charset=["\']*(.+?)["\'>]', flags=re.I)
    pragma_re = re.compile(
        r'<meta.*?content=["\']*;?charset=(.+?)["\'>]',
        flags=re.I)
    xml_re = re.compile(r'^<\?xml.*?encoding=["\']*(.+?)["\'>]')

    # Extract and concatenate encodings from different sources
    return (
        charset_re.findall(content)
        + pragma_re.findall(content)
        + xml_re.findall(content)
    )


def _parse_content_type_header(header):
    """Returns content type and parameters from given header

    :param header: string
    :return: tuple containing content type and dictionary of parameters
    """

    # Split header into tokens based on ';'
    tokens = header.split(";")

    # Extract content type and list of parameters
    content_type, params = tokens[0].strip(), tokens[1:]

    # Initialize an empty dictionary to store parameters
    params_dict = {}

    # Define characters to strip from parameter keys and values
    items_to_strip = "\"' "

    # Process each parameter in the list
    for param in params:
        param = param.strip()  # Remove leading/trailing spaces
        if param:
            key, value = param, True
            index_of_equals = param.find("=")

            # Check if parameter has a key-value pair
            if index_of_equals != -1:
                # Extract key and value
                key = param[:index_of_equals].strip(items_to_strip)
                value = param[index_of_equals + 1:].strip(items_to_strip)

            # Store parameter in lowercase key in dictionary
            params_dict[key.lower()] = value

    return content_type, params_dict


def get_encoding_from_headers(headers):
    """Returns the encoding specified in the Content-Type header.

    :param headers: dictionary containing HTTP headers
    :return: encoding string
    """

    # Get the Content-Type header from the headers dictionary
    content_type = headers.get("content-type")

    # If Content-Type header is not present, return None
    if not content_type:
        return None

    # Parse the Content-Type header to get the content type and parameters
    content_type, params = _parse_content_type_header(content_type)

    # Check if 'charset' is specified in the params, return the charset value
    if "charset" in params:
        return params["charset"].strip("'\"")

    # If 'text' is part of content type, return ISO-8859-1
    if "text" in content_type:
        return "ISO-8859-1"

    # If 'application/json' is part of content type, assume UTF-8
    if "application/json" in content_type:
        return "utf-8"


def stream_decode_response_unicode(iterator, response):
    """
    Stream decodes an iterator.

    Args:
    iterator: An iterator containing chunks of encoded data.
    response: An HTTP response object.

    Returns:
    A generator that yields decoded data.
    """

    # If encoding is not specified, return the iterator as is
    if response.encoding is None:
        yield from iterator
        return

    # Get incremental decoder for specified encoding
    decoder = codecs.getincrementaldecoder(response.encoding)(errors="replace")

    # Decode each chunk of data from the iterator
    for chunk in iterator:
        decoded_chunk = decoder.decode(chunk)
        if decoded_chunk:
            yield decoded_chunk

    # Decode any remaining data and yield
    remaining_data = decoder.decode(b"", final=True)
    if remaining_data:
        yield remaining_data


def iter_slices(string, slice_length):
    """
    Iterates over slices of a string.

    Args:
    string (str): The input string.
    slice_length (int): The length of each slice.

    Returns:
    generator: A generator yielding slices of the input string.
    """
    if slice_length is None or slice_length <= 0:
        # Set slice length to length of string if invalid
        slice_length = len(string)
    pos = 0  # Initialize position
    while pos < len(string):
        yield string[pos: pos + slice_length]  # Yield the current slice
        pos += slice_length  # Move to the next slice


def get_unicode_from_response(r):
    """
    Returns the requested content back in unicode.

    :param r: Response object to get unicode content from.

    Looks for charset from content-type and falls back to replace all unicode characters.

    :rtype: str
    """
    warnings.warn(
        (
            "In requests 3.0, get_unicode_from_response will be removed. For "
            "more information, please see the discussion on issue #2266. (This"
            " warning should only appear once.)"
        ),
        DeprecationWarning,
    )

    tried_encodings = []

    encoding = get_encoding_from_headers(r.headers)

    unicode_content = try_encoding(r, encoding, tried_encodings)

    if not unicode_content:
        unicode_content = try_encoding(
            r, None, tried_encodings, errors="replace")

    return unicode_content


def try_encoding(response, encoding, tried_encodings, errors=None):
    """
    Tries encoding the response content using specified encoding.

    :param response: Response object containing content.
    :param encoding: Encoding to use for decoding content.
    :param tried_encodings: List to store tried encodings.
    :param errors: How to handle encoding errors.

    :rtype: str
    """
    try:
        if encoding:
            return str(response.content, encoding)
        else:
            return str(response.content, errors=errors)
    except (UnicodeError, TypeError):
        tried_encodings.append(encoding)
        return None


# The unreserved URI characters (RFC 3986)
UNRESERVED_SET = frozenset(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz" + "0123456789-._~"
)


def unquote_unreserved(uri):
    """Un-escape any percent-escape sequences in a URI that are unreserved
    characters. This leaves all reserved, illegal and non-ASCII bytes encoded.

    :rtype: str
    """

    def process_escape_sequence(part):
        """
        Process each escape sequence and return the unquoted part if it's unreserved,
        otherwise return the original part with '%' symbol.

        :param part: A part of the URI.
        :return: Updated part after unquoting escape sequence.
        """
        h = part[0:2]
        if len(h) == 2 and h.isalnum():
            try:
                c = chr(int(h, 16))
            except ValueError:
                raise InvalidURL(f"Invalid percent-escape sequence: '{h}'")

            if c in UNRESERVED_SET:
                return c + part[2:]
        return f"%{part}"

    parts = uri.split("%")
    for i in range(1, len(parts)):
        parts[i] = process_escape_sequence(parts[i])

    return "".join(parts)


def requote_uri(uri):
    """Re-quote the given URI.

    This function passes the given URI through an unquote/quote cycle to
    ensure that it is fully and consistently quoted.

    :rtype: str
    """
    safe_with_percent = "!#$%&'()*+,/:;=?@[]~"
    safe_without_percent = "!#$&'()*+,/:;=?@[]~"
    try:
        # Unquote only the unreserved characters
        # Then quote only illegal characters (do not quote reserved,
        # unreserved, or '%')
        return quote(unquote_unreserved(uri), safe=safe_with_percent)
    except InvalidURL:
        # We couldn't unquote the given URI, so let's try quoting it, but
        # there may be unquoted '%'s in the URI. We need to make sure they're
        # properly quoted so they do not cause issues elsewhere.
        return quote(uri, safe=safe_without_percent)


def address_in_network(ip, net):
    """Check if an IP belongs to a network subnet.

    :param ip: IP address to check
    :param net: Network subnet in CIDR notation
    :return: True if IP belongs to the network subnet, False otherwise
    """
    # Convert IP address to a long integer
    ip_address = struct.unpack("=L", socket.inet_aton(ip))[0]

    # Get network address and subnet bits
    net_address, bits = net.split("/")

    # Calculate netmask
    netmask = struct.unpack(
        "=L", socket.inet_aton(
            dotted_netmask(
                int(bits))))[0]

    # Calculate the base network address
    network = struct.unpack("=L", socket.inet_aton(net_address))[0] & netmask

    # Check if IP address is in the network subnet
    return (ip_address & netmask) == (network & netmask)


def dotted_netmask(mask):
    """Converts mask from /xx format to xxx.xxx.xxx.xxx

    Example: if mask is 24 function returns 255.255.255.0

    :param mask: integer representing the subnet mask length
    :return: string representing the subnet mask in dotted decimal format
    """
    # Calculate the number of bits in the subnet mask
    bits = 0xFFFFFFFF ^ (1 << (32 - mask)) - 1
    # Convert the bits to dotted decimal format
    return socket.inet_ntoa(struct.pack(">I", bits))


def is_ipv4_address(string_ip):
    """
    Check if a given string represents a valid IPv4 address.

    :param string_ip: A string representing an IPv4 address
    :rtype: bool
    """
    try:
        # Try to convert the string to an IPv4 address
        socket.inet_aton(string_ip)
    except OSError:
        # If an exception occurs, the string is not a valid IPv4 address
        return False
    return True


def is_valid_cidr(string_network):
    """
    Check if the CIDR format in the no_proxy variable is valid.

    :param string_network: A string representing a network address with CIDR notation
    :type string_network: str
    :rtype: bool
    """
    # Check if the CIDR notation is in the correct format
    if string_network.count("/") == 1:
        try:
            # Extract the mask from the CIDR string
            mask = int(string_network.split("/")[1])
        except ValueError:
            return False

        # Check if the mask value falls within the valid range
        if mask < 1 or mask > 32:
            return False

        try:
            # Check if the IP address part is a valid IP address
            socket.inet_aton(string_network.split("/")[0])
        except OSError:
            return False
    else:
        return False

    return True


@contextlib.contextmanager
def set_environ(env_name, value):
    """Set the environment variable 'env_name' to 'value' temporarily.

    This function sets the environment variable 'env_name' to 'value'
    temporarily and restores it to its previous value after the execution
    completes. If 'value' is None, no action is taken.

    Args:
    env_name (str): The name of the environment variable to set.
    value (str): The value to set for the environment variable.
    """
    value_changed = value is not None

    # Save the previous value of the environment variable
    if value_changed:
        old_value = os.environ.get(env_name)
        os.environ[env_name] = value

    try:
        yield

    finally:
        # Restore the previous value of the environment variable
        if value_changed:
            if old_value is None:
                del os.environ[env_name]
            else:
                os.environ[env_name] = old_value


def should_bypass_proxies(url, no_proxy):
    """
    Returns whether proxies should be bypassed for the given URL.

    :rtype: bool
    """

    def get_proxy_value(key):
        """
        Get the proxy value for a given key.
        """
        return os.environ.get(key) or os.environ.get(key.upper())

    # Check if no_proxy is provided, if not, retrieve it from environment
    # variables
    no_proxy_original = no_proxy
    if no_proxy is None:
        no_proxy = get_proxy_value("no_proxy")

    parsed_url = urlparse(url)

    if parsed_url.hostname is None:
        return True  # Some URLs may not have hostnames (e.g., file:// urls)

    if no_proxy:
        no_proxy_list = (
            host for host in no_proxy.replace(
                " ", "").split(",") if host)

        if is_ipv4_address(parsed_url.hostname):
            for proxy_ip in no_proxy_list:
                if is_valid_cidr(proxy_ip):
                    if address_in_network(parsed_url.hostname, proxy_ip):
                        return True
                elif parsed_url.hostname == proxy_ip:
                    return True  # Check for exact match with IP in plain notation
        else:
            host_with_port = parsed_url.hostname
            if parsed_url.port:
                host_with_port += f":{parsed_url.port}"

            for host in no_proxy_list:
                if parsed_url.hostname.endswith(
                        host) or host_with_port.endswith(host):
                    return True  # Match found in no_proxy list

    with set_environ("no_proxy", no_proxy_original):
        try:
            bypass = proxy_bypass(parsed_url.hostname)
        except (TypeError, socket.gaierror):
            bypass = False

    return bypass or False


def get_environ_proxies(url, no_proxy=None):
    """
    Return a dict of environment proxies.

    :rtype: dict
    """
    if should_bypass_proxies(url, no_proxy=no_proxy):
        return {}

    return getproxies()


def select_proxy(url, proxies):
    """Select a proxy for the url, if applicable.

    :param url: The url being for the request
    :param proxies: A dictionary of schemes or schemes and hosts to proxy URLs
    """
    # Initialize proxies if None
    proxies = proxies or {}

    # Parse the url
    url_parts = urlparse(url)

    # If hostname is None, return accordingly
    if url_parts.hostname is None:
        return proxies.get(url_parts.scheme, proxies.get("all"))

    # Define possible keys for proxy lookup
    proxy_keys = [
        url_parts.scheme + "://" + url_parts.hostname,
        url_parts.scheme,
        "all://" + url_parts.hostname,
        "all",
    ]

    # Find a matching proxy
    proxy = None
    for key in proxy_keys:
        if key in proxies:
            proxy = proxies[key]
            break

    return proxy


def resolve_proxies(request, proxies, trust_env=True):
    """Resolve proxies based on request information and configuration settings,
    taking into account the NO_PROXY configuration.

    :param request: Request or PreparedRequest object
    :param proxies: A dictionary containing proxy information for different schemes or hosts
    :param trust_env: A boolean indicating whether to trust environment configurations

    :return: A dictionary of resolved target proxies
    """
    proxies = proxies if proxies is not None else {
    }  # If proxies is None, initialize it as an empty dictionary
    url = request.url
    scheme = urlparse(url).scheme
    no_proxy = proxies.get("no_proxy")
    new_proxies = proxies.copy()

    # Check if environment proxies should be trusted and if the request should
    # bypass proxies based on no_proxy setting
    if trust_env and not should_bypass_proxies(url, no_proxy=no_proxy):
        # Retrieve environment proxies based on the request URL
        environ_proxies = get_environ_proxies(url, no_proxy=no_proxy)

        # Get the proxy based on the scheme or default to 'all'
        proxy = environ_proxies.get(scheme, environ_proxies.get("all"))

        # Set the proxy for the scheme if found
        if proxy:
            new_proxies.setdefault(scheme, proxy)

    return new_proxies


def default_user_agent(name="python-requests"):
    """
    Return a string representing the default user agent.

    :param name: str, name of the user agent
    :rtype: str
    """
    user_agent = f"{name}/{__version__}"

    return user_agent


def default_headers():
    """
    This function returns a dictionary with default headers for a request.

    :rtype: requests.structures.CaseInsensitiveDict
    """
    headers = {
        "User-Agent": default_user_agent(),
        "Accept-Encoding": DEFAULT_ACCEPT_ENCODING,
        "Accept": "*/*",
        "Connection": "keep-alive",
    }
    return CaseInsensitiveDict(headers)


def parse_header_links(value):
    """Return a list of parsed link headers proxies.

    i.e. Link: <http:/.../front.jpeg>; rel=front; type="image/jpeg",<http://.../back.jpeg>; rel=back;type="image/jpeg"

    :rtype: list
    """

    links = []

    replace_chars = " '\""

    # Strip unnecessary characters from the beginning and end of the value
    value = value.strip(replace_chars)

    # If the value is empty, return an empty list
    if not value:
        return links

    # Split the value by commas followed by '<'
    for val in re.split(", *<", value):
        try:
            url, params = val.split(";", 1)
        except ValueError:
            url, params = val, ""

        link = {"url": url.strip("<> '\"")}

        # Split params by ';'
        for param in params.split(";"):
            try:
                key, val = param.split("=")
            except ValueError:
                break

            # Add key-value pairs to the link dictionary
            link[key.strip(replace_chars)] = val.strip(replace_chars)

        # Append the link dictionary to the links list
        links.append(link)

    return links


# Null bytes; no need to recreate these on each call to guess_json_utf
_null = "\x00".encode("ascii")  # encoding to ASCII for Python 3
_null2 = _null * 2
_null3 = _null * 3


def guess_json_utf(data):
    """
    :rtype: str
    """
    # JSON always starts with two ASCII characters, so detection is as
    # easy as counting the nulls and from their location and count
    # determine the encoding. Also detect a BOM, if present.
    sample = data[:4]
    if sample in (codecs.BOM_UTF32_LE, codecs.BOM_UTF32_BE):
        return "utf-32"  # BOM included
    if sample[:3] == codecs.BOM_UTF8:
        return "utf-8-sig"  # BOM included, MS style (discouraged)
    if sample[:2] in (codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE):
        return "utf-16"  # BOM included
    nullcount = sample.count(_null)
    if nullcount == 0:
        return "utf-8"
    if nullcount == 2:
        if sample[::2] == _null2:  # 1st and 3rd are null
            return "utf-16-be"
        if sample[1::2] == _null2:  # 2nd and 4th are null
            return "utf-16-le"
        # Did not detect 2 valid UTF-16 ascii-range characters
    if nullcount == 3:
        if sample[:3] == _null3:
            return "utf-32-be"
        if sample[1:] == _null3:
            return "utf-32-le"
        # Did not detect a valid UTF-32 ascii-range character
    return None


def prepend_scheme_if_needed(url, new_scheme):
    """
    Given a URL that may or may not have a scheme, prepend the given scheme.
    Does not replace a present scheme with the one provided as an argument.

    :rtype: str
    """
    parsed_url = parse_url(url)

    scheme, auth, host, port, path, query, fragment = parsed_url

    # A defect in urlparse determines that there isn't a netloc present in some
    # urls. We previously assumed parsing was overly cautious, and swapped the
    # netloc and path. Due to a lack of tests on the original defect, this is
    # maintained with parse_url for backwards compatibility.
    netloc = parsed_url.netloc
    if not netloc:
        netloc, path = path, netloc

    if auth:
        # parse_url doesn't provide the netloc with auth
        # so we'll add it ourselves.
        netloc = f"{auth}@{netloc}"

    if scheme is None:
        scheme = new_scheme

    if path is None:
        path = ""

    return urlunparse((scheme, netloc, path, "", query, fragment))


def get_auth_from_url(url):
    """
    Extracts authentication components (username, password) from a given URL.

    :param url: str, URL with authentication components
    :return: tuple, (username, password)
    """
    parsed_url = urlparse(url)

    try:
        username = unquote(parsed_url.username)
        password = unquote(parsed_url.password)
    except (AttributeError, TypeError):
        username = ""
        password = ""

    return (username, password)


def check_header_validity(header):
    """Verifies that header parts do not contain leading whitespace,
    reserved characters, or return characters.

    :param header: tuple, in the format (name, value).
    """
    name, value = header
    _validate_header_part(header, name, 0)
    _validate_header_part(header, value, 1)


def _validate_header_part(header, header_part, header_validator_index):
    """
    Validate the header part based on the header type and index.

    Args:
    header: The header being validated.
    header_part: The specific part of the header to validate.
    header_validator_index: The index used to determine the validator to use.

    Raises:
    InvalidHeader: if the header part is not of type str or bytes, or if it contains invalid characters.
    """
    if isinstance(header_part, str):
        validator = _HEADER_VALIDATORS_STR[header_validator_index]
    elif isinstance(header_part, bytes):
        validator = _HEADER_VALIDATORS_BYTE[header_validator_index]
    else:
        raise InvalidHeader(
            f"Header part ({header_part!r}) from {header} "
            f"must be of type str or bytes, not {type(header_part)}"
        )

    if not validator.match(header_part):
        header_kind = "name" if header_validator_index == 0 else "value"
        raise InvalidHeader(
            f"Invalid leading whitespace, reserved character(s), or return"
            f"character(s) in header {header_kind}: {header_part!r}"
        )


def urldefragauth(url):
    """
    Given a URL, remove the fragment and the authentication part.

    :rtype: str
    """
    # Parse the URL into its components
    scheme, netloc, path, params, query, fragment = urlparse(url)

    # Swap netloc and path if netloc is empty
    if not netloc:
        netloc, path = path, netloc

    # Remove authentication part from netloc
    netloc = netloc.rsplit("@", 1)[-1]

    # Reconstruct the URL without the fragment and authentication
    return urlunparse((scheme, netloc, path, params, query, ""))


def rewind_body(prepared_request):
    """Move file pointer back to its recorded starting position
    so it can be read again on redirect.
    """
    body_seek_function = getattr(prepared_request.body, "seek", None)

    if body_seek_function is not None and isinstance(
            prepared_request._body_position, integer_types):
        try:
            body_seek_function(prepared_request._body_position)
        except OSError:
            raise UnrewindableBodyError(
                "An error occurred when rewinding request body for redirect.")
    else:
        raise UnrewindableBodyError(
            "Unable to rewind request body for redirect.")
