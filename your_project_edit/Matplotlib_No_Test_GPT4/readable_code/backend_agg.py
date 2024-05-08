"""
An `Anti-Grain Geometry`_ (AGG) backend.

Features that are implemented:

* capstyles and join styles
* dashes
* linewidth
* lines, rectangles, ellipses
* clipping to a rectangle
* output to RGBA and Pillow-supported image formats
* alpha blending
* DPI scaling properly - everything scales properly (dashes, linewidths, etc)
* draw polygon
* freetype2 w/ ft2font

Still TODO:

* integrate screen dpi w/ ppi and text

.. _Anti-Grain Geometry: http://agg.sourceforge.net/antigrain.com
"""

from contextlib import nullcontext
from math import radians, cos, sin

import numpy as np

import matplotlib as mpl
from matplotlib import _api, cbook
from matplotlib.backend_bases import (
    _Backend, FigureCanvasBase, FigureManagerBase, RendererBase)
from matplotlib.font_manager import fontManager as _fontManager, get_font
from matplotlib.ft2font import (LOAD_FORCE_AUTOHINT, LOAD_NO_HINTING,
                                LOAD_DEFAULT, LOAD_NO_AUTOHINT)
from matplotlib.mathtext import MathTextParser
from matplotlib.path import Path
from matplotlib.transforms import Bbox, BboxBase
from matplotlib.backends._backend_agg import RendererAgg as _RendererAgg


def get_hinting_flag():
    """
    Retrieves the hinting flag based on the matplotlib configuration.

    It maps the user's hinting preference, as specified in matplotlib's
    configuration, to the appropriate load flag. This is used to determine
    how text rendering should handle hinting.
    """
    # Mapping of user preference to text hinting flags
    hinting_preference_to_flag = {
        'default': LOAD_DEFAULT,  # Use the default hinting mode
        'no_autohint': LOAD_NO_AUTOHINT,  # Disable automatic hinting
        'force_autohint': LOAD_FORCE_AUTOHINT,  # Force automatic hinting
        'no_hinting': LOAD_NO_HINTING,  # Disable hinting
        True: LOAD_FORCE_AUTOHINT,  # Interpret True as forcing automatic hinting
        False: LOAD_NO_HINTING,  # Interpret False as disabling hinting
        'either': LOAD_DEFAULT,  # 'either' is treated as 'default'
        'native': LOAD_NO_AUTOHINT,  # Use the hinting of the font's native environment
        'auto': LOAD_FORCE_AUTOHINT,  # Alias for forcing automatic hinting
        'none': LOAD_NO_HINTING,  # Alias for disabling hinting
    }
    # Return the flag corresponding to the user's configured hinting preference
    return hinting_preference_to_flag[mpl.rcParams['text.hinting']]


class RendererAgg(RendererBase):
    """
    The renderer handles all the drawing primitives using a graphics
    context instance that controls the colors/styles.
    """

    ...

    def draw_path(self, gc, path, transform, rgbFace=None):
        """
        Draw a path with possible chunking if the path has too many points.

        This method handles drawing of paths, applying chunking strategies
        if the number of points in the path exceeds a certain threshold.
        It simplifies the drawing of complex paths by breaking them into
        smaller chunks if necessary, to avoid issues like exceeding cell block
        limits in the Agg renderer.
        """
        should_chunk_path = self._should_chunk_path(path, rgbFace, gc)

        if should_chunk_path:
            self._draw_path_chunked(gc, path, transform, rgbFace)
        else:
            self._try_draw_path_directly(gc, path, transform, rgbFace)

    def _should_chunk_path(self, path, rgbFace, gc):
        """
        Determine whether the path needs to be chunked.

        Returns True if the path exceeds the maximum number of points and
        the other conditions for not simplifying the path are not met.
        """
        nmax = mpl.rcParams['agg.path.chunksize']
        return (
            path.vertices.shape[0] > nmax > 100 and
            path.should_simplify and rgbFace is None and
            gc.get_hatch() is None
        )

    def _draw_path_chunked(self, gc, path, transform, rgbFace):
        """
        Draw the path in chunks, splitting it up if it's too long.
        """
        nmax = mpl.rcParams['agg.path.chunksize']
        npts = path.vertices.shape[0]
        nch = np.ceil(npts / nmax)
        chsize = int(np.ceil(npts / nch))
        for i0, i1 in self._chunk_indices(npts, chsize):
            self._draw_path_chunk(gc, path, transform, rgbFace, i0, i1)

    def _chunk_indices(self, npts, chsize):
        """
        Generate start and end indices for each chunk.
        """
        i0 = np.arange(0, npts, chsize)
        i1 = np.zeros_like(i0)
        i1[:-1] = i0[1:] - 1
        i1[-1] = npts
        return zip(i0, i1)

    def _draw_path_chunk(self, gc, path, transform, rgbFace, i0, i1):
        """
        Draw a single chunk of a path.
        """
        v = path.vertices[i0: i1, :]
        c = path.codes if path.codes is not None else None
        if c is not None:
            c = c[i0:i1]
            c[0] = Path.MOVETO
        p = Path(v, c)
        p.simplify_threshold = path.simplify_threshold
        self._try_draw_path_directly(gc, p, transform, rgbFace)

    def _try_draw_path_directly(self, gc, path, transform, rgbFace):
        """
        Try drawing a path directly, handling OverflowError by presenting a
        detailed error message.
        """
        try:
            self._renderer.draw_path(gc, path, transform, rgbFace)
        except OverflowError:
            self._handle_overflow_error(rgbFace, gc, path)

    def _handle_overflow_error(self, rgbFace, gc, path):
        """
        Handle OverflowError by generating a detailed error message and raising
        the error.
        """
        cant_chunk_msg = self._generate_cant_chunk_message(rgbFace, gc, path)
        if cant_chunk_msg:
            msg = (
                f"Exceeded cell block limit in Agg, reasons:\n\n"
                f"{cant_chunk_msg}\n"
                "Cannot automatically split this path to draw.\n"
                "Please manually simplify your path."
            )
        else:
            msg = self._generate_simplification_advice()
        raise OverflowError(msg) from None

    def _generate_cant_chunk_message(self, rgbFace, gc, path):
        """
        Generates a message detailing why a path could not be chunked.
        """
        messages = []
        if rgbFace is not None:
            messages.append("- cannot split filled path")
        if gc.get_hatch() is not None:
            messages.append("- cannot split hatched path")
        if not path.should_simplify:
            messages.append("- path.should_simplify is False")
        return '\n'.join(messages)

    def _generate_simplification_advice(self):
        """
        Generates advice for increasing the path simplification threshold.
        """
        nmax = mpl.rcParams['agg.path.chunksize']
        inc_threshold = (
            "Increase the path simplification threshold"
            "(rcParams['path.simplify_threshold'] = "
            f"{mpl.rcParams['path.simplify_threshold']} "
            "by default and path.simplify_threshold "
            f"= {path.simplify_threshold} on the input)."
        )
        if nmax > 100:
            return (
                "Exceeded cell block limit in Agg. Reduce the value of "
                f"rcParams['agg.path.chunksize'] (currently {nmax}). {inc_threshold}")
        else:
            return (
                "Exceeded cell block limit in Agg. Set rcParams['agg.path.chunksize'] "
                f"(currently {nmax}) to be greater than 100. {inc_threshold}")

    ...


class FigureCanvasAgg(FigureCanvasBase):
    """Inherits from FigureCanvasBase to provide a canvas representation in the Agg backend."""

    # Tracks the last renderer configuration to optimize re-use.
    _lastKey = None

    def copy_from_bbox(self, bbox):
        """Returns a region from the renderer that covers the specified bounding box."""
        renderer = self.get_renderer()
        return renderer.copy_from_bbox(bbox)

    def restore_region(self, region, bbox=None, xy=None):
        """Restores the specified region in the renderer."""
        renderer = self.get_renderer()
        return renderer.restore_region(region, bbox, xy)

    def draw(self):
        """
        Renders the figure using the Agg renderer.

        This method acquires a lock on the font cache before drawing to ensure thread safety.
        """
        self.renderer = self.get_renderer()
        self.renderer.clear()  # Clears the current rendering surface.
        with self._acquire_draw_lock():
            self.figure.draw(self.renderer)
            # Calls the superclass draw method to potentially update the UI.
            super().draw()

    def _acquire_draw_lock(self):
        """
        Acquires a lock for drawing operations.

        Returns a context manager that is responsible for acquiring a lock on the
        shared resources during drawing operations.
        """
        return (self.toolbar._wait_cursor_for_draw_cm()
                if self.toolbar else nullcontext())

    def get_renderer(self):
        """
        Retrieves or creates a renderer based on the figure's current size and DPI.

        Returns the renderer instance used for drawing operations.
        """
        current_key = self.figure.bbox.size + (self.figure.dpi,)
        if self._lastKey != current_key:
            self.renderer = RendererAgg(*current_key)
            self._lastKey = current_key
        return self.renderer

    @_api.deprecated("3.8", alternative="buffer_rgba")
    def tostring_rgb(self):
        """Returns the rendering buffer as RGB bytes."""
        return self.renderer.tostring_rgb()

    def tostring_argb(self):
        """Returns the rendering buffer as ARGB bytes."""
        return self.renderer.tostring_argb()

    def buffer_rgba(self):
        """Returns a memoryview of the renderer's RGBA buffer."""
        return self.renderer.buffer_rgba()

    def print_raw(self, filename_or_obj, *, metadata=None):
        """Writes the figure to a file in raw RGBA format."""
        if metadata is not None:
            raise ValueError("metadata not supported for raw/rgba")
        self.draw()
        renderer = self.get_renderer()
        with cbook.open_file_cm(filename_or_obj, "wb") as fh:
            fh.write(renderer.buffer_rgba())

    print_rgba = print_raw  # Alias for print_raw

    def _print_pil(self, filename_or_obj, fmt, pil_kwargs, metadata=None):
        """
        Helper method to save the canvas content using PIL based formats.

        Uses the `.imagesave` function to write the figure to a file in a specified
        format, forwarding `pil_kwargs` and `metadata` to the saving function.
        """
        self.draw()
        mpl.image.imsave(
            filename_or_obj, self.buffer_rgba(), format=fmt, origin="upper",
            dpi=self.figure.dpi, metadata=metadata, pil_kwargs=pil_kwargs)

    def print_png(self, filename_or_obj, *, metadata=None, pil_kwargs=None):
        """
        Writes the figure to a PNG file.

        Parameters are documented in the class docstring. For common image formats, use the
        corresponding print_* method.
        """
        self._print_pil(filename_or_obj, "png", pil_kwargs, metadata)

    def print_to_buffer(self):
        """Returns a tuple of the rendered buffer bytes and the renderer dimensions."""
        self.draw()
        renderer = self.get_renderer()
        return (
            bytes(
                renderer.buffer_rgba()), (int(
                    renderer.width), int(
                    renderer.height)))

    def print_jpg(self, filename_or_obj, *, metadata=None, pil_kwargs=None):
        """Writes the figure to a JPEG file with potential additional PIL arguments."""
        self._prepare_and_print_image(
            filename_or_obj, "jpeg", pil_kwargs, metadata)

    def print_tif(self, filename_or_obj, *, metadata=None, pil_kwargs=None):
        """Writes the figure to a TIFF file."""
        self._prepare_and_print_image(
            filename_or_obj, "tiff", pil_kwargs, metadata)

    def print_webp(self, filename_or_obj, *, metadata=None, pil_kwargs=None):
        """Writes the figure to a WebP file."""
        self._prepare_and_print_image(
            filename_or_obj, "webp", pil_kwargs, metadata)

    def _prepare_and_print_image(
            self,
            filename_or_obj,
            format_type,
            pil_kwargs,
            metadata):
        """
        Prepares the environment and prints an image in a specific format.

        Sets the assumed background color for semi-transparent figures and forwards
        the call to `_print_pil`.
        """
        with mpl.rc_context({"savefig.facecolor": "white"}):
            self._print_pil(filename_or_obj, format_type, pil_kwargs, metadata)

    # Alias methods for different file formats
    print_jpeg = print_jpg
    print_tiff = print_tif


@_Backend.export
class _BackendAgg(_Backend):
    backend_version = 'v2.2'
    FigureCanvas = FigureCanvasAgg
    FigureManager = FigureManagerBase
