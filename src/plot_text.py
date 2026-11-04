"""Shared notebook display and text helpers."""

import re


THREAD_COLORS = ("#3366AA", "#DD8844", "#228877", "#8855AA")


_TOKEN = re.compile(r"\{([A-Za-z_]\w*)(?::([^{}]*))?\}")


def render_text(template, **context):
    """Fill known named tokens while preserving LaTeX and unknown braces.

    For example ``{frequency_GHz:.4f}`` uses the actual plotted frequency.
    Mathtext such as ``$E_{T}$`` remains unchanged unless ``T`` is explicitly
    supplied in context. Use an empty string or None to omit a text item.
    """
    if template is None:
        return ""
    if not isinstance(template, str):
        raise TypeError("Plot text must be a string or None.")

    def replace(match):
        name, specification = match.groups()
        if name not in context:
            return match.group(0)
        return format(context[name], specification or "")

    return _TOKEN.sub(replace, template)


def save_figure(figure, path, **kwargs):
    """Include 3D axis labels when saving with a tight bounding box."""
    labels = [axis.label for ax in figure.axes if ax.name == "3d"
              for axis in (ax.xaxis, ax.yaxis, ax.zaxis)]
    if labels:
        kwargs["bbox_extra_artists"] = [*figure.get_default_bbox_extra_artists(), *labels]
    figure.savefig(path, **kwargs)


def display_figure(figure):
    """Embed an actual PNG even when execution uses a headless backend."""
    from io import BytesIO
    from IPython.display import Image, display

    buffer = BytesIO()
    save_figure(figure, buffer, format="png", dpi=110, bbox_inches="tight")
    display(Image(data=buffer.getvalue()))
