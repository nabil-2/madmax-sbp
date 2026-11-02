"""Small text-template helper shared by the notebook plotting functions."""

import re


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
