"""Sphinx configuration; building the docs never imports instrument code."""

project = "MADMAX SBP"
author = "MADMAX SBP contributors"
copyright = "2026, MADMAX SBP contributors"
extensions = ["sphinx.ext.mathjax", "sphinx.ext.extlinks", "sphinx.ext.githubpages"]
exclude_patterns = ["_build"]
root_doc = "index"
html_theme = "furo"
html_title = "MADMAX SBP"
html_static_path = ["_static"]
html_css_files = ["pipeline.css"]
html_theme_options = {
    "light_css_variables": {
        "color-brand-primary": "#156b85",
        "color-brand-content": "#156b85",
    },
    "dark_css_variables": {
        "color-brand-primary": "#73c8dc",
        "color-brand-content": "#73c8dc",
    },
}
html_show_sourcelink = False
html_baseurl = "https://nabil-2.github.io/madmax-sbp/"
extlinks = {
    "codefile": ("https://github.com/nabil-2/madmax-sbp/blob/main/%s", "%s"),
    "notebook": ("https://github.com/nabil-2/madmax-sbp/blob/main/%s", "%s"),
}
mathjax3_config = {"tex": {"tags": "none"}}
