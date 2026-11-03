MADMAX SBP
==========

Code and mathematical steps for **spider bead pull**: verify reduced sampling,
take measurements, then reconstruct fields and the power boost factor
:math:`\beta^2`. Each pipeline links to its implementation and notebook.

.. raw:: html

   <div class="workflow-grid">
     <a href="verification.html"><strong>1 · Verification</strong>Full grid → limited lines → field-integral comparison</a>
     <a href="data_taking.html"><strong>2 · Data taking</strong>Endpoint calibration → motor/VNA scan → recorded traces</a>
     <a href="analysis/index.html"><strong>3 · Analysis</strong>Measured traces → 1D or 3D fit → boost and diagnostics</a>
   </div>

.. toctree::
   :hidden:
   :maxdepth: 2

   Verification <verification>
   Data taking <data_taking>
   Analysis <analysis/index>

Start here
----------

* **Few lines from a full grid:** :doc:`verification`.
* **A new spider measurement:** :doc:`data_taking`.
* **An existing spider scan:** :doc:`analysis/one_dimensional` or
  :doc:`analysis/three_dimensional`.

The notebooks contain plot controls and rendered results. Calculations live in
``src/``; the large measurement files and propagation banks are local inputs.
The documentation build does not load measurements or connect to instruments.

Build locally
-------------

From the repository root:

.. code-block:: bash

   python -m pip install -r docs/requirements.txt
   python -m sphinx -n -W --keep-going -b html docs docs/_build/html
   python -m http.server --directory docs/_build/html 8000

Open ``http://localhost:8000``. GitHub Actions builds the same sources and
publishes them to `GitHub Pages <https://nabil-2.github.io/madmax-sbp/>`_.
