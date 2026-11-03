Data taking
===========

**Calibration → preview → acquisition → inspect the saved run.**
Both notebooks default to simulation. Simulation opens no instruments;
the acquisition test mode writes zero-valued VNA traces.

Calibration
-----------

Notebook: :notebook:`sbp_calibration.ipynb`.
Code: :codefile:`src/calibration.py`, :codefile:`src/motor.py`.

1. Enter the measured XYZ endpoints :math:`\mathbf r_A,\mathbf r_B` of each
   subthread and capture their motor counts :math:`n_A,n_B`.
2. Convert coordinate units to metres. Each thread has its own length and scale:

   .. math::

      L=\|\mathbf r_B-\mathbf r_A\|,\qquad
      k_s=\frac{|n_B-n_A|}{L}\quad[\mathrm{steps/m}].

3. Position along the straight thread and XYZ position follow linear
   interpolation, including the sign of the motor direction:

   .. math::

      u=\frac{n-n_A}{n_B-n_A},\qquad
      s=uL,\qquad
      \mathbf r(n)=(1-u)\mathbf r_A+u\mathbf r_B.

   Requested targets use the nearest integer count
   :math:`n(s)=\operatorname{round}[n_A+(n_B-n_A)s/L]`.
   The scan excludes the configured start/end margins.

4. Inspect each steps-per-metre value and save the schema-2 calibration JSON.
   XYZ coordinates in that file are metres. The chosen XYZ frame need not be
   the mirror frame used later in analysis.

The motor home is a manually established reference. Configure physical travel
limits before real motion; after emergency STOP, establish the position again
before another move. The real motor's stored position/home normally lives in
``~/.madmax_bead_pull/motor_state.json``.

Acquisition
-----------

Notebook: :notebook:`sbp_acquisition.ipynb`.
Code: :codefile:`src/acquisition.py`.

1. Set the serial connection, travel limits, VNA address/ports/sweep, scan
   spacing, settling time and calibration path in the local configuration.
2. Preview the target list without connecting to hardware.
3. For a real scan, disable both simulation and VNA test mode. Acquisition reads
   back the instrument's actual frequency axis, moves to each target, and stores
   the complex trace with the reached motor position and interpolated XYZ.
4. Inspect the manifest's completion status and point count. Kernel Interrupt
   aborts the scan; cleanup releases the motor and VNA.

.. list-table:: Run directory
   :header-rows: 1
   :widths: 35 65

   * - File
     - Contents
   * - ``s11_traces.h5``
     - Frequency :math:`\nu` in ``frequencies_Hz`` and complex :math:`\Gamma`
       in ``s_parameter[trace, frequency]``
   * - ``scan_log.csv`` / ``scan_log.jsonl``
     - Reached positions, XYZ, trace index and time
   * - ``scan_manifest.json``
     - Settings, calibration, status and counts
   * - ``vna_log.txt``
     - Instrument communication, for a real VNA

The HDF5 filename is historical: its attributes identify the reflection port,
normally port 2. The measured coefficient is denoted :math:`\Gamma` here.
Existing scan files are refused by default.
