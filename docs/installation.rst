Installation
============

Requirements
------------

For a full local installation (including native extensions), you need:

- Python 3.10+
- A C/C++ toolchain
- A Fortran compiler (for RADIAL-related builds)
- CMake/Ninja

Install from source
-------------------

Editable install:

.. code-block:: bash

   python3 -m pip install -e .

Regular install:

.. code-block:: bash

   python3 -m pip install .

Optional: docs dependencies
---------------------------

.. code-block:: bash

   python3 -m pip install -r docs/requirements.txt

Read the Docs note
------------------

The Read the Docs setup in this repository intentionally uses a docs-only build (it does not run ``pip install .``), so documentation can build without compiling native extensions.
