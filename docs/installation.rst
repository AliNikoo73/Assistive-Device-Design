Installation
============

GaitSim Assist requires Python 3.9+ and OpenSim 4.3+. There are several ways to install the library.

Using pip
--------

The easiest way to install GaitSim Assist is using pip:

.. code-block:: bash

   pip install gaitsim-assist

From source
-----------

You can also install GaitSim Assist from source:

.. code-block:: bash

   git clone https://github.com/yourusername/gaitsim-assist.git
   cd gaitsim-assist
   pip install -e .

Dependencies
-----------

GaitSim Assist depends on the following packages:

- opensim (4.3+)
- numpy (1.20+)
- matplotlib (3.3+)
- pandas (1.2+)
- scipy (1.6+)
- seaborn (0.11+)
- casadi (3.5.5+)

Installing OpenSim
-----------------

OpenSim is a critical dependency for GaitSim Assist. Here's how to install it:

Windows
~~~~~~~

1. Download the OpenSim installer from the `OpenSim website <https://opensim.stanford.edu/downloads/>`_.
2. Run the installer and follow the instructions.
3. Add OpenSim to your Python environment:

.. code-block:: bash

   pip install opensim

macOS
~~~~~

1. Download the OpenSim installer for macOS from the `OpenSim website <https://opensim.stanford.edu/downloads/>`_.
2. Mount the disk image and drag the OpenSim application to your Applications folder.
3. Add OpenSim to your Python environment:

.. code-block:: bash

   pip install opensim

Linux
~~~~~

On Linux, you'll need to build OpenSim from source:

1. Clone the OpenSim repository:

.. code-block:: bash

   git clone https://github.com/opensim-org/opensim-core.git
   cd opensim-core

2. Follow the build instructions in the repository's README.

3. After building, install the Python bindings:

.. code-block:: bash

   cd build
   pip install -e python

Verifying Installation
---------------------

To verify that GaitSim Assist is installed correctly, run:

.. code-block:: python

   import gaitsim_assist as gsa
   
   # Create a simulator with default 2D walking model
   simulator = gsa.GaitSimulator()
   
   print(f"GaitSim Assist version: {gsa.__version__}")
   print(f"Model created: {simulator.model.getName()}")

This should output the version of GaitSim Assist and the name of the default model. 