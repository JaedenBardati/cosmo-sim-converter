# cosmo-sim-converter

Converts between various cosmological simulation data formats, including from [pynbody](https://pynbody.github.io/pynbody/)-loadable simulations to multi-file binary [Gadget](https://wwwmpa.mpa-garching.mpg.de/gadget/), to [Gadget-HDF5](https://wwwmpa.mpa-garching.mpg.de/gadget4/06_snapshotformat/) (supported by the [Powderday](https://powderday.readthedocs.io/en/latest/) radiative transfer code), or to a specific ASCII format supported by the [SKIRT](https://skirt.ugent.be/root/_home.html) radiative transfer code.

The currently supported conversions are:

- **Pynbody to Gadget**: Convert to multi-file binary Gadget format.
- **Pynbody to Gadget HDF5**: Convert to single-file Gadget HDF5 format (for use with Powderday, requires `h5py`).
- **Pynbody to ASCII**: Convert to a specific ASCII format (for use with SKIRT).

It can also load SKIRT ASCII input/output into `pandas` dataframes (via the `load_skirt` function, originally from my SKIRT output [repo](https://github.com/JaedenBardati/skirt-datacube)). 

The main file is `cosmo_sim_converter.py`. See the Jupyter notebooks for example usages.
