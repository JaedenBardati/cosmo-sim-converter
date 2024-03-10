"""
Converts simulation data files between formats. Jaeden Bardati, 2021-2024.

Requires numpy along other packages depending on the loader or writer used (see below). 
Optionally uses my_timing.py (https://gist.github.com/JaedenBardati/e953033508000f637a4121982429a56e) 
and/or file_arguments.py (https://gist.github.com/JaedenBardati/81c4543b84a49584ea09bf529fbdf29c).

Currently supports conversions:
 - Pynbody --> Gadget (multi-file binary)                            : Requires pynbody
 - Pynbody --> Gadget HDF5 (form recognizable by Powderday RT code)  : Requires pynbody, h5py
 - Pynbody --> ASCII (form recognizable by SKIRT RT code)            : Requires pynbody

Also supports loading of SKIRT ASCII output (i.e. .dat files) into pandas dataframes (via ASCII_SKIRT.read_into_dataframe).


------ How To Run ------

Running this file in the command line will attempt to load using pynbody and 
then convert to a desired format in the form:

  $ python cosmo_sim_converter.py (input filename) (output filename) (file conversion type) [input halo number]

This requires the file_arguments.py file. The [] indicates optional support for halo finder data. See the Jupyter notebooks 
in the associated GitHub repo (https://github.com/JaedenBardati/pynbody2gadget) for example usages in python.

"""


import os
from abc import ABCMeta, abstractmethod

import numpy as np


try:
    from my_timing.my_timing import log_timing
except ModuleNotFoundError:
    log_timing = lambda x, log_it=True: print(x) if log_it else None



class FileType(object):
    """ An abstract class for a general file type and underlying loader/writer. """
    __metaclass__ = ABCMeta

    def __init__(self, *args, read=False, **kwargs):
        if read: 
            self.read(*args, **kwargs)

    @abstractmethod
    def read(self, filename, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def write(self, filename, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def convert_to(self, to_file_type, *args, **kwargs):
        raise NotImplementedError
        


##### FILE TYPE CLASSES #####

class GadgetBinary(FileType):
    """
    Class for Gadget binary conversion (allows for multiple files). 
    Requires pynbody Python module. Adapted from pynbody code (https://github.com/pynbody/pynbody).
    """
    
    DEFAULT_GADGET_UNIT_BASE = {
        "distance": "kpc h**-1",
        "velocity": "km s**-1",
        "mass": "10.0e+10 Msol h**-1"
    } # note changing this will do nothing

    def write(self, filename, num_files=1, _debug_print=False, unit_base=None):
        """Write an entire Gadget file (with multi-file support). The new parameter num_files determines the number of output files.
        This code was adapted from pynbody code to support multiple-file writing."""
        sim = self.sim
        
        if type(num_files) != int: 
            raise Exception("The parameter num_files must be an integer.")

        import pynbody
        from pynbody.snapshot.gadget import GadgetSnap, N_TYPE, gadget_type, GadgetHeader, _output_order_gadget, WriteBlock, _translate_array_name, GadgetWriteFile, _to_raw

        # function to combine two pynbody snapshots
        def combine_snaps(*snaps, loadable_sims=True, _debug_print=False):
            """
            Combines multiple pynbody snapshots. 
            Adopts the properties of the first snap entered and, for each family, adopts 
            the keys of the first snapshot that contains particles in that family.
            If loadable_sims is True, it uses loadable_keys() to copy, otherwise uses keys().
            Note that it does not lazy load and thus can take a lot of memory.
            """
            assert len(snaps) >= 2
            families = list(set([fam for f in snaps for fam in f.families()]))
            assert len(families) >= 1
            loadable_keys = {fam:[f[fam].loadable_keys() if loadable_sims else f[fam].keys() for f in snaps if fam in f.families()][0] for fam in families}
            lengths = {}
            for fam in families:
                lengths[fam.name] = sum([len(f[fam]) for f in snaps if fam in f.families()])
            combined = pynbody.new(**lengths)
            combined.properties = snaps[0].properties
            for fam in families:
                pindex = 0
                for f in snaps:
                    if fam in f.families():
                        s = f[fam]
                        if _debug_print: print(s)
                        for arname in loadable_keys[fam]:
                            if _debug_print: print(arname, pindex, pindex+len(s))
                            try:
                                combined[fam][pindex:pindex+len(s)][arname] = s[arname]
                            except KeyError:
                                pass
                        pindex += len(s)
            return combined

        # update the GadgetHeader class to allow for multi-file loads
        previous_GadgetHeader_init = GadgetHeader.__init__

        def GadgetHeader_init(self, npart, mass, time, redshift, BoxSize, Omega0, OmegaLambda, HubbleParam, num_files=1, npartTotal=None):
            previous_GadgetHeader_init(self, npart, mass, time, redshift, BoxSize, Omega0, OmegaLambda, HubbleParam, num_files=num_files if npartTotal is not None else 1) # adding backwards support
            if npartTotal is not None:
                if self.NallHW.any() and num_files > 1:
                    raise NotImplementedError("There cannot be more than 2^32 particles in the multi-file mode, otherwise NallHW must be updated accordingly in the code.")
                self.num_files = num_files
                self.npartTotal = npartTotal

        GadgetHeader.__init__ = GadgetHeader_init

        with sim.lazy_derive_off:
            # If caller is not a GadgetSnap, construct the GadgetFiles, so that format conversion works.
            all_keys = set(sim.loadable_keys()).union(list(sim.keys())).union(sim.family_keys())
            all_keys = [k for k in all_keys if not k in ["x", "y", "z", "vx", "vy", "vz"]]
            if _debug_print: 
                print("all_keys =", all_keys)

            # If we are writing to a new type,
            if sim.__class__ is not GadgetSnap:
                # Then we need a filename
                if filename == None:
                    raise Exception("Please specify a filename to write a new file.")

                if unit_base is not None: sim.physical_units(**unit_base)  # convert units to given gadget units

                # Make sure the data fits into one files. The magic numbers are:
                # 12 - the largest block is likely to be POS with 12 bytes per particle.
                # 2**31 is the largest size a gadget block can safely have
                if sim.__len__() * 12. > (2 ** 31 - 1)*num_files:
                    raise OSError("Data too large to fit into %d gadget file(s). Cannot write. Must have at least num_files >= %d" % (num_files, int(sim.__len__() * 12./(2 ** 31 - 1))+1))

                # Make npart
                def _get_npart(_sim):
                    # alternatively, if you want to hardcode it for Romulus: return np.array([len(_sim.g), len(_sim.dm), 0, 0, len(_sim.s), 0])
                    npart = np.zeros(N_TYPE, int)
                    arr_name = (list(_sim.keys()) + _sim.loadable_keys())[0]
                    for f in _sim.families():
                        # Note that if we have more than one type per family, we cannot
                        # determine which type each individual particle is, so
                        # assume they are all the first.
                        npart[np.min(gadget_type(f))] = len(_sim[f][arr_name])
                    return npart
                
                npartTotal = _get_npart(sim)
                if _debug_print: 
                        print("npartTotal =", npartTotal)

                # split the files
                npart_1file = npartTotal/num_files # number of particles in 1 file (float)

                for file_index in range(num_files):
                    if _debug_print: 
                        print("Writing file %d/%d ..." % (file_index+1, num_files))
                    
                    subslice = [slice(int(file_index*s), int((file_index+1)*s)) for s in npart_1file]
                    if _debug_print: 
                        print("  subslice =", subslice)
                    
                    subsim = combine_snaps(*[sim[f][subslice[np.min(gadget_type(f))]] for f in sim.families()], _debug_print=_debug_print)
                    if _debug_print: 
                        print("  subsim =", subsim)
                    
                    npart = _get_npart(subsim)
                    filename_extension = "" if num_files == 1 else "." + str(file_index)  ##.zfill(int(np.log10(num_files))+1)
                    
                    # Construct a header
                    # FORM: npart, mass, time, redshift, BoxSize, Omega0, OmegaLambda, HubbleParam, num_files=1
                    gheader = GadgetHeader(
                        npart,                     # number of particles for each of the N_TYPE=6 types
                        np.zeros(N_TYPE, float),   # if non-zero, specifies the mass of all particles in a type; if zero: the mass is specified in each particle 
                        subsim.properties["a"],       # Time of output, or expansion factor for cosmological simulations
                        subsim.properties["z"],       # redshift
                        subsim.properties["boxsize"].in_units(subsim['pos'].units, **subsim.conversion_context()),
                        subsim.properties["omegaM0"], 
                        subsim.properties["omegaL0"], 
                        subsim.properties["h"], 
                        num_files=num_files,
                        npartTotal=npartTotal
                    )
                    
                    # Construct the block_names; each block_name needs partlen, data_type, and p_types,
                    # as well as a name. Blocks will hit the disc in the order they are in all_keys.
                    # First, make pos the first block and vel the second.
                    all_keys = _output_order_gadget(all_keys)  # order the keys

                    # No writing format 1 files.
                    block_names = []
                    for k in all_keys:
                        types = np.zeros(N_TYPE, bool)
                        for f in subsim.families():
                            try:
                                dtype = subsim[f][k].dtype
                                types[np.min(gadget_type(f))] += True
                                try:
                                    partlen = np.shape(subsim[f][k])[1]  # *dtype.itemsize
                                except IndexError:
                                    partlen = 1  # dtype.itemsize
                            except KeyError:
                                pass
                        bb = WriteBlock(partlen, dtype=dtype, types=types, name=_translate_array_name(k).upper().ljust(4)[0:4])
                        block_names.append(bb)
                    
                    # Create an output file
                    out_file = GadgetWriteFile(filename+filename_extension, npart, block_names, gheader)
                    
                    # Write the header
                    out_file.write_header(gheader, filename+filename_extension)
                    
                    # Write all the arrays
                    for x in all_keys:
                        g_name = _to_raw(_translate_array_name(x).upper().ljust(4)[0:4])

                        for fam in subsim.families():
                            try:
                                data = subsim[fam][x]
                                gfam = np.min(gadget_type(fam))
                                out_file.write_block(g_name, gfam, data, filename=filename+filename_extension)
                            except KeyError:
                                pass
                return

        # Write headers
        if filename != None:
            if np.size(sim._files) > 1:
                for i in np.arange(0, np.size(sim._files)):
                    ffile = filename + "." + str(i)
                    sim._files[i].write_header(sim.header, ffile)
            else:
                sim._files[0].write_header(sim.header, filename)
        else:
            # Call write_header for every file.
            [f.write_header(sim.header) for f in sim._files]
        # Call _write_array for every array.
        for x in all_keys:
            GadgetSnap._write_array(sim, x, filename=filename)



class GadgetHDF(FileType):
    """Class for GadgetHDF conversion. Requires H5py Python module."""

    def write(self, filename, log=False):
        """Writes file at filename in Gadget HDF5 format (recognizable to Powderday radiative transfer simulation code)."""
        log_timing("Loading h5py module...", log_it=log)
        import h5py as h5

        try:
            with h5.File(filename, 'w') as f:
                header_attrs = f.create_group('Header').attrs
                for gadgetheadername, gadgetdata in self._GadgetHeaderData.items():
                    log_timing("Creating the Header attribute: {}".format(gadgetheadername), log_it=log)
                    header_attrs.create(gadgetheadername, gadgetdata)
                log_timing(log_it=log)
        
                for groupkeyname, groupdata in self._GadgetPartTypeIData.items():
                    g = f.create_group(groupkeyname)
                    for gadgetkeyname, gadgetdatatuple in groupdata.items():
                        log_timing("Creating the {} entry: {}".format(groupkeyname, gadgetkeyname), log_it=log)
                        data = np.asarray(gadgetdatatuple[0](*gadgetdatatuple[1:]))  # first element in tuple is the function and the rest are the parameters
                        g.create_dataset(gadgetkeyname, data.shape, data=data)
                        #del data    # memory management
        except Exception as e:
            os.remove(filename)
            raise e
        
        log_timing(log_it=log)
        return True



class ASCII_SKIRT(FileType):
    """Class for ASCII conversion in SKIRT-acceptable format. Requires numpy Python module."""

    @staticmethod
    def read_into_dataframe(filename):
        """Function that loads a file in the format of SKIRT input/output. The same as the load_dat_file function
        from fitsdatacube.py in my SKIRT output repo https://github.com/JaedenBardati/skirt-datacube ."""
        # get header
        import pandas as pd

        header = {}
        firstNonCommentRowIndex = None
        with open(filename) as file:
            for i, line in enumerate(file):
                l = line.strip()
                if l[0] == '#':
                    l = l[1:].lstrip()
                    if l[:6].lower() == 'column':
                        l = l[6:].lstrip()
                        split_l = l.split(':')
                        assert len(split_l) == 2 # otherwise, unfamiliar form!
                        icol = int(split_l[0]) # error here means we have the form: # column %s, where %s is not an integer
                        l = split_l[1].lstrip() # this should be the column name
                        header[icol] = l
                else:
                    firstNonCommentRowIndex = i
                    break
        assert firstNonCommentRowIndex is not None # otherwise the entire file is just comments
        
        # get data
        df = pd.read_csv(filename, delim_whitespace=True, skiprows=firstNonCommentRowIndex, header=None)
        
        # adjust column names
        if firstNonCommentRowIndex == 0:
            columns = None
        else:
            columns = [None for i in range(max(header.keys()))]
            for k, v in header.items(): columns[k-1] = v
            assert None not in columns # otherwise, missing column 
            df.columns = columns
        
        return df

    def read(self, filename):
        """Reads the """
        self.df = read_into_dataframe(filename)
        return self.df

    def write(self, filename, log=False, delim=' '):
        """
        Writes file at filename in an ASCII text file form recognizable to SKIRT radiative transfer simulation code. 
        Note that there are two outputted files from this function (the star and the gas files) indicated with _star and _gas, respectively.
        Therefore, the filename parameter is extended to two files (with the respective filename tags).
        """
        star_filename = filename + "_stars.txt"
        gas_filename = filename + "_gas.txt"

        # make star file
        log_timing("Finding the star file data ...", log_it=log)
        star_header = "Stellar particles for a simulated galaxy in SKIRT import format.\n"
        if self._star_comment is not None and self._star_comment != '': 
            star_header += self._star_comment + '\n'
        star_header += '\n'
        for i, column in enumerate(self._star_data.keys()):
            star_header += "Column %d: %s\n" % (i+1, column)

        star_data = np.array([list(np.asarray(star_tuple[0](*star_tuple[1:]))) for star_tuple in self._star_data.values()], dtype=float).T

        # make gas file
        log_timing("Finding the gas file data ...", log_it=log)
        gas_header = "Gas particles for a simulated galaxy in SKIRT import format.\n"
        if self._gas_comment is not None and self._gas_comment != '': 
            gas_header += self._gas_comment + '\n'
        gas_header += '\n'
        for i, column in enumerate(self._gas_data.keys()):
            gas_header += "Column %d: %s\n" % (i+1, column)

        gas_data = np.array([list(np.asarray(gas_tuple[0](*gas_tuple[1:]))) for gas_tuple in self._gas_data.values()], dtype=float).T

        # write files
        try:
            log_timing('Writing star file ...', log_it=log)
            np.savetxt(star_filename, star_data, fmt='%.7g', delimiter=' ', newline='\n', header=star_header, comments='# ')
            log_timing('Writing gas file ...', log_it=log)
            np.savetxt(gas_filename, gas_data, fmt='%.7g', delimiter=' ', newline='\n', header=gas_header, comments='# ')

        except Exception as e:
            os.remove(star_filename)
            os.remove(gas_filename)
            log_timing(log_it=log)
            raise e
        
        log_timing(log_it=log)
        return True



class Nchilada_Pynbody(FileType):
    """Class for Pynbody loading. Requires pynbody (and possibly tangos) Python module(s). Only tested on Romulus25 (Nchilada) files."""
    
    DEFAULT_GADGET_UNIT_BASE = {
        "length": "kpc h**-1",
        "velocity": "km s**-1",
        "mass": "10.0e+10 Msun h**-1",
        "time": "kpc s km**-1 h**-1"
    }

    DEFAULT_NCHILADA_UNIT_BASE = {        # note: these are the values (in a form that yt understands) that i currently use when loading the HDF5 result in yt (unit_base, currently adjusted in powderday)
        "length": '2.50e+04 kpc a', 
        "mass": '1.99e+15 Msol', 
        "velocity": '5.85e+02 km s**-1 a',
        "time": '4.27e+01 kpc s km**-1',        ## this may pose an issue of some sort: I set it here to length/velocity, but it is untested (if the yt loads it that way with the unit_base or some other way)
        "density": "1.27e+02 Msol kpc**-3 a**-3",   ## same as above: found with mass/length**3
        "spenergy": "3.42e+05 km**2 a**2 s**-2",   ## specific energy: same as above: found with velocity**2
        "mdot": "4.66e+13 Msol km kpc**-1 s**-1",   ## Mass/time
    }
    
    def read(self, sim, extra_data=None, tangosdb=None):
        """Reads Nchilada files at filename using Pynbody. """
        if type(sim) is str:
            # assume it is a filename and attempt to load the data.
            log_timing("Attempting to load simulation at {} in pynbody . . .".format(sim))
            import pynbody
            sim = pynbody.load(sim)
        # otherwise, assume sim is a pynbody snapshot or halo object.

        self.sim = sim
        self.extra_data = extra_data  # extra data is a dictionary of data to be added to the hdf5 file.
        self.tangos_db = tangos_db
        return self.sim
        

    def convert_to(self, output_file_type, unit_base=None):
        """Converts the current filetype to the entered file type  and underlying loader/writer. Returns converted file type object."""
        from pynbody.array import SimArray
        from pynbody.units import Unit
        
        sim = self.sim  
        _a, _h = sim.properties['a'], sim.properties['h']

        if output_file_type == GadgetHDF:
            if unit_base is None: unit_base = Nchilada_Pynbody.DEFAULT_NCHILADA_UNIT_BASE

            GadgetHeaderData = {
                'BoxSize':                  sim.properties['boxsize'].in_units(unit_base['length'], a=_a, h=_h),
                'HubbleParam':              _h,                                                        # little h (no units, not H0)
                'MassTable':                [0, 0, 0, 0, 0, 0],                                        # mass is specified for each particle for each particle type (so all set to 0)
                'NumFilesPerSnapshot':      1,                                                         # we will only save to 1 file per halo for now
                'NumPart_ThisFile':         [len(sim.g), len(sim.dm), 0, 0, len(sim.s), 0],         # number of particles in form [PartType0, PartType1, ..., PartType6]
                'NumPart_Total':            [len(sim.g), len(sim.dm), 0, 0, len(sim.s), 0],         # we will only save to 1 file for now (so NumPart_Total = NumPart_ThisFile)
                'NumPart_Total_HighWord':   [0, 0, 0, 0, 0, 0],                                        # not important for single galaxy files
                'Omega0':                   sim.properties['omegaM0'],                                # according to the way yt loads, this is really omegaM0 
                'OmegaLambda':              sim.properties['omegaL0'],
                'Redshift':                 1.0/_a - 1,
                'Time':                     sim.properties['time'].in_units(unit_base['time'], a=_a, h=_h),
            }

            for k, v in self.extra_data.items():
                GadgetHeaderData[k] = v

            GadgetPartTypeIData = { # In the form:  keyname: (function, param1, param2, ...)
                'PartType0': {        ## (gas)
                    'Coordinates':          (lambda h: (h['pos'].in_units('kpc', a=_a, h=_h)).in_units(unit_base['length'], a=_a, h=_h), sim.g),
                    'Velocities':           (lambda h: (h['vel']).in_units(unit_base['velocity'], a=_a, h=_h), sim.g),
                    'ParticleIDs':          (lambda h: h['iord'], sim.g),
                    'Masses':               (lambda h: h['mass'].in_units(unit_base['mass'], a=_a, h=_h), sim.g),          # only for types with variable mass (when MassTable[i] = 0)
                    'InternalEnergy':       (lambda h: (h['u']).in_units(unit_base['spenergy'], a=_a, h=_h), sim.g),       # Thermal energy per unit mass: only for SPH particle types
                    'Density':              (lambda h: h['rho'].in_units(unit_base['density'], a=_a, h=_h), sim.g),        # only for SPH particle types
                    'SmoothingLength':      (lambda h: h['smooth'].in_units(unit_base['length'], a=_a, h=_h), sim.g),      # SPH smoothing length h
                    'Potential':            (lambda h: h['phi'].in_units(unit_base['spenergy'], a=_a, h=_h), sim.g),       # Gravitational potential of particles
                    'FractionH2':           (lambda h: np.zeros(len(h)), sim.g),                                           # Romulus does not use H2 
                    'StarFormationRate':    (lambda h: np.zeros(len(h)), sim.g)                                            # units of Mass/time; Romulus does not track this
                }, 'PartType1': {   ## (dark matter)
                    'Coordinates':          (lambda h: (h['pos'].in_units('kpc', a=_a, h=_h)).in_units(unit_base['length'], a=_a, h=_h), sim.dm),
                    'Velocities':           (lambda h: (h['vel']).in_units(unit_base['velocity'], a=_a, h=_h), sim.dm), 
                    'ParticleIDs':          (lambda h: h['iord'], sim.dm),
                    'Masses':               (lambda h: h['mass'].in_units(unit_base['mass'], a=_a, h=_h), sim.dm), 
                    'Potential':            (lambda h: h['phi'].in_units(unit_base['spenergy'], a=_a, h=_h), sim.dm)
                }, 'PartType4': {    ## (stars)
                    'Coordinates':          (lambda h: (h['pos'].in_units('kpc', a=_a, h=_h)).in_units(unit_base['length'], a=_a, h=_h), sim.s),
                    'Velocities':           (lambda h: (h['vel']).in_units(unit_base['velocity'], a=_a, h=_h), sim.s),
                    'ParticleIDs':          (lambda h: h['iord'], sim.s),
                    'Masses':               (lambda h: h['mass'].in_units(unit_base['mass'], a=_a, h=_h), sim.s), 
                    'Potential':            (lambda h: h['phi'].in_units(unit_base['spenergy'], a=_a, h=_h), sim.s),
                    'StellarFormationTime': (lambda h: h['tform'].in_units(unit_base['time'], a=_a, h=_h), sim.s)      # could also use aform for scale factor (unitless)
                }
            }
            

            ## Gas Smoothing Length
            ##   this is for the case where there is not enough gas particles to create a sensible smoothing length from (pynbody: < 25 particles)
            set_smoothing_length = None #"10 kpc"  # OPTION: None uses the smoothing lengths from the Nchilada file, otherwise, it overwrites it with that value
            
            if set_smoothing_length is not None: 
                GadgetPartTypeIData['PartType0']['SmoothingLength'] = (
                    lambda h: SimArray(np.ones(len(h)), Unit(set_smoothing_length)).astype('float32').in_units(unit_base['length'], a=_a, h=_h), sim.g
                )
            ####

            ## Metallicity
            partitioned_metallicity = False  # OPTION: splits up the metals in bins, generally safer to set to False
            
            if partitioned_metallicity:
                GadgetPartTypeIData['PartType0']['Metallicity'] = (lambda h: np.transpose([
                    h['metals'],            # metals 00
                    h['HeI'] + h['HeII'],      # He  01
                    np.zeros(len(h)),          # C   02
                    np.zeros(len(h)),          # N   03
                    h['OxMassFrac'],           # O   04
                    np.zeros(len(h)),          # Ne  05
                    np.zeros(len(h)),          # Mg  06
                    np.zeros(len(h)),          # Si  07
                    np.zeros(len(h)),          # S   08
                    np.zeros(len(h)),          # Ca  09
                    h['FeMassFrac']            # Fe  10
                  ]), sim.g)
               
                GadgetPartTypeIData['PartType4']['Metallicity'] = (lambda h: np.transpose([
                    h['metals'],            # metals 00
                    np.zeros(len(h)),          # He  01
                    np.zeros(len(h)),          # C   02
                    np.zeros(len(h)),          # N   03
                    h['OxMassFrac'],           # O   04
                    np.zeros(len(h)),          # Ne  05
                    np.zeros(len(h)),          # Mg  06
                    np.zeros(len(h)),          # Si  07
                    np.zeros(len(h)),          # S   08
                    np.zeros(len(h)),          # Ca  09
                    h['FeMassFrac']            # Fe  10
                  ]), sim.s)
                  
            else:
                metals_override = None  # OPTION: overrides all metals values (for instance, 0 or 1), does not override if set to None
            
                GadgetPartTypeIData['PartType0']['Metallicity'] = (lambda h: np.ones(len(h))*metals_override if metals_override is not None else h['metals'], sim.g)
                GadgetPartTypeIData['PartType4']['Metallicity'] = (lambda h: np.ones(len(h))*metals_override if metals_override is not None else h['metals'], sim.s)
            ####

            ## Extra black holes info
            blackholes = False  # flag to turn on/off black holes (this is not needed for Powderday unless you are including AGN)
            black_hole_type = 'BH_central'  # 'BH_central' or 'BH'
            
            if blackholes:
                tangos_db = self.tangos_db
                if tangos_db is None:
                    raise NotImplementedError('No support for explicit black holes unless a tangos database is added.')

                if black_hole_type in tangos_db: # only if there are black holes in the halo
                    BHs = tangos_db[black_hole_type]
                    if type(BHs) is db.core.halo.BH: BHs = [BHs] # if only one black hole
                    GadgetHeaderData['NumPart_Total'][5], GadgetHeaderData['NumPart_ThisFile'][5] = len(BHs), len(BHs)  ## we will store the info in PartType5
                    GadgetPartTypeIData['PartType5'] = {
                        'Coordinates': (lambda h, hdb: SimArray([bh['BH_central_offset'] for bh in hdb[black_hole_type]], 'kpc').in_units(unit_base['length'], a=_a, h=_h), sim, tangos_db),  # could have +hdb['shrink_center']
                        'BH_Mass': (lambda h, hdb: SimArray([bh['BH_mass'] for bh in hdb[black_hole_type]], 'Msol').in_units(unit_base['mass'], a=_a, h=_h), sim, tangos_db),
                        'BH_Mdot': (lambda h, hdb: SimArray([bh['BH_mdot'] for bh in hdb[black_hole_type]], 'Msol yr**-1').in_units(unit_base['mdot'], a=_a, h=_h), sim, tangos_db)
                    }
            ####

            converted = GadgetHDF()
            converted._GadgetHeaderData = GadgetHeaderData
            converted._GadgetPartTypeIData = GadgetPartTypeIData

            return converted

        elif output_file_type == ASCII_SKIRT:
            converted = ASCII_SKIRT()
            converted._star_comment = 'Converted from Nchilada file format using Pynbody and code by Jaeden Bardati.'
            converted._gas_comment = 'Converted from Nchilada file format using Pynbody and code by Jaeden Bardati.'

            converted._star_data = {
                'x-coordinate (kpc)': (lambda h: h['pos'][:,0].in_units('kpc', a=_a, h=_h), sim.s), 
                'y-coordinate (kpc)': (lambda h: h['pos'][:,1].in_units('kpc', a=_a, h=_h), sim.s),
                'z-coordinate (kpc)': (lambda h: h['pos'][:,2].in_units('kpc', a=_a, h=_h), sim.s),
                'smoothing length (kpc)': (lambda h: h['smooth'].in_units('kpc', a=_a, h=_h), sim.s),
                'velocity vx (km/s)': (lambda h: h['vel'][:,0].in_units('km s**-1', a=_a, h=_h), sim.s),
                'velocity vy (km/s)': (lambda h: h['vel'][:,1].in_units('km s**-1', a=_a, h=_h), sim.s),
                'velocity vz (km/s)': (lambda h: h['vel'][:,2].in_units('km s**-1', a=_a, h=_h), sim.s),
                'initial mass (Msun)': (lambda h: h['massform'].in_units('Msol', a=_a, h=_h), sim.s),
                'metallicity (1)': (lambda h: h['metals'], sim.s),
                'age (Gyr)': (lambda h: h['age'].in_units('Gyr', a=_a, h=_h), sim.s) 
            }
            converted._gas_data = {
                'x-coordinate (kpc)': (lambda h: h['pos'][:,0].in_units('kpc', a=_a, h=_h), sim.g),
                'y-coordinate (kpc)': (lambda h: h['pos'][:,1].in_units('kpc', a=_a, h=_h), sim.g),
                'z-coordinate (kpc)': (lambda h: h['pos'][:,2].in_units('kpc', a=_a, h=_h), sim.g),
                'smoothing length (kpc)': (lambda h: h['smooth'].in_units('kpc', a=_a, h=_h), sim.g),
                'mass (Msun)': (lambda h: h['mass'].in_units('Msol', a=_a, h=_h), sim.g),
                'metallicity (1)': (lambda h: h['metals'], sim.g),
                'velocity vx (km/s)': (lambda h: h['vel'][:,0].in_units('km s**-1', a=_a, h=_h), sim.g),
                'velocity vy (km/s)': (lambda h: h['vel'][:,1].in_units('km s**-1', a=_a, h=_h), sim.g),
                'velocity vz (km/s)': (lambda h: h['vel'][:,2].in_units('km s**-1', a=_a, h=_h), sim.g)
            }

        elif output_file_type == GadgetBinary:
            converted = GadgetBinary()
            converted.sim = sim
        else:
            raise NotImplementedError("File Type {} not supported.".format(output_file_type))

        return converted


##### CONVENIENCE FUNCTIONS #####

def pynbody2gadget(sim, filename, *args, **kwargs):
    Nchilada_Pynbody(sim).convert_to(GadgetBinary).write(filename, *args, **kwargs)

def pynbody2gadgetHDF5(sim, filename, *args, **kwargs):
    Nchilada_Pynbody(sim).convert_to(GadgetHDF).write(filename, *args, **kwargs)

def pynbody2skirt(sim, filename, *args, **kwargs):
    Nchilada_Pynbody(sim).convert_to(ASCII_SKIRT).write(filename, *args, **kwargs)

def load_skirt(filename):
    return ASCII_SKIRT.read_into_dataframe(filename)


SUPPORTED_OUTPUT_TYPES = {
    None: GadgetHDF,  # default type
    'gadgethdf5': GadgetHDF,
    'gadgethdf': GadgetHDF,
    'gadgeth5': GadgetHDF,
    'gadgetbinary': GadgetBinary,
    'gadget2': GadgetBinary,
    'gadget': GadgetBinary,
    'ascii_skirt': ASCII_SKIRT,
    'asciiskirt': ASCII_SKIRT,
    'ascii': ASCII_SKIRT,
    'skirt': ASCII_SKIRT,
}


##### RUNNNING THE FILE WITH ARGUMENTS #####

if __name__ == "__main__":
    try:
        from file_arguments import file_arguments
    except ModuleNotFoundError:
        print('Unable to find custom file_arguments module. Exiting program . . .')
        exit()

    IN_FN, OUT_FN, TO_TYPE, IN_HN = file_arguments.get(str, str, str, int, fill_empties_with_none=True)

    if None in [IN_FN, OUT_FN, TO_TYPE]:
        raise Exception('The first three arguments must be entered.')

    TO_TYPE = TO_TYPE.lower() if TO_TYPE is not None else None
    if TO_TYPE not in SUPPORTED_OUTPUT_TYPES.keys():
        raise Exception('Output type not supported.')
    TO_TYPE_CLASS = SUPPORTED_OUTPUT_TYPES[TO_TYPE]

    log_timing("Loading simulation at {} in pynbody . . .".format(IN_FN))
    sim = pynbody.load(IN_FN)
    if IN_HN is not None:
        log_timing("Loading halo {} . . .".format(IN_HN))
        sim = sim.halos(dosort=True).load_copy(IN_HN)

    sim_obj = Nchilada_Pynbody(sim)
    log_timing("Converting file format to {}:\n".format(repr(TO_TYPE_CLASS)))
    sim_obj.convert_to(TO_TYPE_CLASS).write(OUT_FN, log=True)
    log_timing('\nSaved converted file at {}.'.format(OUT_FN))
