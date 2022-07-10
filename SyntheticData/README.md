# Synthetic Data

## Dependencies

This script requires the following nonstandard packages to run
 - bpycv
 - cv2
 - numpy
 - bpy (included with blender)

The script also utilizes the following standard packages
 - abc
 - sys 
 - random
 - os
 - dataclasses
 - typing
 - math
 - time 

## Running the script

To execute the python script on Blender using a config file:

    blender -b --python generate.py -- example.cfg
    
The "--" argument is used to separate blender arguments from script arguments, as per Blender's documentation. 

The "-b" flag indicates that blender should be run in the background. This speeds up the execution of the script, and is generally preferred since the blender GUI will not render until the python script has finished executing.

The "--python" flag indicates that the next argument is a python script to execute.

### Script arguments

The script takes several optional arguments, all of which must be specified after the argument separator "--" as mentioned above. Because config files can specify the same amount of information with a more manageable editing scheme, it is recommended that the only argument you supply for execution is the name of the config file immediately after the argument separator. In the example above, the "example.cfg" file is the only argument passed to the python script, and the script's parameters are parsed from that file.
    
### Config Files

The config files for this script are intended to make manipulation of the dataset parameters more flexible, but the parsing algorithm is not very robust (it doesn't even check the file extension at the moment XD). Config files are text files in which each line is a key-value pair separated by an "=". See below for a list of keys currently recommended for use:
 - **dataset**: the name of the dataset. This can be a new dataset or a previously existing one, the script is capable of appending classes to the yaml files when necessary. Defaults to "data".
 - **parts**: a comma-separated list of part numbers for the lego models to be included in the dataset. 
 - **engine**: the engine blender should use to render images. Optional values for this key are: *BLENDER_EEVEE*, or _CYCLES_
 - **size**: the number of images to generate. Defaults to 1000
 - **capacity**: the maximum number of legos to include in a single image.
 - **gravity**: boolean variable with values _on_ or _off_. Enables gravity simulation prior to rendering the image in order to add realism. Defaults to _on_.
 - **split**: controls the split between training, validation, and test data. Should be a comma-separated list of three numeric values in the order _train_, _val_, _test_. The values will automatically be normalized, but will default to 70% train, 20% validate, and 10% test (i.e., "split=7, 2, 1").
 - **use_blend_file**: lets the script know whether Blender was ran with a .blend file. If so, the script will not create the background/lights/camera on its own.
 - **save_generated_scenes**: whether to save a new blend file for every training image generated. Useful for debugging, otherwise just clutters up the folder.
 - **pkg_dir**: this directory is added to PATH on startup. Allows the user to specify where Python should look for modules.

An [example config file](SyntheticData/example.cfg) is available for viewing in the SyntheticData folder. 

## Fender Blenders

If you haven't worked external python scripts and Blender before, there are a few things you need to be sure of:
 - Blender must have access to the python binaries
 - Blender must have access to any python modules imported in the script

Blender will search the PATH environment variable on the machine, and will search its own directory. This leads to two solutions for granting blender access to all python packages and binaries:

1. Ensure that your python directory is in the PATH environment variable
2. Replace Blender's python directory with a directory junction (or symbolic link) that redirects to the binaries and modules.

When using virtual environments, it is generally preferable to implement option two and redirect Blender's python directory to the python directory of the environment designated for operating with blender. 

## LDView and LDraw

If you have LDView and LDraw on your machine and would like to automatically generate stl files for parts that are requested, then you will need to create a file titled gen.inf in the same directory as the generator script, and add the following lines of text.

    ldraw=/path/to/ldraw/parts/
    ldview=/path/to/ldview/excutables/
