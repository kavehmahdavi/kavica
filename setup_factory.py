#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Kavica is a Feature Selection and Cluster interpretation package.
It provides:
- powerful feature selection methods.
- practical (HPC) functions
- convenient dynamic graph object
- useful missing value imputation, transformation, and  handling outliers capabilities
- innovative cluster shape portrayal
- and much more scientific uses
All KAVICA wheels distributed on PyPI are BSD 3 clause licensed.
"""

# Test: $python3 setup_factory.py install

from __future__ import print_function, division
from setuptools import setup

from urllib import request
import os
from fnmatch import fnmatch
import importlib
import urllib
import re
import fnmatch
import sys
import subprocess
import traceback
import json
import textwrap

__doclines__ = (__doc__ or '').split("\n")

# Global data. It should be updated manually.
# ----------------------------------------------------------------------------------------------------------------------
__python_version__ = (3, 6)
if sys.version_info[:2] < __python_version__:
    raise RuntimeError("Python version >= {} required.".format(__python_version__))

__author__ = "KAVEH MAHDAVI"
__license__ = "BSD 3 clause"
__major__ = 0
__minor__ = 1
__micro__ = 0
__isreleased__ = False
__version__ = '{:d}.{:d}.{:d}'.format(__major__, __minor__, __micro__)
__suffixes__ = ["*.py", "*.pyx"]

__classifier__ = """\
Development Status :: 1 - Beta
Intended Audience :: Science/Research
Intended Audience :: Developers
License :: OSI Approved :: MIT License
Programming Language :: C
Programming Language :: Python
Programming Language :: Python :: 3
Programming Language :: Python :: 3.5
Programming Language :: Python :: 3.6
Programming Language :: Python :: 3.7
Programming Language :: Python :: Implementation :: CPython
Topic :: High Performance Computing
Topic :: Scientific/Engineering
Topic :: Software Development :: Build Tools
Operating System :: OS Independent
Natural Language :: English
"""

# General package info.
__metadata__ = dict(
    version=__version__,
    name='kavica',
    maintainer="BSC Performance Tools",
    maintainer_email="tools@bsc.es",
    description=__doclines__[0],
    long_description=[_f for _f in '\n'.join(__doclines__[2:]).split('\n') if _f],
    # Fixme:should be long string
    url="https://github.com/kavehmahdavi/kavica",
    author="Kaveh Mahdavi",
    author_email="kavehmahdavi74@gmail.com",
    download_url="https://pypi.python.org/pypi/kavica",
    license='BSD 3 clause licensed',
    classifiers=[_f for _f in __classifier__.split('\n') if _f],
    platforms=["Windows", "Linux", "Mac OS-X", "Unix"],
    test_suite='nose.collector',
    python_requires='>={}.{}'.format(__python_version__[0], __python_version__[1]),
    include_package_data=True,
    zip_safe=False,
)

# TODO: it is needed to updated manually. (should be automatic)
__cython_build_dict__ = {'./kavica/parser/': 'setup_factory.py'}


# Requirement modules.
# ----------------------------------------------------------------------------------------------------------------------

def get_full_directory_files(root='.', pattern=__suffixes__):
    folders = dict()
    for path, _, file in os.walk(root):
        files = []
        for name in file:
            for suffix in pattern:
                if fnmatch.fnmatch(name, suffix):
                    files.append(name)
        if len(files) == 0:
            pass
        else:
            folders.update({path: files})
    return folders


def __get_imported_modules_generator():
    import_log = []
    folders = get_full_directory_files()
    for folder_item, files in folders.items():
        for file_item in files:
            with open(os.path.join(folder_item, file_item), mode="r") as code:
                lines = code.read()
                matched = re.findall(r"(?<!from)import (\w+)[\n.]|(?<=from) (\w+)|(?<!from)import (\w+)", lines)
                for modules in matched:
                    for module_item in modules:
                        if len(module_item):
                            if module_item not in import_log:
                                yield module_item

def is_builtin_module(url="https://docs.python.org/{}/py-modindex.html".format(__python_version__[0]),
                      module_item=None):

    webp = urllib.request.urlopen(url).read().decode('utf-8')

    pattern = re.compile(r'\b{}\b'.format(str(module_item)))
    match = pattern.search(webp)
    return match


def required_modules_test():
    required_version_dict = dict()
    version_attribute_list = ['__version__',
                              'version',
                              '__version_info__',
                              'format_version']
    # TODO: needed to be parallel.
    # TODO: Generate the builtin list local.
    # TODO: the kavica has to be added with the version specification.
    for module in __get_imported_modules_generator():
        if not is_builtin_module(url="https://docs.python.org/{}/py-modindex.html".format(__python_version__[0]),
                                 module_item=module):
            if module not in required_version_dict.keys():
                try:
                    globals()[module] = importlib.import_module(module)
                    gotten_attribute = next((x for x in version_attribute_list if hasattr(globals()[module], x)),
                                            'last')
                    if gotten_attribute is not 'last':
                        gotten_version = getattr(globals()[module], gotten_attribute)
                    else:
                        gotten_version = 'last'
                    required_version_dict.update({str(module): str(gotten_version)})
                    print('{} ({}) is imported successfully.'.format(str(module), str(gotten_version)))
                    print('{} ({}) is imported successfully.'.format(str(module), str(gotten_version)))
                except ModuleNotFoundError:
                    print('{} is not a solely module.'.format(module))
                except ImportError:
                    print('=ERROR= Could not import {}. Please make sure to install a current version.'.format(module))
        else:
            print('{} is a built in one in python {}.'.format(module, __python_version__))
    return required_version_dict


# Get last git rev.
# ----------------------------------------------------------------------------------------------------------------------
def grab_git_revision():
    try:
        out = subprocess.check_output(['git', 'rev-parse', 'HEAD'])
        __git_revision__ = out.strip().decode('ascii')
    except OSError:
        __git_revision__ = "Unknown"

    return __git_revision__


def get_version_info():
    # Adding the git rev number needs to be done inside write_version_py(),
    # otherwise the import of numpy.version messes up the build under Python 3.
    __fullrevision__ = __version__
    if os.path.exists('.git'):
        __git_revision__ = grab_git_revision()
    elif os.path.exists('numpy/version.py'):
        # must be a source distribution, use existing version file
        try:
            from numpy.version import git_revision as __git_revision__
        except ImportError:
            raise ImportError("Unable to import git_revision. Try removing " \
                              "numpy/version.py and the build directory " \
                              "before building.")
    else:
        __git_revision__ = "Unknown"

    if not __isreleased__:
        __fullrevision__ += '.dev0+' + __git_revision__[:7]

    return __fullrevision__, __git_revision__


def generate_version_py(filename='version.py'):
    cnt = """#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# THIS FILE IS GENERATED FROM KAVICA SETUP.PY
short_version = '%(version)s'
version = '%(version)s'
full_version = '%(full_version)s'
git_revision = '%(git_revision)s'
release = %(isrelease)s
if not release:
    version = full_version
print(short_version)
"""
    __fullrevision__, __git_revision__ = get_version_info()

    a = open(filename, 'w')
    version_info = {'version': __version__,
                    'full_version': __fullrevision__,
                    'git_revision': __git_revision__,
                    'isrelease': str(__isreleased__)}
    try:
        a.write(cnt % version_info)
    finally:
        a.close()
    return version_info


def build_cython():
    old_path = os.getcwd()
    for src_path, cython_setup in __cython_build_dict__.items():
        os.chdir(src_path)
        subprocess.call(["python3", str(cython_setup), 'install'])
        os.chdir(old_path)


def parse_setup_py_commands():
    """Check the commands and respond appropriately.  Disable broken commands.
    Return a boolean value for whether or not to run the build or not.
    """
    args = sys.argv[1:]

    if not args:
        # Do the forced completed setup the package as like as pip install
        return True

    info_commands = ['--name', '--version', '--download_url', '--author',
                     '--python_requires', '--url', '--license', '--description',
                     '--author_email', '--maintainer', '--maintainer_email',
                     '--long_description', '--platforms', '--classifiers']

    # Retrieve information from the package without installing.
    for command in info_commands:
        if command in args:
            info = __metadata__[command.replace('--', '')]
            if isinstance(info, list):
                print("\n".join(info), )
            else:
                print(info)
            return False

    update_commands = {'--Cythonize': 'build_cython',
                       '--V': 'generate_version_py',
                       '--requires': 'required_modules_test'}

    # Retrieve information from the package without installing.
    for command, method in update_commands.items():
        if command in args:
            info = globals()[method]()
            if isinstance(info, dict):
                info = list(info.values())[0]
                for key, value in info.items():
                    print('{} == {}'.format(key, value))
            return False

    # The following commands are supported, but we need to show more
    # useful messages to the user
    if 'install' in args:
        print(textwrap.dedent("""
            Note: if you need reliable uninstall behavior, then install
            with pip instead of using `setup_factory.py install`:
              - `pip install .`       (from a git repo or downloaded source
                                       release)
              - `pip install Kavica`   (last Kavica release on PyPi)
            """))
        return True

    if '--help' in args or '-h' in sys.argv[1]:
        print(textwrap.dedent("""
            Kavica-specific help
            -------------------
            To install Kavica from here with reliable uninstall, we recommend
            that you use `pip install .`. To install the latest Kavica release
            from PyPi, use `pip install Kavica`.
            For help with build/installation issues, please ask on the
            Kavica-discussion mailing list.  If you are sure that you have run
            into a bug, please report it at https://github.com/kavehmahdavi/kavica.
            Setuptools commands help
            ------------------------
            """))
        return False

    # Commands that do more than print info, but also don't need Cython and
    # template parsing.
    other_commands = ['egg_info', 'install_egg_info', 'rotate']
    for command in other_commands:
        if command in args:
            return False

    # If we got here, we didn't detect what setup_factory.py command was given
    import warnings
    warnings.warn("Unrecognized setuptools command, proceeding and Cythonizing", stacklevel=2)

    return True


def setup_package(metadata_avail=False):
    """
    Build and setup the package.
    :param metadata_avail: boolean, True means the metadata is available in json file.
    :return: None
    """
    run_build = parse_setup_py_commands()

    if run_build:
        if not metadata_avail:
            # Rewrite the version file every time.
            version_dict = generate_version_py()

            # Required packages.
            required_dict = required_modules_test()
            # Cython build
            build_cython()

            # Write the package meta data in as json format.
            with open('metadata.json', 'w', encoding=' utf-8') as fp:
                json.dump({**__metadata__,
                           **{'version': version_dict},
                           **{'Required_modules': required_dict}}, fp, ensure_ascii=False, indent=4)
        else:
            # TODO: read from the json file
            generate_version_py()

        try:
            setup(**__metadata__)
        except BaseException as ex:
            # Get current system exception
            ex_type, ex_value, ex_traceback = sys.exc_info()

            # Extract unformulated stack traces as tuples
            trace_back = traceback.extract_tb(ex_traceback)

            # Format stacktrace
            stack_trace = list()

            for trace in trace_back:
                stack_trace.append(
                    "File : %s , Line : %d, Func.Name : %s, Message : %s" % (trace[0], trace[1], trace[2], trace[3]))
            print('=' * 60)
            print("Exception type : %s " % ex_type.__name__)
            print("Exception message : %s" % ex_value)
            print("Stack trace : %s" % stack_trace)
            print('=' * 60)
        return True


if __name__ == '__main__':
    setup_package(metadata_avail=False)
