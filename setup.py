#!/usr/bin/env python
# setup.py adapted from the PySCF. Reused under the Apache 2.0 License

import os
import sys
from setuptools import setup
from setuptools.command.build_py import build_py
from pathlib import Path

def get_platform():
    from distutils.util import get_platform
    platform = get_platform()
    if sys.platform == 'darwin':
        arch = os.getenv('CMAKE_OSX_ARCHITECTURES')
        if arch:
            osname = platform.rsplit('-', 1)[0]
            if ';' in arch:
                platform = f'{osname}-universal2'
            else:
                platform = f'{osname}-{arch}'
        elif os.getenv('_PYTHON_HOST_PLATFORM'):
            # the cibuildwheel environment
            platform = os.getenv('_PYTHON_HOST_PLATFORM')
            if platform.endswith('arm64'):
                os.putenv('CMAKE_OSX_ARCHITECTURES', 'arm64')
            elif platform.endswith('x86_64'):
                os.putenv('CMAKE_OSX_ARCHITECTURES', 'x86_64')
            else:
                os.putenv('CMAKE_OSX_ARCHITECTURES', 'arm64;x86_64')
    return platform

class CMakeBuildPy(build_py):

    # List of CMake projects
    PROJECTS = [
        "lib",
        # "my_pyscf/lib"
    ]

    def CMake_build(self, src_dir):
        """Build CMake project found in `src_dir`

        Build artifict will be added to the `build` directory in the project root.
        Extension modules will be left in-place in the source directoris.
        """
        src_dir = Path(src_dir)

        # Create build directory
        build_dir = (src_dir / "build" / f"tmp.{self.plat_name}").absolute()
        build_dir.mkdir(parents=True, exist_ok=True)

        # Pass CMake an absolute path
        src_dir = src_dir.absolute()

        # Configure project
        self.announce(f'Configuring {src_dir}...', level=0)
        cmd = ['cmake', f'-S{src_dir}', f'-B{build_dir}'] + self.configure_args
        self.spawn(cmd)

        # Build project
        self.announce(f'Building {src_dir}...', level=3)
        # By default do not use high level parallel compilation.
        # OOM may be triggered when compiling certain functionals in libxc.
        # Set the shell variable CMAKE_BUILD_PARALLEL_LEVEL=n to enable
        # parallel compilation.
        cmd = ['cmake', '--build', build_dir] + self.build_args
        if self.dry_run:
            self.announce(' '.join(cmd))
        else:
            self.spawn(cmd)

    def run(self):
        self.plat_name = get_platform()

        configure_args = os.getenv('CMAKE_CONFIGURE_ARGS')
        self.configure_args = configure_args.split(' ') if configure_args else []

        build_args = os.getenv('CMAKE_BUILD_ARGS')
        self.build_args = build_args.split(' ') if build_args else []

        # Build each project
        for project in self.PROJECTS:
            self.CMake_build(project)

        super().run()

# build_py will produce plat_name = 'any'. Patch the bdist_wheel to change the
# platform tag because the C extensions are platform dependent.
# For setuptools<70
from wheel.bdist_wheel import bdist_wheel
initialize_options_1 = bdist_wheel.initialize_options
def initialize_with_default_plat_name(self):
    initialize_options_1(self)
    self.plat_name = get_platform()
    self.plat_name_supplied = True
bdist_wheel.initialize_options = initialize_with_default_plat_name

# For setuptools>=70
try:
    from setuptools.command.bdist_wheel import bdist_wheel
    initialize_options_2 = bdist_wheel.initialize_options
    def initialize_with_default_plat_name(self):
        initialize_options_2(self)
        self.plat_name = get_platform()
        self.plat_name_supplied = True
    bdist_wheel.initialize_options = initialize_with_default_plat_name
except ImportError:
    pass


setup(
    #include *.so *.dat files. They are now placed in MANIFEST.in
    #package_data={'': ['*.so', '*.dylib', '*.dll', '*.dat']},
    include_package_data=True,  # include everything in source control
    cmdclass={'build_py': CMakeBuildPy},
)
