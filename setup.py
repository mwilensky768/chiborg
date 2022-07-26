from setuptools import setup
import os
import sys
import json


def branch_scheme(version):
    """
    Local version scheme that adds the branch name for absolute reproducibility.
    If and when this is added to setuptools_scm this function and file can be removed.
    Reproduced from SSINS (this fn. originally written by P. Laplant.)
    """
    if version.exact or version.node is None:
        return version.format_choice("", "+d{time:{time_format}}", time_format="%Y%m%d")
    else:
        if version.branch in ["main", "master"]:
            return version.format_choice("+{node}", "+{node}.dirty")
        else:
            return version.format_choice("+{node}.{branch}", "+{node}.{branch}.dirty")


def package_files(package_dir, subdirectory):
    # walk the input package_dir/subdirectory
    # return a package_data list
    paths = []
    directory = os.path.join(package_dir, subdirectory)
    for (path, directories, filenames) in os.walk(directory):
        for filename in filenames:
            path = path.replace(package_dir + '/', '')
            paths.append(os.path.join(path, filename))
    return paths


# data_files = package_files('chiborg', 'data')

setup_args = {
    'name': 'chiborg',
    'author': 'M. Wilensky',
    'url': 'https://github.com/mwilensky768/chiborg',
    'license': 'MIT',
    'description': 'Bayesian Jackknife Test',
    'package_dir': {'chiborg': 'chiborg'},
    'packages': ['chiborg'],
    'include_package_data': True,
    # 'package_data': {'SSINS': data_files},
    'setup_requires': ['setuptools_scm'],
    'use_scm_version': {
        "write_to": "chiborg/version.py",
        "local_scheme": branch_scheme,
    },
    'install_requires': ['numpy', 'scipy'],
    'zip_safe': False,
}


if __name__ == '__main__':
    setup(**setup_args)
