from setuptools import setup

if __name__=='__main__':
    setup(
        name='synthnoise',
        version='0.1',
        packages=['synthnoise'],
        install_requires=['numpy', 'scipy'],
        package_data={'synthnoise': ['*.npz']}
    )
