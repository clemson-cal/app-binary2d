import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="cdc_tools",
    version="0.1.8",
    author="J. Zrake",
    author_email="jzrake@clemson.edu",
    description="Python module to work with data output from the CDC code",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/clemson-cal/app-binary2d",
    packages=setuptools.find_packages(),
    entry_points={
        'console_scripts': [
            'cdc-plot     = cdc_tools.plot:main',
            'cdc-model    = cdc_tools.model:main',
            'cdc-tseries  = cdc_tools.tseries :main',
            'cdc-upsample = cdc_tools.upsample:main',
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=['numpy', 'scipy', 'matplotlib', 'h5py'],
    python_requires='>=3.6',
)
