from setuptools import setup, find_packages

setup(
    name='credictor',
    version='0.3.0',
    packages=find_packages(),
    include_package_data=True,
    url='',
    license='',
    author='Webmedia DataHub',
    author_email='datahub@g.globo',
    description='A series ending credits start predictor package',
    install_requires=[
        "numpy==1.15.4",
        "opencv-python==3.4.5.20",
        "pandas==1.0.0",
        "scikit-learn==0.20.0",
        "scipy==1.2.1",
        "tqdm==4.42.1"
    ],
    tests_require=[
        "pytest==5.3.5",
        "expects==0.9.0",
        "pytest-watch==4.2.0",
        "doubles==1.5.3"
    ]
)
