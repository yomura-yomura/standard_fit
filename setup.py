from setuptools import setup, find_packages

setup(
    name='standard_fit',
    version='4.4.0',
    description='',
    author='yomura',
    author_email='yomura@hoge.jp',
    url='https://github.com/yomura-yomura/standard_fit',
    packages=find_packages(),
    install_requires=[
        "iminuit >= 2.0.0",
        "scipy",
        "plotly_utility @ git+https://github.com/yomura-yomura/plotly_utility",
        "numpy_utility @ git+https://github.com/yomura-yomura/numpy_utility",
        "parse",
        "Pillow",
        "matplotlib"
    ]
)
