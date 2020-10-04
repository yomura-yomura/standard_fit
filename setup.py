from setuptools import setup, find_packages

setup(
    name='standard_fit',
    version='1.0',
    description='',
    author='yomura',
    author_email='yomura@hoge.jp',
    url='https://github.com/yomura-yomura/standard_fit',
    packages=find_packages(),
    install_requires=[
        "numpy", "iminuit"
    ]
)
