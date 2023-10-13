from setuptools import setup, find_packages

setup(
    name="GRANDE",
    version="0.0.1",
    packages=find_packages(),
    install_requires=[
        'tensorflow==2.13',
        'tensorflow_addons>=0.21.0',
        'numpy>=1.23.2',
        'scikit_learn>=1.3.0'
    ],
    author="Sascha Marton",
    author_email="sascha.marton@gmail.com",
    description=" A novel ensemble method for hard, axis-aligned decision trees learned end-to-end with gradient descent.",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/s-marton/GRANDE",
)
