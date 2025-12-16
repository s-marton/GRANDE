from setuptools import setup, find_packages

setup(
    name="GRANDE",
    license="MIT",
    version="0.2.0",
    packages=find_packages(),
    install_requires=[
        'numpy>=1.25.0,<2.4.0',
        'pandas>=2.0.0,<2.4.0',
        'torch>=2.6,<2.10',
        'scikit-learn>=1.4.0,<1.8.0',
        'category-encoders>=2.6.4,<=2.9.0',
        'tqdm>=4.38,<5',
        'autogluon>=1.3.0,<=1.4.0',
    ],
    author="Sascha Marton",
    author_email="sascha.marton@gmail.com",
    description="A novel ensemble method for hard, axis-aligned decision trees learned end-to-end with gradient descent.",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/s-marton/GRANDE",
)