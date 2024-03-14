from setuptools import setup, find_packages

setup(
    name="GRANDE",
    license="MIT",
    version="0.1.3",
    packages=find_packages(),
    install_requires=[
        'category_encoders==2.6.1',
        'pandas>=2.0.3',
        'tensorflow==2.13',
        'numpy>=1.24.3',
        'scikit_learn>=1.3.0',
        'focal_loss==0.0.7',
    ],
    author="Sascha Marton",
    author_email="sascha.marton@gmail.com",
    description="A novel ensemble method for hard, axis-aligned decision trees learned end-to-end with gradient descent.",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/s-marton/GRANDE",
)
