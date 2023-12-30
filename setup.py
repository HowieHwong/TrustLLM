from setuptools import setup, find_packages

setup(
    name='TrustLLM',  
    version='0.1.0', 
    description='TrustLLM',  
    author='Yue Huang', 
    author_email='your.email@example.com',  
    url='https://github.com/HowieHwong/TrustLLM',  
    packages=find_packages(),  
install_requires=[
    'transformers',
    'numpy>=1.18.1',
    'scipy',
    'pandas>=1.0.3',
    'scikit-learn',
    'openai',
    'tqdm',
    'tenacity',
],
    classifiers=[
    ],
)
