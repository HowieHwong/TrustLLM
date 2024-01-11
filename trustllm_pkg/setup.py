from setuptools import setup, find_packages

setup(
    name='trustllm',
    version='0.1.0', 
    description='TrustLLM',  
    author='Yue Huang & Siyuan Wu & Haoran Wang',
    author_email='trustllm.benchmark@gmail.com',
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
