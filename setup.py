from setuptools import setup, find_packages

setup(
    name='TrustLLM',  # 替换为你的包名
    version='0.1.0',  # 你的包的当前版本
    description='TrustLLM',  # 一个简短的项目描述
    author='Yue Huang',  # 替换为你的名字
    author_email='your.email@example.com',  # 替换为你的邮箱地址
    url='https://github.com/HowieHwong/TrustLLM',  # 项目的URL，通常是GitHub/GitLab等的仓库地址
    packages=find_packages(),  # 自动发现项目中的所有包
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
