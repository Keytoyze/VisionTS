from setuptools import setup

setup(
    name='visionts',
    version='0.1.4',
    author='Mouxiang Chen',
    author_email='chenmx@zju.edu.cn',
    description='Using a vision MAE for time series forecasting.',
    long_description=open('README.md', encoding='utf8').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/Keytoyze/VisionTS',
    packages=['visionts'],
    python_requires='>=3.6',
    install_requires=[
        "torch",
        "torchvision",
        "einops",
        "numpy",
        "timm"
    ],
)