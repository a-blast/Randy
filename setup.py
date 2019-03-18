from setuptools import setup

setup(name='randyhand',
      version='0.3.2',
      description="Generate random images handwritten words for OCR training",
      url="https://github.com/a-blast/randyhand",
      author="Austin Armstrong",
      license='MIT',
      packages=['randyhand'],
      install_requires=[
          'numpy',
          'requests',
          'pillow',
          'pandas',
      ],
      zip_safe=False)
