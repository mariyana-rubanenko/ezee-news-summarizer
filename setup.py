from setuptools import setup
from setuptools import find_packages

setup(name='eeze-news-summarizer',
      version='0.0.1',
      description='Summarization with BERT',
      long_description=open("README.md", "r", encoding='utf-8').read(),
      long_description_content_type="text/markdown",
      url='https://github.com/mariyana-rubanenko/ezee-news-summarizer.git',
      download_url='https://github.com/mariyana-rubanenko/ezee-news-summarizer/archive/0.0.1.tar.gz',
      author='Mariya Rubanenko',
      author_email='maryrubik@yandex.ru',
      install_requires=['transformers', 'scikit-learn', 'spacy'],
      packages=find_packages(),
      zip_safe=False)
