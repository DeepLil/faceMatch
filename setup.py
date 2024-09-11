from setuptools import setup

setup(
    name = 'src',
    version = '0.0.1',
    author = 'Phani Sandeep',
    description= 'A small package for To whom your face would match',
    packages = ['src'],
    python_requires='>3.7',
    install_requires = [
        'mtcnn == 0.1.0',
        'tensorflow==2.3.1',
        'keras ==2.4.3',
        'keras-vggface==0.6',
        'keras_applications==1.0.8',
        'PyYAML',
        'tqdm',
        'scikit-learn',
        'streamlit',
        'bing-image-downloader'

    ]

)