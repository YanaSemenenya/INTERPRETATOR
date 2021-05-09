from setuptools import setup, find_packages

setup(
    name='yanapy',
    version='0.0.1',
    description="Black box model interpretator",
    url='http://google.com',
    author='Yana Semenenya',
    author_email='',
    license='MIT',
    packages=find_packages(),
    keywords=['modeling', 'black box', 'statistics'],  # arbitrary keywords
    zip_safe=False,
    test_suite='tests',
    install_requires=[
        'shap',
        'skater'
    ]
)