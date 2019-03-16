from setuptools import setup, find_packages

setup(name='trainer',
	version='0.1',
	packages=find_packages(),
	description='AI that generates short poems',
	author='Bruno Freeman',
	author_email='brunofreeman21@gmail.com',
	license='MIT',
	install_requires=[
		'keras',
		'h5py'],
	zip_safe=False)