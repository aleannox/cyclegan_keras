import setuptools

for module in [
    'config'
]:
    setuptools.setup(name=module, packages=[module])
