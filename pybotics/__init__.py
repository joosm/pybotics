"""Pybotics modules."""
from importlib import import_module
from pathlib import Path

# glob modules
path = Path(__file__).parent
modules = list(path.glob('*.py'))

# import
for mod in modules:
    import_module('.{}'.format(mod.stem), package=path.name)
