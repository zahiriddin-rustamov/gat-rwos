import sys
from .main import main as run_main

def main():
    if 'src.gat_rwos.main' in sys.argv:
        sys.argv.remove('src.gat_rwos.main')
    run_main()

if __name__ == '__main__':
    main()