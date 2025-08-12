from lib.circuit_meta import *

def main() -> None:
    idx = CircuitShape(20, 12, 3)
    print()
    for tile in idx.tiles(1):
        print(tile)

if __name__ == "__main__":
    main()
