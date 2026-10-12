"""Compute-efficient relaxed RNG at 10%, 30%, 45%; reuse completed 20%."""
from compare_dynamic_overlap import main

if __name__ == '__main__':
    main(multibudget_rng=True)
