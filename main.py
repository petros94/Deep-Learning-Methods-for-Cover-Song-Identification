import logging

from experiments.execute_experiment import execute_single

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    execute_single()
