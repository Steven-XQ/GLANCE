"""Generic experiment runner - passes all CLI args to traineval.py"""
import sys
import os
sys.path.append('.')

if __name__ == '__main__':
    # All args after the script name are passed to traineval.py
    extra_args = ' '.join(sys.argv[1:])
    COMMANDLINE = f"python traineval.py {extra_args}"
    print(COMMANDLINE)
    os.system(COMMANDLINE)
