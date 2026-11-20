import argparse
from .runtime import warmup

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,default=0);p.add_argument('--max-length',type=int,default=128)
    args=p.parse_args();warmup(args.seed,args.max_length)
