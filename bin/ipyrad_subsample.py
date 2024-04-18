#!/usr/bin/env python

import ipyrad as ipp
import argparse

def subsample_file(**kwargs):
    ## load and assembly object
	data1 = ipp.load_json(kwargs['assembly_name']+'.json')
    
	#gathering samples with good enough depth
	keep=list(data1.stats_dfs.s3.index[data1.stats_dfs.s3["clusters_hidepth"]>10000])
	#branch assembly object
	child = data1.branch(kwargs['new_name'], subsample=keep)

	child.save()
	child.write_params()

def main():
	parser = argparse.ArgumentParser(description='Subsample ipyrad assembly object based on clusters_hidepth', argument_default=argparse.SUPPRESS)
	parser.add_argument('--assembly_name', help='Name used as prefix for assembly file', required=True)
	parser.add_argument('--new_name', help='Name to use as prefix for branched assembly directories', required=True)
	args = vars(parser.parse_args())
	subsample_file(**args)
