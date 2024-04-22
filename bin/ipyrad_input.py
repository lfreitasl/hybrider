#!/usr/bin/env python3

import ipyrad as ipp
import argparse

def create_file(**kwargs):
    ## create an Assembly object named name.
	data1 = ipp.Assembly(kwargs['assembly_name'])
    ## setting/modifying parameters for this Assembly object
	data1.set_params('project_dir', "./")
	data1.set_params('sorted_fastq_path', "./*.gz")
	for key, value in kwargs.items():
		if key != "assembly_name":
        ## setting/modifying parameters for this Assembly object
			data1.set_params(key, kwargs[key])

	data1.write_params()

def main():
	parser = argparse.ArgumentParser(description='Create a parameter file to use as input for ipyrad', argument_default=argparse.SUPPRESS)
	parser.add_argument('--assembly_name', help='Name to use as prefix for assembly directories', required=True)
	parser.add_argument('--assembly_method', help='Either denovo or reference methods for assembling RAD data', required=True)
	parser.add_argument('--datatype', help='Type of RAD data. One of: rad, ddrad, gbs, pairddrad, pairgbs, 2brad, pair3rad', required=True)
	parser.add_argument('--restriction_overhang', help='Restriction overhang (cut1,) or (cut1, cut2)', required=True)
	parser.add_argument('--reference_sequence', help='If using reference assembly, specify reference file')
	parser.add_argument('--max_low_qual_bases', help='Maximum number of low quality base calls in a read (Q<20)')
	parser.add_argument('--phred_Qscore_offset', help='Phred Q score offset')
	parser.add_argument('--mindepth_statistical', help='Minimum depth for statistical base calling')
	parser.add_argument('--mindepth_majrule', help='Minimum depth for majority rule base calling ')
	parser.add_argument('--maxdepth', help='Max cluster depth within samples')
	parser.add_argument('--clust_threshold', help='Clustering threshold for de novo assembly')
	parser.add_argument('--filter_adapters', help='Filter for adapters/primers (1 or 2=stricter)')
	parser.add_argument('--filter_min_trim_len', help='Min length of reads after adapter trim')
	parser.add_argument('--max_alleles_consens', help='Max alleles per site in consensus sequences')
	parser.add_argument('--max_Ns_consens', help='Max N (uncalled bases) in consensus')
	parser.add_argument('--max_Hs_consens', help='Max H (Heterozygotes) in consensus')
	parser.add_argument('--min_samples_locus', help='Min samples per locus for output')
	parser.add_argument('--max_SNPs_locus', help=' Max SNPs per locus')
	parser.add_argument('--max_Indels_locus', help=' Max indels per locus')
	parser.add_argument('--trim_reads', help=' Trim raw read edges (R1>, <R1, R2>, <R2)')
	parser.add_argument('--trim_loci', help=' Trim locus edges (see docs) (R1>, <R1, R2>, <R2)')
	parser.add_argument('--output_formats', help=' Output formats, please see ipyrad docs (*=all formats)')
	args = vars(parser.parse_args())
	create_file(**args)

main()
