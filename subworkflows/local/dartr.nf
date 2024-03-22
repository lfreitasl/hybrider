//
// Check input samplesheet and get read channels
//

include { FILTER_VCF } from '../../modules/local/vcf_dartr_filt/main'
include { META_VCF   } from '../../modules/local/meta_filt_vcf/main'

workflow FILT_CONVERTER {
    take:
    vcf // file: /path/to/samplesheet.csv
    meta
    locmiss
    indmiss
    maf
    usepopinfo

    main:
    ch_str = Channel.empty()
    ch_vcf = Channel.empty()
    ch_vcf_meta = Channel.empty()
    ch_versions = Channel.empty()

    FILTER_VCF(vcf,meta,locmiss,indmiss,maf,usepopinfo)
    ch_str = ch_str.mix(FILTER_VCF.out.str.ifEmpty([]))
    ch_vcf = ch_vcf.mix(FILTER_VCF.out.vcf.ifEmpty([]))
    ch_versions = ch_versions.mix(FILTER_VCF.out.versions.first().ifEmpty(null))

    META_VCF(ch_vcf.collect())
    ch_vcf_meta = ch_vcf_meta.mix(META_VCF.out.vcf_meta.ifEmpty([]))


    emit:
    str      = ch_str                                 // channel: [ val(meta), [ reads ] ]
    vcf      = ch_vcf
    vcf_meta = ch_vcf_meta
    versions = ch_versions // channel: [ versions.yml ]
}

// Function to get list of [ meta, [ fastq_1, fastq_2 ] ]
def create_fastq_channel(LinkedHashMap row) {
    // create meta map
    def meta = [:]
    meta.id         = row.sample
    meta.single_end = row.single_end.toBoolean()

    // add path(s) of the fastq file(s) to the meta map
    def fastq_meta = []
    if (!file(row.fastq_1).exists()) {
        exit 1, "ERROR: Please check input samplesheet -> Read 1 FastQ file does not exist!\n${row.fastq_1}"
    }
    if (meta.single_end) {
        fastq_meta = [ meta, [ file(row.fastq_1) ] ]
    } else {
        if (!file(row.fastq_2).exists()) {
            exit 1, "ERROR: Please check input samplesheet -> Read 2 FastQ file does not exist!\n${row.fastq_2}"
        }
        fastq_meta = [ meta, [ file(row.fastq_1), file(row.fastq_2) ] ]
    }
    return fastq_meta
}
