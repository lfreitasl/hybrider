//
// Check input samplesheet and get read channels
//

include { FILTER_VCF  } from '../../modules/local/vcf_dartr_filt/main'
include { META_VCF    } from '../../modules/local/meta_filt_vcf/main'

workflow FILT_CONVERTER {
    take:
    vcf // file: /path/to/samplesheet.csv
    meta
    locmiss
    indmiss
    maf
    usepopinfo

    main:
    ch_str      = Channel.empty()
    ch_vcf      = Channel.empty()
    ch_vcf_meta = Channel.empty()
    ch_versions = Channel.empty()
    ch_ped      = Channel.empty()
    ch_map      = Channel.empty()
    ch_bed      = Channel.empty()
    ch_bim      = Channel.empty()
    ch_fam      = Channel.empty()
    ch_admx     = Channel.empty()

    FILTER_VCF(vcf,meta,locmiss,indmiss,maf,usepopinfo)

    ch_str      = ch_str.mix(FILTER_VCF.out.str.ifEmpty([]))
    ch_vcf      = ch_vcf.mix(FILTER_VCF.out.vcf.ifEmpty([]))
    ch_bed      = ch_bed.mix(FILTER_VCF.out.bed.ifEmpty([]))
    ch_bim      = ch_bim.mix(FILTER_VCF.out.bim.ifEmpty([]))
    ch_fam      = ch_fam.mix(FILTER_VCF.out.fam.ifEmpty([]))
    ch_admx     = ch_admx.mix(ch_bed.combine(ch_bim, by:0).combine(ch_fam, by:0))
    ch_versions = ch_versions.mix(FILTER_VCF.out.versions.first().ifEmpty(null))

    META_VCF(ch_vcf, ch_str)
    
    ch_vcf_meta = ch_vcf_meta.mix(META_VCF.out.vcf_meta.ifEmpty([]))

    ch_vcf_meta
    .map { meta, vcfs ->
        return [ meta.splitCsv( header:true, sep:',' ), vcfs ]
     }
    .map { meta, vcfs ->
        return [create_csv_channel(meta[0]), vcfs] 
    }
    .set { ch_vcf_meta }

    

    emit:
    str      = ch_str                                 // channel: [ val(meta), [ reads ] ]
    vcf      = ch_vcf
    str_meta = ch_vcf_meta
    admix    = ch_admx
    versions = ch_versions // channel: [ versions.yml ]
}

// Function to get list of [ meta, [ fastq_1, fastq_2 ] ]
def create_csv_channel(LinkedHashMap row) {
    // create meta map
    def meta = [:]
    meta.id     = row.filenames
    meta.n_inds = row.n_inds
    meta.n_loc  = row.n_locs

    // add path(s) of the fastq file(s) to the meta map
    def csv_meta = meta 
    return csv_meta
}
