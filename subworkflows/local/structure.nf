//
// Check input samplesheet and get read channels
//

include { STRUCTURE } from '../../modules/local/structure/main.nf'

workflow RUN_STRUCTURE {
    take:
    str // metamap: output from dartR subworkflow (str_meta)
    noadmix 
    freqscorr
    inferalpha
    alpha
    inferlambda
    lambda
    ploidy
    burnin
    mcmc
    rep_per_k
    k_value

    main:
    ch_str = Channel.empty()
    ch_kvalue = Channel.empty()
    ch_rep_per_k = Channel.empty()
    ch_versions = Channel.empty()
    ch_str_in = Channel.empty()
    ch_ffiles = Channel.empty()
    ch_qfiles = Channel.empty()
    ch_versions = Channel.empty()
    
    ch_str = ch_str.mix(str)
    ch_kvalue = ch_kvalue.mix(Channel.from(1..k_value))
    ch_rep_per_k = ch_rep_per_k.mix(Channel.from(1..rep_per_k))
    ch_str_in = ch_str_in.mix(ch_str.combine(ch_kvalue).combine(ch_rep_per_k))

    STRUCTURE(
        ch_str_in,
        noadmix,
        freqscorr,
        inferalpha,
        alpha,
        inferlambda,
        lambda,
        ploidy,
        burnin,
        mcmc
    )

    ch_ffiles = ch_ffiles.mix(STRUCTURE.out.ffiles.groupTuple().ifEmpty([]))
    ch_qfiles = ch_qfiles.mix(STRUCTURE.out.qfiles.groupTuple()ifEmpty([]))
    ch_versions = ch_versions.mix(STRUCTURE.out.versions.first().ifEmpty(null))

  

    emit:
    ffiles      = ch_ffiles                                 // channel: [ val(meta), [ reads ] ]
    qfiles      = ch_qfiles
    versions    = ch_versions // channel: [ versions.yml ]
}