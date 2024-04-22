//
// Check input samplesheet and get read channels
//

include { FASTQC as RAW_FASTQC  } from '../../modules/nf-core/fastqc/main'
include { FASTQC as FILT_FASTQC } from '../../modules/nf-core/fastqc/main'
include { FASTP                 } from '../../modules/nf-core/fastp/main'
include { IPYRAD_INPUT          } from '../../modules/local/ipyrad/API/Input/main'
include { IPYRAD as QC_STEPS    } from '../../modules/local/ipyrad/CLI/main'

workflow QC {
    take:
    reads //collected channel

    main:
    ch_reads          = reads
    ch_assembly_file  = Channel.empty()
    ch_params_file    = Channel.empty()
    ch_edits          = Channel.empty()
    ch_multiqc_raw    = Channel.empty()
    ch_multiqc_filt   = Channel.empty()
    ch_multiqc_all    = Channel.empty()
    ch_versions       = Channel.empty()

    // Running fastqc on raw reads
    RAW_FASTQC(ch_reads)

    ch_versions     = ch_versions.mix(RAW_FASTQC.out.versions.first().ifEmpty(null))
    ch_multiqc_raw  = ch_multiqc_raw.mix(RAW_FASTQC.out.zip.collect{it[1]}.ifEmpty([]))
    ch_multiqc_all = ch_multiqc_all.mix(ch_multiqc_raw.ifEmpty([]))

    // Running fastp to deduplicate reads (in config args)

    FASTP(
        ch_reads,
        [],
        false,
        false
    )

    ch_versions = ch_versions.mix(FASTP.out.versions.first().ifEmpty(null))

    // Generating input file for ipyrad QC steps (s-12)
    IPYRAD_INPUT(
        FASTP.out.reads.collect{it[1]}.ifEmpty([]),
        params.reference,
        params.datatype,
        params.assembly_method,
        params.overhang
    )

    ch_versions     = ch_versions.mix(IPYRAD_INPUT.out.versions.first().ifEmpty(null))
    ch_params_file  = ch_params_file.mix(IPYRAD_INPUT.out.params)

    // Running ipyrad QC steps
    QC_STEPS(
        ch_params_file,
        [],
        IPYRAD_INPUT.out.reads,
        IPYRAD_INPUT.out.reference.ifEmpty([]),
        [],
        [],
        []
    )

    ch_assembly_file = ch_assembly_file.mix(QC_STEPS.out.assembly_file)
    ch_edits         = ch_edits.mix(QC_STEPS.out.trimmed_dir)

    // Running fastqc on trimmed reads
    FILT_FASTQC(QC_STEPS.out.trimmed_reads) // THIS IS WITH NO META-MAP, RESOLVE

    ch_multiqc_filt = ch_multiqc_filt.mix(FILT_FASTQC.out.zip.collect{it[1]}.ifEmpty([]))
    ch_multiqc_all = ch_multiqc_all.mix(ch_multiqc_filt.ifEmpty([]))


    emit:
    assembly_file   = ch_assembly_file
    params_file     = ch_params_file
    edits           = ch_edits
    multiqc         = ch_multiqc_all                                 // channel: [ val(meta), [ reads ] ]
    versions        = ch_versions // channel: [ versions.yml ]
}
