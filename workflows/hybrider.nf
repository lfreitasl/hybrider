/*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    IMPORT MODULES / SUBWORKFLOWS / FUNCTIONS
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*/

include { FASTQC                 } from '../modules/nf-core/fastqc/main'
include { MULTIQC                } from '../modules/nf-core/multiqc/main'
include { ADMIXTURE              } from '../modules/local/admixture/main'
include { PREPARE_ML             } from '../modules/local/prepare_ml/main'
include { SNP_SELECTOR           } from '../modules/local/snp_selector/main'
include { VEP                    } from '../modules/local/vep/main'
include { paramsSummaryMap       } from 'plugin/nf-validation'
include { paramsSummaryMultiqc   } from '../subworkflows/nf-core/utils_nfcore_pipeline'
include { softwareVersionsToYAML } from '../subworkflows/nf-core/utils_nfcore_pipeline'
include { methodsDescriptionText } from '../subworkflows/local/utils_lfreitasl_hybrider_pipeline'
include { FILT_CONVERTER         } from '../subworkflows/local/conversions'
include { RUN_STRUCTURE          } from '../subworkflows/local/structure.nf'
include { PLOT_SELECTED          } from '../subworkflows/local/plot_admix_str'
//include { QC                     } from '../subworkflows/local/QC'

/*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    RUN MAIN WORKFLOW
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*/


workflow HYBRIDER {

    take:
    ch_vcfs // channel: samplesheet read in from --input
    //ch_reads

    main:
    ch_versions      = Channel.empty()
    ch_multiqc_files = Channel.empty()

    // if (params.upstream){
    // QC(
    //     ch_reads
    // )
    // ch_versions = ch_versions.mix(QC.out.versions.first())
    // }

    // if (params.downstream){
    ch_str_in        = Channel.empty()
    ch_admix_in      = Channel.empty()
    ch_kvalue        = Channel.empty()
    ch_vcf           = Channel.empty()
    ch_k2            = Channel.empty()
    ch_reference     = Channel.empty()
    ch_gff           = Channel.empty()

    ch_reference     = ch_reference.mix(Channel.fromPath(params.reference))
    ch_gff           = ch_gff.mix(Channel.fromPath(params.gff))

    //
    // MODULE: Run FastQC
    //
    FILT_CONVERTER(
        ch_vcfs,
        params.meta,
        params.locmiss,
        params.indmiss,
        params.maf,
        params.popinfo,
    )

    ch_str_in   = ch_str_in.mix(FILT_CONVERTER.out.str_meta)
    ch_admix_in = ch_admix_in.mix(FILT_CONVERTER.out.admix)
    ch_vcf      = ch_vcf.mix(FILT_CONVERTER.out.vcf)
    ch_versions = ch_versions.mix(FILT_CONVERTER.out.versions.first())

    if (!params.skip_str){

    RUN_STRUCTURE(
        ch_str_in,
        params.noadmix,
        params.freqscorr,
        params.inferalpha,
        params.alpha,
        params.inferlambda,
        params.lambda,
        params.ploidy,
        params.burnin,
        params.mcmc,
        params.rep_per_k,
        params.k_value
    )

    ch_ffiles = RUN_STRUCTURE.out.ffiles.groupTuple().map{meta,sampmeta,ffiles->return [meta,sampmeta[0][1],ffiles[0]]}
    ch_versions = ch_versions.mix(RUN_STRUCTURE.out.versions.first())
    }

    if (!params.skip_admix){
    ch_kvalue   = ch_kvalue.mix(Channel.from(1..params.k_value))
    ch_admix_in = ch_admix_in.combine(ch_kvalue)

    ADMIXTURE(ch_admix_in)

    ch_versions = ch_versions.mix(ADMIXTURE.out.versions.first())
    ch_admix_out_Q=ADMIXTURE.out.ancestry_fractions.groupTuple()
    ch_admix_out_log=ADMIXTURE.out.cross_validation.groupTuple().map{meta,sampmeta,file -> return [meta, file]}

    ch_admix_final=ch_admix_out_Q.combine(ch_admix_out_log, by:0).map{meta,sampmeta,q,log->return [meta,q,log]}
    }

    if (!params.skip_str && !params.skip_admix){
        ch_plotq_in=ch_ffiles.combine(ch_admix_final,by:0)
    }
    if (!params.skip_str && params.skip_admix){
        ch_plotq_in=ch_ffiles.map{meta,sampmeta,ffiles -> return [meta,sampmeta,ffiles,[],[]]}
    }
    if (params.skip_str && !params.skip_admix) {
        ch_plotq_in=ch_admix_out_Q.combine(ch_admix_out_log, by:0).map{meta,sampmeta,q,log->return [meta,sampmeta[0],[],q,log]}
    }
    
    if (!params.skip_plot){
    PLOT_SELECTED(ch_plotq_in, params.writecsv, params.plot_str, params.plot_admix)

    ch_k2       = ch_k2.mix(PLOT_SELECTED.out.meta)
    ch_versions = ch_versions.mix(PLOT_SELECTED.out.versions.first())
    }

    if (!params.skip_ml){
    PREPARE_ML(
        ch_vcf,
        ch_k2,
        params.inferbypop,
        params.whichpop,
        params.inferbyclust,
        params.smaller,
        params.method,
        params.upper,
        params.lower,
        params.rminvariable,
        params.dropna
    )

    ch_versions = ch_versions.mix(PREPARE_ML.out.versions.first())

    SNP_SELECTOR(
        PREPARE_ML.out.genotype,
        PREPARE_ML.out.sampmeta,
        PREPARE_ML.out.snpmeta,
        params.kfold,
        params.pvalue,
        params.corr,
        params.nsnps
    )
    }

    if (!params.skip_vep){
    VEP(
        ch_vcf,
        ch_gff,
        ch_reference
    )
    }

// }

    //
    // Collate and save software versions
    //
    softwareVersionsToYAML(ch_versions)
        .collectFile(storeDir: "${params.outdir}/pipeline_info", name: 'lfreitasl_hybrider_software_mqc_versions.yml', sort: true, newLine: true)
        .set { ch_collated_versions }

    //
    // MODULE: MultiQC
    //
    ch_multiqc_config                     = Channel.fromPath("$projectDir/assets/multiqc_config.yml", checkIfExists: true)
    ch_multiqc_custom_config              = params.multiqc_config ? Channel.fromPath(params.multiqc_config, checkIfExists: true) : Channel.empty()
    ch_multiqc_logo                       = params.multiqc_logo ? Channel.fromPath(params.multiqc_logo, checkIfExists: true) : Channel.empty()
    summary_params                        = paramsSummaryMap(workflow, parameters_schema: "nextflow_schema.json")
    ch_workflow_summary                   = Channel.value(paramsSummaryMultiqc(summary_params))
    ch_multiqc_custom_methods_description = params.multiqc_methods_description ? file(params.multiqc_methods_description, checkIfExists: true) : file("$projectDir/assets/methods_description_template.yml", checkIfExists: true)
    ch_methods_description                = Channel.value(methodsDescriptionText(ch_multiqc_custom_methods_description))
    ch_multiqc_files                      = ch_multiqc_files.mix(ch_workflow_summary.collectFile(name: 'workflow_summary_mqc.yaml'))
    ch_multiqc_files                      = ch_multiqc_files.mix(ch_collated_versions)
    ch_multiqc_files                      = ch_multiqc_files.mix(ch_methods_description.collectFile(name: 'methods_description_mqc.yaml', sort: false))

    MULTIQC (
        ch_multiqc_files.collect(),
        ch_multiqc_config.toList(),
        ch_multiqc_custom_config.toList(),
        ch_multiqc_logo.toList()
    )

    emit:
    multiqc_report = MULTIQC.out.report.toList() // channel: /path/to/multiqc_report.html
    versions       = ch_versions                 // channel: [ path(versions.yml) ]
}

/*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    THE END
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
*/
