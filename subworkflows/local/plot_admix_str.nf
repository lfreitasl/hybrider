
include { PLOT_CLUSTERING as PLOT_STR       } from '../../modules/local/plot_clustering/main'
include { PLOT_CLUSTERING as PLOT_ADMIX     } from '../../modules/local/plot_clustering/main'
include { PLOT_CLUSTERING as EXPORT_META    } from '../../modules/local/plot_clustering/main'



workflow PLOT_SELECTED {
    take:
    files // files: A tuple with files from STRUCTURE and ADMIXTURE, associated sample metadata and log files from admixture run
    exportmeta
    plot_str
    plot_admix

    main:
    ch_meta_K      = Channel.empty()
    ch_versions    = Channel.empty()
    
    if (exportmeta){
        EXPORT_META(files,false, false, false, true)
        ch_meta_K   = ch_meta_K.mix(EXPORT_META.out.meta)
        ch_versions = ch_versions.mix(EXPORT_META.out.versions)
    }

    if (plot_str){
        PLOT_STR(files, false, true, params.popinfo, false)
        ch_versions = ch_versions.mix(PLOT_STR.out.versions)
    }

    if (plot_admix){
        PLOT_ADMIX(files, true, false, params.popinfo, false)
        ch_versions = ch_versions.mix(PLOT_ADMIX.out.versions)
    }


    emit:
    meta      = ch_meta_K
    versions  = ch_versions // channel: [ versions.yml ]
}
