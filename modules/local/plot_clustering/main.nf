process PLOT_CLUSTERING {
    tag "$meta"
    label 'process_single'
    //errorStrategy 'ignore'

    conda "${moduleDir}/environment.yml"
    container "${ workflow.containerEngine == 'singularity' && !task.ext.singularity_pull_docker_container ?
        'https://depot.galaxyproject.org/singularity/mulled-v2-7cbdd89e4a230442dcff2455c5693faa81d0cda4:f36d68574b255379a8cabacc75a9ec9c783aff6e-0':
        'biocontainers/mulled-v2-7cbdd89e4a230442dcff2455c5693faa81d0cda4:f36d68574b255379a8cabacc75a9ec9c783aff6e-0' }"

    input:
    tuple val(meta), path(sampmeta), path(ffiles), path(qfiles), path(log)
    val plot_admix
    val plot_str
    val usepopinfo
    val writecsv

    output:
    tuple path('*.pdf'), path("*.svg")              , optional: true, emit: graphs
    tuple val(meta), path("meta_str_admix_K2.csv")  , optional: true, emit: meta
    path "versions.yml"                             , emit: versions

    when:
    task.ext.when == null || task.ext.when

    script: // This script is bundled with the pipeline, in nf-core/hybrider/bin/
    def args = task.ext.args ?: ''

    """
    clustering_plot.R \\
        $plot_admix \\
        $plot_str \\
        $sampmeta \\
        $usepopinfo \\
        $writecsv

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        r-base: \$(echo \$(R --version 2>&1) | sed 's/^.*R version //; s/ .*\$//')
        r-reshape2: \$(Rscript -e "library(reshape2); cat(as.character(packageVersion('reshape2')))")
        r-pophelper: \$(Rscript -e "library(pophelper); cat(as.character(packageVersion('pophelper')))")
    END_VERSIONS
    """
}