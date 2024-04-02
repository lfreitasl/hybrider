process META_VCF {
    tag "$meta"
    label 'process_single'
    errorStrategy 'ignore'

    conda "${moduleDir}/environment.yml"
    container "${ workflow.containerEngine == 'singularity' && !task.ext.singularity_pull_docker_container ?
        'docker://lfreitasl/dartr:latest':
        'docker.io/lfreitasl/dartr:latest' }"

    input:
    tuple val(meta), path(sampmeta), path(vcf)
    path  str

    output:
    tuple path('vcfs_info.csv'), path(sampmeta), path(str), emit: vcf_meta
    path "versions.yml"                                   , emit: versions

    when:
    task.ext.when == null || task.ext.when

    script: // This script is bundled with the pipeline, in nf-core/hybrider/bin/
    def args = task.ext.args ?: ''

    """
    report_vcfs_meta.R

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        r-base: \$(echo \$(R --version 2>&1) | sed 's/^.*R version //; s/ .*\$//')
        r-dartr: \$(Rscript -e "library(dartR); cat(as.character(packageVersion('dartR')))")
        r-vcfr: \$(Rscript -e "library(vcfR); cat(as.character(packageVersion('vcfR')))")
    END_VERSIONS
    """
}