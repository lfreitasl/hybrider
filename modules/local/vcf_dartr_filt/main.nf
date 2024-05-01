process FILTER_VCF {
    tag "$vcf.baseName"
    label 'process_single'
    errorStrategy 'ignore'

    conda "${moduleDir}/environment.yml"
    container "${ workflow.containerEngine == 'singularity' && !task.ext.singularity_pull_docker_container ?
        'docker://lfreitasl/dartr:latest':
        'docker.io/lfreitasl/dartr:latest' }"

    input:
    path vcf
    path meta
    val locmiss
    val indmiss
    val maf
    val usepopinfo

    output:
    path '*.str'                                                , emit: str
    path '*.treemix.gz'                         , optional: true, emit: treemix
    tuple val("$vcf.baseName"), path("*sorted_meta.csv"), path('filt_*.vcf')  , emit: vcf
    tuple val("$vcf.baseName"), path("*sorted_meta.csv"), path('*.ped')       , emit: ped
    tuple val("$vcf.baseName"), path("*sorted_meta.csv"), path('*.map')       , emit: map
    tuple val("$vcf.baseName"), path("*sorted_meta.csv"), path('*.bed')       , emit: bed
    tuple val("$vcf.baseName"), path("*sorted_meta.csv"), path('*.bim')       , emit: bim
    tuple val("$vcf.baseName"), path("*sorted_meta.csv"), path('*.fam')       , emit: fam
    path "versions.yml"                             , emit: versions

    when:
    task.ext.when == null || task.ext.when

    script: // This script is bundled with the pipeline, in nf-core/hybrider/bin/
    def args = task.ext.args ?: ''
    def prefix = task.ext.prefix ?: "${vcf.baseName}"

    """
    dartr_filters_format_conversion.R \\
        $vcf \\
        $meta \\
        $locmiss \\
        $indmiss \\
        $maf \\
        $prefix \\
        $usepopinfo \\
        $args

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        r-base: \$(echo \$(R --version 2>&1) | sed 's/^.*R version //; s/ .*\$//')
        r-dartr: \$(Rscript -e "library(dartR); cat(as.character(packageVersion('dartR')))")
        r-vcfr: \$(Rscript -e "library(vcfR); cat(as.character(packageVersion('vcfR')))")
    END_VERSIONS
    """
}
