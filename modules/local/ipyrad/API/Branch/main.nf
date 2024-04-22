process IPYRAD_BRANCH {
    tag "$prefix"
    label 'process_single'

    conda "${moduleDir}/environment.yml"
    container "${ workflow.containerEngine == 'singularity' && !task.ext.singularity_pull_docker_container ?
        'https://depot.galaxyproject.org/singularity/ipyrad:0.9.95--pyh7cba7a3_0':
        'quay.io/biocontainers/ipyrad:0.9.95--pyh7cba7a3_0' }"

    input:
    path assembly
    each mindepth
    each minsamples
    path edits
    path reference


    output:
    path "${prefix}.json"              , emit: assembly_object
    path "params-${prefix}.txt", emit: newparams
    path "versions.yml"        , emit: versions

    when:
    task.ext.when == null || task.ext.when

    script:
    def args   = task.ext.args ?: ''
    def prefix = task.ext.prefix ?: mindepth && minsamples ? "${assembly.baseName}_${mindepth}d_${minsamples}samp" :
                 mindepth && !minsamples ? "${assembly.baseName}_${mindepth}d" :
                 !mindepth && minsamples ? "${assembly.baseName}_${minsamples}samp" : ''
    def ref    = reference ? "--reference_sequence $reference" : ''

    """
    ipyrad_branching.py \\
        --assembly_name $assembly.baseName \\
        --new_name $prefix \\
        --mindepth_statistical $mindepth \\
        --mindepth_majrule $mindepth \\
        $ref \\
        $args

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        ipyrad: \$(ipyrad --version | sed 's/ipyrad //')
    END_VERSIONS
    """

    stub:
    def args = task.ext.args ?: '-s 1234567'


    """
    touch ${prefix}.bam

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        ipyrad: \$(ipyrad --version | sed 's/ipyrad //')
    END_VERSIONS
    """
}
