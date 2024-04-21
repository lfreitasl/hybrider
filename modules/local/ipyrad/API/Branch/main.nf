process IPYRAD_BRANCH_MINDEPTH {
    tag "$prefix"
    label 'process_single'

    conda "${moduleDir}/environment.yml"
    container "${ workflow.containerEngine == 'singularity' && !task.ext.singularity_pull_docker_container ?
        'https://depot.galaxyproject.org/singularity/ipyrad:0.9.95--pyh7cba7a3_0':
        'quay.io/biocontainers/ipyrad:0.9.95--pyh7cba7a3_0' }"

    input:
    tuple val(mindepth), path(assembly)
    path edits
    path reference
    

    output:
    path "*.json"              , emit: assembly_object
    path "params-${prefix}.txt", emit: newparams
    path "versions.yml"        , emit: versions

    when:
    task.ext.when == null || task.ext.when

    script:
    def args   = task.ext.args ?: reference ? "--reference_sequence $reference" : ''
    def prefix = tesk.ext.prefix ?: "${assembly.baseName}_${mindepth}d"

    """
    ipyrad_branching.py \\
        --assembly_name $assembly.baseName \\
        --new_name $prefix \\
        --mindepth_statistical $mindepth \\
        --mindepth_majrule $mindepth \\
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
