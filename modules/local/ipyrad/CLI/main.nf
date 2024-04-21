process IPYRAD {
    tag '$bam'
    label 'process_high'

    conda "${moduleDir}/environment.yml"
    container "${ workflow.containerEngine == 'singularity' && !task.ext.singularity_pull_docker_container ?
        'https://depot.galaxyproject.org/singularity/ipyrad:0.9.95--pyh7cba7a3_0':
        'quay.io/biocontainers/ipyrad:0.9.95--pyh7cba7a3_0' }"

    input:
    tuple path(params_file), path(reads), path(reference)

    output:
    path "*.json"               , emit: assembly_file
    path "params-*.txt"         , emit: parameter_file
    path "${prefix}_edits"      , optional: true, emit: trimmed_dir
    path "${prefix}_clust*"     , optional: true, emit: clustered_reads
    path "${prefix}_outfiles"   , optional: true, emit: outfiles_dir
    path "versions.yml"         , emit: versions

    when:
    task.ext.when == null || task.ext.when

    script:
    def args   = task.ext.args ?: '-s 1234567'
    def prefix = task.ext.args ?: 'assembly'

    """
    ipyrad \\
        -p $params_file \\
        -c $task.cpus \\
        $args \\

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        ipyrad: \$(ipyrad --version | sed 's/ipyrad //')
    END_VERSIONS
    """

    stub:
    def args = task.ext.args ?: '-s 1234567'
    
   
    """
    touch ${prefix}.json

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        ipyrad: \$(ipyrad --version | sed 's/ipyrad //')
    END_VERSIONS
    """
}
