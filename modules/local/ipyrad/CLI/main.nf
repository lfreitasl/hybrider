process IPYRAD {
    tag '$bam'
    label 'process_high'

    conda "${moduleDir}/environment.yml"
    container "${ workflow.containerEngine == 'singularity' && !task.ext.singularity_pull_docker_container ?
        'https://depot.galaxyproject.org/singularity/ipyrad:0.9.95--pyh7cba7a3_0':
        'quay.io/biocontainers/ipyrad:0.9.95--pyh7cba7a3_0' }"

    input:
    path params_file

    output:
    path "*.bam", emit: bam
    path "versions.yml"           , emit: versions

    when:
    task.ext.when == null || task.ext.when

    script:
    def args   = task.ext.args ?: '-s 1234567'

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
    touch ${prefix}.bam

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        ipyrad: \$(ipyrad --version | sed 's/ipyrad //')
    END_VERSIONS
    """
}
