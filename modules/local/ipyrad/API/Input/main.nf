process IPYRAD_INPUT {
    tag "$prefix"
    label 'process_single'

    conda "${moduleDir}/environment.yml"
    container "${ workflow.containerEngine == 'singularity' && !task.ext.singularity_pull_docker_container ?
        'https://depot.galaxyproject.org/singularity/ipyrad:0.9.95--pyh7cba7a3_0':
        'quay.io/biocontainers/ipyrad:0.9.95--pyh7cba7a3_0' }"

    input:
    path reads //flat channel
    path reference
    val datatype
    val method
    val overhang

    output:
    path "params-${prefix}.txt", emit: params
    path reads                 , emit: reads
    path reference             , optional: true, emit: reference
    path "versions.yml"        , emit: versions

    when:
    task.ext.when == null || task.ext.when

    script:
    def args   = task.ext.args ?: ''
    def prefix = task.ext.prefix ?: 'assembly'
    def ref    = reference ? "--reference_sequence $reference" : ''

    """
    ipyrad_input.py \\
        $ref \\
        $args \\
        --assembly_name $prefix \\
        --assembly_method $method \\
        --datatype $datatype \\
        --restriction_overhang $overhang


    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        ipyrad: \$(ipyrad --version | sed 's/ipyrad //')
    END_VERSIONS
    """

    stub:
    def args   = task.ext.args ?: reference ? "--reference_sequence ${reference}" : ''
    def prefix = tesk.ext.prefix ?: 'assembly'


    """
    touch params-${prefix}.txt

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        ipyrad: \$(ipyrad --version | sed 's/ipyrad //')
    END_VERSIONS
    """
}
