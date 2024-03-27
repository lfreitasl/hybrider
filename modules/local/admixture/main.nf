process ADMIXTURE {
    tag "$meta"
    label 'process_medium'

    conda "${moduleDir}/environment.yml"
    container "${ workflow.containerEngine == 'singularity' && !task.ext.singularity_pull_docker_container ?
        'docker://lfreitasl/admixture:latest':
        'docker.io/lfreitasl/admixture' }"

    input:
    tuple val(meta), path (bed_ped_geno), path(bim_map), path(fam)
    val K


    output:
    tuple val(meta), path("*.Q")        , emit: ancestry_fractions
    tuple val(meta), path("*.P")        , emit: allele_frequencies
    tuple val(meta), path("log*.out")   , optional: true, emit: cross_validation
    path "versions.yml"                 , emit: versions

    when:
    task.ext.when == null || task.ext.when

    script:
    def args = task.ext.args ?: ''
    def prefix = task.ext.prefix ?: "${meta}"
    """
    admixture32 \\
        $bed_ped_geno \\
        $K \\
        -j$task.cpus \\
        $args

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        admixture: \$(echo \$(admixture 2>&1) | head -n 1 | grep -o "ADMIXTURE Version [0-9.]*" | sed 's/ADMIXTURE Version //' )
    END_VERSIONS
    """

    stub:
    def args = task.ext.args ?: ''
    def prefix = task.ext.prefix ?: "${meta}"
    """
    touch "${prefix}.Q"
    touch "${prefix}.P"

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        admixture: \$(echo \$(admixture 2>&1) | head -n 1 | grep -o "ADMIXTURE Version [0-9.]*" | sed 's/ADMIXTURE Version //' )
    END_VERSIONS
    """
}
