process SNP_SELECTOR {
    //tag "$vcf.baseName" //TODO
    label 'process_high'

    conda "${moduleDir}/environment.yml"
    container "${ workflow.containerEngine == 'singularity' && !task.ext.singularity_pull_docker_container ?
        'docker://lfreitasl/lalgorithms:latest':
        'docker.io/lfreitasl/lalgorithms:latest' }"

    input:
    path gen
    path meta
    path snpinfo
    val kfold
    val pvalue
    val corr
    val nsnps

    output:
    path '*.svg'                                                , emit: svg
    path '*.jpg'                                                , emit: jpg
    path 'SNP_selector*.txt'                                    , emit: report
    path "versions.yml"                                         , emit: versions

    when:
    task.ext.when == null || task.ext.when

    script: // This script is bundled with the pipeline, in nf-core/hybrider/bin/
    def args = task.ext.args ?: ''
    def prefix = task.ext.prefix ?: ''

    """
    snp_selector.py \\
        --genotype $gen \\
        --metadata $meta \\
        --snpinfo $snpinfo \\
        --k_fold $kfold \\
        --p_value $pvalue \\
        --corr $corr \\
        --n_snps $nsnps

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        python: \$(echo \$(python --version 2>&1) | sed 's/Python //; s/ .*\$//')
    END_VERSIONS
    """
}
