process VEP {
    tag "$meta"
    label 'process_single'

    conda "${moduleDir}/environment.yml"
    container "${ workflow.containerEngine == 'singularity' && !task.ext.singularity_pull_docker_container ?
        'https://depot.galaxyproject.org/singularity/ensembl-vep:112.0--pl5321h2a3209d_0':
        'biocontainers/ensembl-vep:112.0--pl5321h2a3209d_0' }"

    input:
    tuple val(meta), path(sampmeta), path(vcf)
    path gff
    path reference

    output:
    path '*.html'                                   , emit: report
    path '*.txt'                                    , emit: warnings
    path "${meta}"                                  , emit: vep_out
    path "versions.yml"                             , emit: versions

    when:
    task.ext.when == null || task.ext.when

    script: // This script is bundled with the pipeline, in nf-core/hybrider/bin/
    def args = task.ext.args ?: ''
    def prefix = task.ext.prefix ?: "${meta}"

    """
    (grep ^"#" $gff; grep -v ^"#" $gff | sort -k1,1 -k4,4n) | bgzip > sorted.gff.gz
    tabix sorted.gff.gz
    vep -i $vcf -o $prefix --fasta $reference --gff sorted.gff.gz

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        ensemblvep: \$( echo \$(vep --help 2>&1) | sed 's/^.*Versions:.*ensembl-vep : //;s/ .*\$//')
    END_VERSIONS
    """
}
