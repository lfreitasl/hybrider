process STRUCTURE {
    tag "Running_Structure"
    label 'process_single'

    conda "${moduleDir}/environment.yml"
    container "${ workflow.containerEngine == 'singularity' && !task.ext.singularity_pull_docker_container ?
        'docker://lfreitasl/structure-threader:latest':
        'docker.io/lfreitasl/structure-threader:latest' }"

    input:
    path str
    val noadmix 
    val freqscorr
    val inferalpha
    val alpha
    val inferlambda
    val lambda
    val numinds
    val numloci
    val ploidy
    val maxpops
    val burnin
    val mcmc
    val rep_per_k


    output:
    path '*_f'         , optional: true, emit: ffiles
    path '*_q'         , optional: true, emit: qfiles
    path "versions.yml", emit: versions

    when:
    task.ext.when == null || task.ext.when

    script: // This script is bundled with the pipeline, in nf-core/hybrider/bin/
    def args = task.ext.args ?: ''
    def prefix = task.ext.prefix ?: "${str.baseName}"
    def ifnoadmix = noadmix : '1' ? '0'
    def iffreqscorr = freqscorr : '1' ? '0'
    def ifinferalpha = inferalpha : '1' ? '0'
    def ifinferlambda = inferlambda : '1' ? '0'

    """
    sed -i 's/#define NOADMIX[[:space:]]*[0-9]/#define NOADMIX $ifnoadmix/' /extraparams
    sed -i 's/#define FREQSCORR[[:space:]]*[0-9]/#define FREQSCORR $iffreqscorr/' /extraparams
    sed -i 's/#define INFERALPHA[[:space:]]*[0-9]/#define INFERALPHA $ifinferalpha/' /extraparams
    sed -E -i 's/#define ALPHA[[:space:]]*[0-9]+\.[0-9]+/#define ALPHA $alpha/' /extraparams
    sed -i 's/#define INFERLAMBDA[[:space:]]*[0-9]/#define INFERLAMBDA $ifinferlambda/' /extraparams
    sed -E -i 's/#define LAMBDA[[:space:]]*[0-9]+\.[0-9]+/#define LAMBDA $lambda/' /extraparams

    sed -i 's/#define BURNIN[[:space:]]*[0-9]*\(\.[0-9]\+\)\{0,1\}/#define BURNIN $burnin/' /mainparams
    sed -i 's/#define NUMREPS[[:space:]]*[0-9]*\(\.[0-9]\+\)\{0,1\}/#define NUMREPS $mcmc/' /mainparams
    sed -i 's/#define NUMINDS[[:space:]]*[0-9]*\(\.[0-9]\+\)\{0,1\}/#define NUMINDS $numinds/' /mainparams
    sed -i 's/#define NUMLOCI[[:space:]]*[0-9]*\(\.[0-9]\+\)\{0,1\}/#define NUMLOCI $numloci/' /mainparams



    structure_threader run \\
        -Klist \\
        -R $rep_per_k \\
        -i $str\\
        -o ./ \\
        --params /mainparams \\
        -t $task.cpus \\
        -st /bin/structure \\

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        structure: \$(echo \$(structure --version 2>&1) | grep 'Version' | grep -o '[0-9]\+\.[0-9]\+\.[0-9]\+'
    END_VERSIONS
    """
}