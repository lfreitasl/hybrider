process STRUCTURE {
    tag "${meta.id}_K${k_value}_R${rep_per_k}"
    label 'process_high'

    conda "${moduleDir}/environment.yml"
    container "${ workflow.containerEngine == 'singularity' && !task.ext.singularity_pull_docker_container ?
        'docker://lfreitasl/structure-threader:latest':
        'docker.io/lfreitasl/structure-threader:latest' }"

    input:
    tuple val(meta), path(str), val(k_value), val(rep_per_k)
    val noadmix 
    val freqscorr
    val inferalpha
    val alpha
    val inferlambda
    val lambda
    val ploidy
    val burnin
    val mcmc


    output:
    tuple val(meta), path('*rep*_f')         , emit: ffiles
    tuple val(meta), path('*rep*_q')         , emit: qfiles
    path "versions.yml"                  , emit: versions

    when:
    task.ext.when == null || task.ext.when

    script: // This script is bundled with the pipeline, in nf-core/hybrider/bin/
    def args = task.ext.args ?: ''
    def prefix = task.ext.prefix ?: "${meta.id}"
    def ifnoadmix = noadmix ? '1' : '0'
    def iffreqscorr = freqscorr ? '1' : '0'
    def ifinferalpha = inferalpha ? '1' : '0'
    def ifinferlambda = inferlambda ? '1' : '0'
    def rename_f = rep_per_k > 1 ? "mv *rep1_f ${str.baseName}_str_K${k_value}_rep${rep_per_k}_f" : ''
    def rename_q = rep_per_k > 1 ? "mv *rep1_q ${str.baseName}_str_K${k_value}_rep${rep_per_k}_q" : '' 

    """
    cat /tmp/extraparams > ./extraparams
    cat /tmp/mainparams > ./mainparams
    sed -i 's/#define NOADMIX[[:space:]]*[0-9]/#define NOADMIX ${ifnoadmix} /' ./extraparams
    sed -i 's/#define FREQSCORR[[:space:]]*[0-9]/#define FREQSCORR ${iffreqscorr} /' ./extraparams
    sed -i 's/#define INFERALPHA[[:space:]]*[0-9]/#define INFERALPHA ${ifinferalpha} /' ./extraparams
    sed -E -i 's/#define ALPHA[[:space:]]*[0-9]+\\.[0-9]+/#define ALPHA ${alpha} /' ./extraparams
    sed -i 's/#define INFERLAMBDA[[:space:]]*[0-9]/#define INFERLAMBDA ${ifinferlambda}/' ./extraparams 
    sed -E -i 's/#define LAMBDA[[:space:]]*[0-9]+\\.[0-9]+/#define LAMBDA ${lambda} /' ./extraparams
    sed -i 's/#define BURNIN[[:space:]]*[0-9]*\\(\\.[0-9]\\+\\)\\{0,1\\}/#define BURNIN ${burnin} /' ./mainparams
    sed -i 's/#define NUMREPS[[:space:]]*[0-9]*\\(\\.[0-9]\\+\\)\\{0,1\\}/#define NUMREPS ${mcmc} /' ./mainparams
    sed -i 's/#define NUMINDS[[:space:]]*[0-9]*\\(\\.[0-9]\\+\\)\\{0,1\\}/#define NUMINDS ${meta.n_inds} /' ./mainparams
    sed -i 's/#define NUMLOCI[[:space:]]*[0-9]*\\(\\.[0-9]\\+\\)\\{0,1\\}/#define NUMLOCI ${meta.n_loc} /' ./mainparams
    sed -i 's/#define PLOIDY[[:space:]]*[0-9]*\\(\\.[0-9]\\+\\)\\{0,1\\}/#define PLOIDY ${ploidy} /' ./mainparams

    structure_threader run \\
        -Klist $k_value \\
        -R 1 \\
        -i $str\\
        -o ./ \\
        --params ./mainparams \\
        -t $task.cpus \\
        -st /bin/structure \\
        --no_tests true \\
        --no_plots true

    $rename_f
    $rename_q 

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        structure: \$(echo \$(structure --version 2>&1 | grep 'Version' | grep -o '[0-9]\\+\\.[0-9]\\+\\.[0-9]\\+'))
    END_VERSIONS
    """
}