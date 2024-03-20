params.input="/data/home/lucas.freitas/nextflow_modules/nf-core-hybrider/modules/local/test/test.csv"
// Substituir input pela saída do processo com script em R, esse script tem que gerar 
// um csv com todos os vcfs depois dos filtros e as informações de número de indivíduos e locus

workflow INPUT_CHECK { // Esse workflow vai ser incorporado ao subworkflow do structure, antes é necessário fazer
// um módulo para extrair as informações dos vcfs de saida do dartR em csv (ou usar dois scripts no modulo do dartR),
// um após o outro, talvez essa segunda seja a melhor opção. Depois usar a função com o que está escrito em main
// para obter os números de input do módulo do structure (meta.numind e meta.numloc)
    take:
    samplesheet

    main:
    samplesheet
    .splitCsv (header: true, sep: ',')
    .map { create_meta_vcf(it) }
    .set { vcf_meta }

    emit:
    vcf_meta
}

def create_meta_vcf(LinkedHashMap sheet) {
    def meta = [:]
    meta.numloc = sheet.numloc
    meta.numind = sheet.numind

    def vcf_meta = []
    vcf_meta = [meta , file(sheet.file)]
}


workflow{
    INPUT_CHECK(Channel.fromPath(params.input)).view()
}