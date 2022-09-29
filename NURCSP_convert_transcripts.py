import os
import re
import argparse
from num2words import num2words
from os import path, listdir
from NURCSPTranscriptConverter import NURCSPTranscriptConverter

parser = argparse.ArgumentParser()
parser.add_argument('--dir', default='./')
parser.add_argument('--smart', default=1)
parser.add_argument('--sent_min', default=3)
parser.add_argument('--consec_max', default=2)
parser.add_argument('--output_name', default="_smart")

args = parser.parse_args()

transcripts_dir = args.dir
smart_seg = args.smart
sent_min = args.sent_min
consec_max = args.consec_max
output_name = args.output_name

converter = NURCSPTranscriptConverter()

raw_vocab ='ABCDEFGHIJKLMNOPQRSTUVWXYZÇÃÀÁÂÊÉÍÓÔÕÚÛabcdefghijklmnopqrstuvwxyzçãàáâêéíóôõúû1234567890@()\-\'\n\?\./,\'\": '
clean_vocab ='ABCDEFGHIJKLMNOPQRSTUVWXYZÇÃÀÁÂÊÉÍÓÔÕÚÛabcdefghijklmnopqrstuvwxyzçãàáâêéíóôõúû1234567890@\-\'\n\? '

# transcripts_dir = './IBORUNA/Amostra-Censo/1-Transcricao'
output_dir = transcripts_dir
#output_dir = './segmented_text'

if(not os.path.exists(transcripts_dir)):
    print('ERROR: Transcripts folder does not exist')
    exit()

if(not os.path.exists(output_dir)):
    os.mkdir(output_dir)

total_files = 0
for file in os.listdir(transcripts_dir):
    if('.pdf' in file):
        print(args.dir)
        total_files += 1
successful_files = 0
failed_files = 0

def rebuild_text(lines):
    nt = ""    
    for l in lines:
        nt = nt + l + "\n\n"
    return nt  

def contains_num(s):
    return any(i.isdigit() for i in s)

def create_aux_file(new_text):
    sentences = new_text.split("\n\n")
    curr = ""
    locs_list = []
    with open(output_dir+'/'+"locutores.txt", "w") as aux_f:
        for s in sentences:
            if len(s) <= 0:
                continue
            inicio = s.split(" ")[0].lower()
            #print(s)

            if "inf" in inicio and len(inicio) <= 6:
                curr = inicio
            elif "doc" in inicio and len(inicio) <= 6:
                curr = inicio
            elif "loc" in inicio and len(inicio) <= 6:
                curr = inicio
            elif inicio[0] == "l":
                if len(inicio) > 1 and inicio[1].isdigit():
                    curr = inicio
            #elif inicio[0].isdigit():
            #    curr = inicio

            if curr not in locs_list:
                locs_list.append(curr)

            #print(curr)
            aux_f.write(curr+";"+s+"\n")

    new_text = new_text.lower()
    for loc in locs_list:
        print("locutor: "+loc)
        new_text = new_text.replace(loc, "")
    return new_text

def clean_text(new_text):
    if new_text[0] == ' ':
        new_text = new_text[1:]

    new_text = re.sub("ininteligível", "", new_text)
    new_text = re.sub("inint", "", new_text)
    new_text = re.sub("inint\.", "", new_text)

    # Remove texto entre parênteses duplos
    new_text = re.sub("\(\([^)]*\)\)", "", new_text)

    # Remove texto entre parênteses duplos e "..." (caso o transcritor tenha esquecido de fechar os parênteses)
    new_text = re.sub("\(\([^(\.\.\.)]*\.\.\.", "", new_text)

    # Remove ::
    new_text = re.sub("::", "", new_text)

    # Troca ` por '
    new_text = new_text.replace("`","'")

    # se não há texto, só pontuação, retornamos a string vazia ""
    if not re.search('[A-Za-z0-9áàâãéèêíóôõúçÁÀÂÃÉÈÍÓÔÕÚÇ]', new_text):
        return ""

    # Formata conforme o vocabulário limpo
    new_text = re.sub("[^{}]".format(clean_vocab), "", new_text)

    # Remove múltiplos espaços
    new_text = re.sub("[ ]+", " ", new_text)

    new_text = re.sub("(?<![A-Z])\.", "", new_text)
    new_text = re.sub("\n[ ]+", "\n", new_text)
    new_text = re.sub("\n{3, 6}", "\n\n", new_text)
    new_text = re.sub("[ ]+", " ", new_text)

    new_text = re.sub(' +', ' ', new_text)
    new_text = new_text.replace("\n ", "\n")
    
    if len(new_text.split("\n")) > 0:
        new_text = os.linesep.join([s for s in new_text.splitlines() if s])
    return new_text

# Se ativada, segmenta o texto não apenas em reticências, mas também em pontos de interrogação, e concatena sentenças curtas
def smart_segmentation(filename):
	# lê o arquivo gerado pela segmentação simples e salva sentenças
    with open(output_dir+'/'+filename.replace(".pdf",".txt"), 'r') as f:
        linhas = f.readlines()

    # flag que indica se a transcriçao possui mais de um locutor (0 se não, 1 se sim)
    two_voices = 1

    with open(output_dir+'/'+"raw_"+output_name, 'w') as nrf:
        concat = []
        new_text = ""
        ch = 0
        curr = ""
        locs_list = []
        i = 0
        troca_loc = -1
        mult_locs = False
        for linha in linhas:
            if i < 20:            
                print(linha)

            if linha in ['\n', '\n', '\n ']:
                continue

            # Verifica se mudou o locutor
            inicio = linha.split(" ")[0].lower()
            if(two_voices == 1):
                if "inf" in inicio and len(inicio) <= 6:
                    curr = linha.split(" ")[0]
                    ch = 1
                elif "doc" in inicio and len(inicio) <= 6:
                    curr = linha.split(" ")[0]
                    ch = 1
                elif "loc" in inicio and len(inicio) <= 6:
                    curr = linha.split(" ")[0]
                    ch = 1
                elif inicio[0] == 'l' and len(inicio) == 1:
                    curr = linha.split(" ")[0]
                    ch = 1
                elif inicio[0] == "l" and inicio[1].isdigit():
                    #print("aqui!")
                    curr = linha.split(" ")[0]
                    ch = 1
                #elif inicio[0].isdigit():
                #    curr = linha.split(" ")[0]
                #    ch = 1
                else:
                    ch = 0

                # Tratamos o caso em que uma mesma sentença possui mais de uma troca de locutores, e fazemos uma lista de subsentenças separadas por locutor
                if len([l for l in locs_list if l in linha]) > 1:
                    mult_locs = True
                    linha = re.sub("[ ]+", " ", linha)
                    palavras = linha.split()
                    print("mult_locs, palavras:", palavras)
                    sublines = []
                    last_ch = 0
                    for pi, p in enumerate(palavras):
                        if p in locs_list or pi == len(palavras)-1:
                            if pi == len(palavras)-1:
                                pi += 1
                            subl = ""
                            for j in palavras[last_ch:pi]:
                                subl = subl + " " + j
                            sublines.append(subl)
                            last_ch = pi

                # Se o locutor mudou e haviam sentenças pequenas na fila, concatena essas da fila com a sentença anterior, do mesmo locutor, e esvazia a fila
                if (ch == 1):
                    if linha.split(" ")[0] not in locs_list:
                        locs_list.append(linha.split(" ")[0])
                    if concat:
                        nl = "\n"
                        for l in concat:
                            nl = nl + l.replace("\n", "") + ' '
                        new_text = new_text + nl + '\n'
                        concat = []
                
                # Se tivemos mais de uma troca de locutores, aqui despejamos a lista das subsentenças no arquivo, uma por uma
                if mult_locs:
                    for subl in sublines:
                        if subl:
                            new_text += subl + '\n'
                    mult_locs = False
                    continue

            # Verifica se há troca de locutor sem quebra de linha, e insere a quebra na troca, caso ocorra
            for loc in locs_list:
                if loc == curr:
                    continue
                if loc in linha:
                    ch = 1
                    #print("concat:", concat)
                    if concat:
                        nl = "\n"
                        for l in concat:
                            nl = nl + l.replace("\n", " ")
                        nl = nl[:-1]
                        new_text = new_text + nl + '\n'
                        concat = []
                    pos = linha.find(loc)
                    linha = linha[:pos] + "\n" + linha[pos:]
                    new_text = new_text + linha
                    troca_loc = i
                    i += 1                    
                    continue

            # Prossegue para a próxima iteração caso tenha havido troca de locutor sem quebra de linha, tratada no bloco anterior
            if troca_loc != -1:
                troca_loc = -1
                continue      

            # Segmenta em ? e em !, adicionando à sentença anterior
            if('?' in linha or '!' in linha):
                if concat:
                    nl = "\n"
                    for l in concat:
                        nl = nl + l.replace("\n", "")+ ' '
                    new_text = new_text + nl + '\n'
                    concat = []
                
                if i == (troca_loc+1):
                    new_text = new_text + '\n' + linha
                    troca_loc = -1
                elif(ch == 0):
                    new_text = new_text[:-1]+ " " +linha
                else:
                    new_text = new_text + '\n' + linha

            # Segmenta se a linha não é mais curta que a sentença mínima e a fila de sentenças pequenas não é maior que consec_max
            elif(len(linha.split(' ')) > int(sent_min) or len(concat) == int(consec_max)):
                flagdebug = False
                if concat:
                    nl = ""
                    for l in concat:
                        nl = nl + l.replace("\n", "")+' '
                    linha = nl + linha
                    concat = []
                new_text = new_text + linha
            
            # Se a sentença é mais curta que a sent_min, adiciona na fila de sentenças pequenas 
            else:
                concat.append(linha)

            i += 1
        # formata texto cru
        new_text = re.sub("[^{}]".format(raw_vocab), "", new_text)
        # remove quebras de linha repetidas
        new_text = re.sub("[\n]+", "\n", new_text)        

        # remove espaços repetidos
        new_text = re.sub(' +', ' ', new_text)
        # remove todas as quebras de linha repetidas
        new_text = os.linesep.join([s for s in new_text.splitlines() if s])
        # remove sentenças que começam ou terminam com espaço
        new_text = new_text.replace("\n ", "\n")
        new_text = new_text.replace(" \n", "\n")

        # remove sentenças compostas apenas pela quebra "..."
        new_text = new_text.replace("\n...\n", "\n")

        # Adiciona quebras de linha duplas
        lns = new_text.split("\n")
        new_text = ""
        for ln in lns:
            new_text = new_text + ln + "\n\n"
        new_text = new_text[:-1]

        # escreve em arquivo _raw
        nrf.write(new_text)

        # abre arquivo _clean
        with open(output_dir+'/'+"clean_"+output_name, 'w') as ncf:
            ## separa arquivo _raw em linhas
            #new_lines = new_text.split("\n\n")
            ##print(new_lines[0])
            ## limpa individualmente cada linha nos formato _clean
            #for nl in new_lines:
            #    if len(nl) > 0:
            #        nl = clean_text(nl)
            #    #print(nl)
            ## junta texto das linhas em uma string só
            #new_text = rebuild_text(new_lines)
            
            # criamos a lista de locutores
            new_text = create_aux_file(new_text)

            # novamente separa o texto em linhas
            new_lines = new_text.split("\n")

            # limpa novamente as linhas após as alterações da função create_aux_file
            new_text = ""
            for nl in new_lines:
                if len(nl) > 0:
                    nl = clean_text(nl)
                new_text = new_text + nl + '\n'
                #print(nl)

            new_text = os.linesep.join([s for s in new_text.splitlines() if s])
            new_text = new_text.replace("\n ", "\n")
            new_text = new_text.replace(" \n", "\n")
            new_text = new_text.replace("\n", "\n\n")

            # Escreve para o arquivo final
            ncf.write(new_text)

# converter.convert_transcript(os.path.join(transcripts_dir, 'AI-001-CAS-5-TRANS.pdf'), output_dir, 'interacao')
for file in os.listdir(transcripts_dir):
    if('.pdf' in file):
        try:
            converter.convert_transcript(os.path.join(transcripts_dir, file), output_dir, 'interacao')
            print('Successfully converted \'' + file + '\'')
            successful_files += 1
            if(smart_seg == 1):
                smart_segmentation(file)
        
        except Exception as e:
            print('Failed to convert \'' + file + '\'')
            print(e)
            failed_files += 1

        print('Sus: ' + str(successful_files) + ' | Fai: ' + str(failed_files))
        print('Progress: ' + str(successful_files+failed_files) + '/' + str(total_files))
