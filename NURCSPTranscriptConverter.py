import PyPDF2
import re
import unicodedata


# TRATAMENTO DA TRANSCRIÇÃO
# Objetivo: rodar o script "ALIP_convert_transcripts.py"
# Como rodar:
# - o script deve estar na mesma pasta que o arquivo "AlipTranscriptConverter.py"
# - dentro do script, indique o diretório com os arquivos de transcrição bruta
# - indique também o diretório que receberá os arquivos de saida
# - confira se o "sample_type" está especificado corretamente ("censo" ou "interação")
# - $ python3 ALIP_convert_transcripts.py


class NURCSPTranscriptConverter:
    
    __std_alphabet='abcdefghijklmnopqrstuvwxyzç'
    __vocab='abcdefghijklmnopqrstuvwxyzçãàáâêéíóôõúû\-\'\n '
    __case_sens_vocab='ABCDEFGHIJKLMNOPQRSTUVWXYZÇÃÀÁÂÊÉÍÓÔÕÚÛabcdefghijklmnopqrstuvwxyzçãàáâêéíóôõúû1234567890@\-\'\n\?\./ '

    #is_single_letter
    #receives: a string
    #returns: True if the string is a single letter and that is part of the standard alfabet. False if otherwise
    def __is_single_letter(self, letter):

        if(len(letter) == 1 and letter in self.__std_alphabet):
            return True
        
        return False


    #expand_letter
    #receives: a string
    #returns: the letter equivalent in full
    #throws an error if the string is more than a single char or if the char is not an alfabetic letter
    def __expand_letter(self, letter):

        vet = ['a','bê','cê','dê','e','éfe','gê','agá','i','jota','cá','éle','eme','ene','o','pê','quê','érre',
        'ésse','tê','u','vê','dáblio','xis','ípsilom','zê', 'c cedilha']

        if(len(letter) > 1):
            raise NameError('Error in \'expand_letter\', passed value must be a single character')
            return

        if(ord(letter) < 97):
            raise NameError('Error in \'expand_letter\', passed value is not a valid letter')
            return

        if(letter == 'ç'):
            return vet[len(vet) - 1]

        return vet[ord(letter) - 97]

        
    #expand_letters
    #given a line, replaces single letters by its full equivalent
    #receives: a string
    #returns: a copy of the string, with single letters expanded
    def __expand_letters(self, text):

        vet = []

        for s in text.split():
            #print(s)
            if(self.__is_single_letter(s)):
                vet.append(s)
        # vet = [v for s in text.split() if is_single_letter(s)]

        for letter in vet:
            text = re.sub(' ' + letter + ' ', ' ' + self.__expand_letter(letter) + ' ', text)

        return text

    # remove_accent
    # given an accentuated letter, return its correspondant non-accentuated conterpart
    def __remove_accent(self, letter):

        letter = letter.lower()

        if((letter == 'á' or letter == 'â') or letter == 'ã'):
            letter = 'a'
        elif(letter == 'ê'):
            letter = 'e'
        elif(letter == 'í'):
            letter = 'i'
        elif(letter == 'ó' or letter == 'ô'):
            letter = 'o'
        elif(letter == 'ú' or letter == 'û'):
            letter = 'u'

        return letter


    # convert_to_speech_type
    # given a text following NURCSP notation, converts it to a single form of speech (formal or informal)
    # receives: the type of speech to be converted to (formal or informal); the text to be converted
    # return: the converted text
    def __convert_to_speech_type(self, type, text):

        if(type == 'form'):
            # remove mid word parenthesis
            text = re.sub("(?<=[a-zA-ZÀ-û])\(([a-z]*)\)(?=[a-zA-ZÀ-û])", r"\1", text)
            # remove end word parenthesis, remove accent
            text = re.sub("([a-zA-ZÀ-û])\(([a-zA-ZÀ-û]*)\)", lambda m: self.__remove_accent(m.groups()[0]) + m.groups()[1], text)
            no_ac = list(text)
            for i in range(len(text)):
                no_ac[i] = self.__remove_accent(text[i])
            text = ''.join(no_ac)
            #print(text)
                
        else:
            text = re.sub("(?<=[a-zA-ZÀ-û])\([a-z]+\)", "", text)
            text = re.sub("\([a-z]+\)(?=[a-zA-ZÀ-û]+)", "", text)

        return text


    def convert_transcript(self, filePath, output_dir, sample_type):

        # print('Hello')
        global __case_sens_vocab
        # 1-----------------------------------
        # add all pdf pages
        pdfFileObj = open(filePath, 'rb')
        pdfReader = PyPDF2.PdfFileReader(pdfFileObj)

        num_of_pages = pdfReader.getNumPages()

        raw_text = ""
        for page in range(num_of_pages):
            pageObj = pdfReader.getPage(page)
            raw_text += pageObj.extractText()

        # DEGUB CODE
        #pos = raw_text.find("mais jovem")
        #for j in range(pos-50, pos+50):
        #    print(raw_text[j], end='')
        #print("---")

        # print(raw_text)
        # print('done')

        # 2------------------------------------
        # remove everything before "\n----------------\s*\n"
        begin_marker = "------------------------\s*\n"
        raw_text = re.sub(begin_marker, "$$$", raw_text)
        transcript_start_index = raw_text.find('$$$')
        raw_text = raw_text[transcript_start_index+2:]

        # 4------------------------------------
        # remove "\n"
        raw_text = re.sub("\n{1}", "", raw_text)

        # 5------------------------------------
        # - remove "Projeto ALIP Œ Banco de dados IBORUNA\s*[0-9]*\s*"
        raw_text = re.sub("Projeto ALIP\s*.{1} Banco de dados IBORUNA[0-9\s\.]*", "", raw_text)

        # 6------------------------------------
        # replace "Doc.:" by "\nDoc.:"
        # raw_text = re.sub("Doc\.*:\s*", "\n", raw_text)
        #raw_text = re.sub("Doc\.*:*", "\nDoc@ ", raw_text)
        # - raplace "Inf.:" by "\nInf.:"
        # raw_text = re.sub("Inf\.*:\s*", "\n", raw_text)
        #raw_text = re.sub("Inf\.*:*", "\nInf@ ", raw_text)
        # break line on change of speeaker
        #raw_text = re.sub("L1", "\nL1@", raw_text)
        #raw_text = re.sub("L2", "\nL2@", raw_text)
        raw_text = raw_text[1:]

        # 7------------------------------------
        # mod
        #raw_text = re.sub("ininteligível", "", raw_text)
        #raw_text = re.sub("inint", "", raw_text)
        #raw_text = re.sub("inint\.", "", raw_text)

        # 8------------------------------------
        # normalize text
        # Apply NFC
        accents = ('COMBINING ACUTE ACCENT', 'COMBINING GRAVE ACCENT') #portuguese
        chars = [c for c in unicodedata.normalize('NFD', raw_text) if c not in accents]
        raw_text = unicodedata.normalize('NFC', ''.join(chars))# Strip accent

        # 9-----------------------------------
        # break text
        #raw_text = re.sub("\.\.\.(?!\.)", "\n\n", raw_text) # no caso de 4 pontos seguidos, quebra no final
        raw_text = re.sub("\.\.\.", "...\n\n", raw_text) # no caso de 3 pontos seguidos, quebra no final
        raw_text = re.sub("\?", "?\n\n", raw_text)
        #raw_text = re.sub("\s*\n\s*", "\n\n", raw_text)
        #raw_text = re.sub("\s*– –\s*", "\n\n", raw_text)
        # 10-----------------------------------
        # remove ((xxx))
        #raw_text = re.sub("\(\([^)]*\)\)", "", raw_text)

        ## 11-----------------------------------
        ## remove :: (so it won't interfere with the formal notation convertion)
        #raw_text = re.sub("::", " ", raw_text)

        ## 12-----------------------------------
        ## keep only informal notation
        #raw_text = self.__convert_to_speech_type('form', raw_text)

        ## 13-----------------------------------
        ## Remove any remaining trash
        #raw_text = raw_text.replace("`","'")
        #raw_text = re.sub("[^{}]".format(self.__case_sens_vocab), "", raw_text)
        #raw_text = re.sub("[ ]+", " ", raw_text)

        #raw_text = re.sub("(?<![A-Z])\.", "", raw_text)
        raw_text = re.sub("\n[ ]+", "\n", raw_text)
        raw_text = re.sub("\n{3, 6}", "\n\n", raw_text)
        #raw_text = re.sub("[ ]+", " ", raw_text)

        ## 14-----------------------------------
        ## Expand letters
        ## raw_text = self.__expand_letters(raw_text)

        output_name = output_dir + '/' + filePath.split('/')[-1].split('.')[0].replace('TRANS', 'SPLITED') + '.txt'

        # DEBUG CODE
        #lines = raw_text.split("\n")
        #i = 0
        #for l in lines:
        #    if "mais jovem" in l:
        #        i = 3
        #    if (i > 0):
        #        print(l)
        #        i -= 1
        
        # 15------------------------------------
        # Write file
        with open(output_name, 'w') as f:
            f.write(raw_text)
