from hash_map import *
import sys
import re
from phoneme_rules import *
from Word import *
#from GramaticalClass import *
import string

from phonemizer.backend import EspeakBackend
backend = EspeakBackend('pt-br')

def ipa_to_ascii(word):
    new_word = ""
    for c in word:
        if c == 'i':
            new_word += 'i'
        elif c == 'e':
            new_word += 'e'
        elif c == 'ɛ':
            new_word += 'eh'
        elif c == 'a':
            new_word += 'a'
        elif c == 'ɔ':
            new_word += 'oh'
        elif c == 'o':
            new_word += 'o'
        elif c == 'u':
            new_word += 'u'
        elif c == 'e':
            new_word += 'e'

    return new_word

class Conversor:
	def convert_sentence(self, sentence, rel_path):
		Word.update_exceptions(rel_path)	
		sentence = sentence.replace('-', ' ')
		sentence = sentence.translate(str.maketrans('', '', string.punctuation))
		converted_sentence = ''
		#gramatica = GramaticalClass(sentence)
		
		lista = [i.lower() for i in re.findall(r"[\w]+", sentence)]
		words = [Word(palavra.strip(), pos, rel_path) for pos, palavra in enumerate(lista)]
		words = list(map(lambda x: x.find_tonic(), words))

		for num, word in enumerate(words):
			#word.setClass()
			#word.setBase()
			word.setIsExc()
			word.find_tonic()
			if(num != len(words)-1):
				word.setIsSandi(words[num+1])
				#print('palavra:', word.word)

			#conversion_list = backend.phonemize([word.word])
			#conversion = conversion_list[0]
			#print(conversion)
			try:
				# para debugar:
				print(word.word)
				conversion = word.convert_word()
                # o alinhador não reconhece os fonemas w e y
				conversion = conversion.replace("w", "v")
				conversion = conversion.replace("y", "i")
				print(conversion)
			except:
				print("problema:", word.word)
				conversion_ipa = backend.phonemize([word.word])
				conversion = ipa_to_ascii(conversion_ipa)
			if(word.getIsSandi()):
				converted_sentence += conversion
			else:
				converted_sentence += conversion + ' '

		return converted_sentence
