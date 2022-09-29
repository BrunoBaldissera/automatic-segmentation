from spellchecker import SpellChecker
from Conversor import *
from TextConverter import *
import re
from num2words import num2words
from os import path, listdir
import pydub
import glob
from os.path import isfile, join
from collections import OrderedDict
import librosa
import numpy as np
import os
import sys
from scipy.io.wavfile import write
import chardet
import tgt

raw_vocab ='ABCDEFGHIJKLMNOPQRSTUVWXYZÇÃÀÁÂÊÉÍÓÔÕÚÛabcdefghijklmnopqrstuvwxyzçãàáâêéíóôõúû()\-\'\n\?\./,\'\": '
clean_vocab ='ABCDEFGHIJKLMNOPQRSTUVWXYZÇÃÀÁÂÊÉÍÓÔÕÚÛabcdefghijklmnopqrstuvwxyzçãàáâêéíóôõúû\-\'\n\? '

#############################################################
# Linked segments lists
#############################################################
class AudioSegment:
  def __init__(self, start, end):
    self.start = start
    self.end = end
    self.next = None
    self.gap = 0 # gap between segments (current and next)

  def set_next(self, next):
    self.next = next
    self.gap = next.start - self.end

  def set_filename_and_id(self, filename, id):
    self.filename = filename
    self.id = id

  def merge_from(self, next): 
    # merge two segments (current and next)
    self.next = next.next
    self.gap = next.gap
    self.end = next.end

  def duration(self, sample_rate):
    return (self.end - self.start - 1) / sample_rate

class AutomaticSegmentation:
    def __init__(self, path, audio_file, locs_file):
        self.path = path
        self.audio_file = audio_file
        self.locs_file = locs_file
        self.text_align = ""
        self.silences_file = ""
        self.alignment_tg = ""

    # palavras_por_locutor
    def clean_text(self, new_text):
        if new_text[0] == ' ':
            new_text = new_text[1:]

        new_text = re.sub("ininteligível", "", new_text)
        new_text = re.sub("inint", "", new_text)
        new_text = re.sub("inint\.", "", new_text)

        # Remove texto entre parênteses duplos
        new_text = re.sub("\(\([^)]*\)\)", "", new_text)

        # Remove texto entre parênteses duplos e "..." (caso o transcritor tenha esquecido de fechar os parênteses)
        new_text = re.sub("\(\([^(\.\.\.)]*\.\.\.", "", new_text)

        # Troca :: por espaço
        new_text = re.sub("::", " ", new_text)

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

        # Substitui ehhhhhh por eh e afins    
        new_text = re.sub("h+", "h", new_text)

        new_text = re.sub(' +', ' ', new_text)
        new_text = new_text.replace("\n ", "\n")
        
        if len(new_text.split("\n")) > 0:
            new_text = os.linesep.join([s for s in new_text.splitlines() if s])
        return new_text

    def filled_pause_normalization(word):
        # éh, eh
        filled_pause_eh = ["éh","ehm","ehn","he","éhm","éhn","hé"]
        if word in filled_pause_eh:
            word = "eh"

        # uh, hum, hm, uhm
        filled_pause_uh = ["hum","hm","uhm","hu","uhn"]
        if word in filled_pause_uh:
            word = "uh"

        # uhum, aham
        filled_pause_aham = ["uhum","uhun","unhun","unhum","umhun","umhum",
                             "hunhun","humhum","hanhan","ahan","uhuhum"]
        if word in filled_pause_aham:
            word = "aham"

        # ah, hã, ãh, ã
        filled_pause_ah = ["hã","ãh","ã","ah","ahn","han","ham"]
        if word in filled_pause_ah:
            word = "ah"
        return word

    # Se a palavra não é reconhecida pela lista de palavras da biblioteca, retorna a palavra corrigida.
    #  Se não, retorna a mesma palavra passada por parâmetro
    def spellcheck(word):
        spell = SpellChecker(language='pt')
        #print("spellchecking", word)

        fp_word = self.filled_pause_normalization(word)
        if fp_word != word:
            return fp_word

        #if spell.unknown([word]):
        #    if '-' or '?' in word:
        #        return word
        #    print("palavra \"", word, "\" não reconhecida convertida para", spell.correction(word))
        #    return spell.correction(word)
        #else:
        #    return word
        return word

    # Gera dois novos arquivos com as falas e os turnos
    def generate_words_file(self, locs_file):
        with open(locs_file, 'r') as lf:
            linhas = lf.readlines()

        is_new = True
        locs_list = []
        for i, l in enumerate(linhas):
            if l == '\n':
                break
            loc = l.split(';')[0]
            if loc not in locs_list:
                for ll in locs_list:
                    # se o locutor atual é igual a um anterior com a adição ou remoção de um caractere "-" ou ".", é o mesmo locutor, e não adicionamos novamente
                    if loc == ll+"." or loc == ll+"-" or (ll[-1] == "." and ll[:len(loc)] == loc) or (ll[-1] == "-" and ll[:len(loc)] == loc):
                        is_new = False
                        print("mesmo locutor com nomes diferentes", loc, ll)
                        print("trocando", l)
                        linhas[i] = ll + ";" + l.split(';')[1]
                        print("por", l)
                        break
                if is_new:
                    print("loc appended", loc)
                    locs_list.append(loc)
                is_new = True
                        

        with open(locs_file.replace(".txt", "_palavras_align.txt"), 'w') as nlf2:
            with open(locs_file.replace(".txt", "_palavras.txt"), 'w') as nlf:
                for l in linhas:
                    loc = l.split(";")[0]
                    l = l.split(";")[1]
                    l = l.lower()
                    # remove indicação do locutor no início da frase
                    for iloc in locs_list:
                        if iloc in l:
                            l = l.replace(iloc, "")
                    l = self.clean_text(l)
                    for lp in l.split():
                        #lp = spellcheck(lp)
                        # se palavra é só um número continua sem escrever
                        if lp.isnumeric():
                            continue
                        nlf.write(loc+';'+lp+"\n")
                        nlf2.write(lp+"\n")
        self.text_align = locs_file.replace(".txt", "_palavras_align.txt")

    # alinhador_fonetico
    def clean_tg(self):
	    if('win' not in sys.platform.lower()):
		    subprocess.run(['rm','-f', '*.TextGrid'])
		    subprocess.run(['rm','-f','-r','output'])
	    else:
		    os.system(" ".join(['del', '*.TextGrid', '>NUL 2>&1']))
		    #subprocess.run(['del',\
		    #		'*.wav',\
		    #		'a',\
		    #		'b',\
		    #		'*.mlf',\
		    #		'dict',\
		    #		'*.out',\
		    #		'*.lab',\
		    #		'*.scp',\
		    #		'*.matl',\
		    #		'*.mfc',\
		    #		'hmmdefs',\
		    #		'*.TextGrid'],\
		    #		stdout=subprocess.DEVNULL,\
		    #		stderr=subprocess.DEVNULL)

    def clean(self):
        TMP = os.path.join(base_dir,'tmp/')
        if('win' not in sys.platform.lower()):
            subprocess.run(['rm',\
                    '-f',\
                    os.path.join(TMP,'*.wav'),\
                    os.path.join(TMP,'a'),\
                    os.path.join(TMP,'b'),\
                    os.path.join(TMP,'dict'),\
                    os.path.join(TMP,'*.mlf'),\
                    os.path.join(TMP,'*.out'),\
                    os.path.join(TMP,'*.lab'),\
                    os.path.join(TMP,'*.scp'),\
                    os.path.join(TMP,'*.matl'),\
                    os.path.join(TMP,'*.mfc'),\
                    os.path.join(TMP,'hmmdefs')])
            subprocess.run(['rm','-f','-r','output'])
        else:
            os.system(" ".join(['del',\
                    os.path.join(TMP,'*.wav'),\
                    os.path.join(TMP,'a'),\
                    os.path.join(TMP,'b'),\
                    os.path.join(TMP,'*.mlf'),\
                    os.path.join(TMP,'dict'),\
                    os.path.join(TMP,'*.out'),\
                    os.path.join(TMP,'*.lab'),\
                    os.path.join(TMP,'*.scp'),\
                    os.path.join(TMP,'*.matl'),\
                    os.path.join(TMP,'*.mfc'),\
                    os.path.join(TMP,'hmmdefs'),\
                    '>NUL 2>&1']))

    def align(self, audio_file, text_align):
        TC = TextConverter()
        self.clean_tg()

        with open(text_align, 'r') as tf:
            text = tf.read()

        hmmdefs = False
        req_in = 'graf'
        req_out = 'fonema'
        aligner = 'HTK'

        # Cria todos os arquivos necessarios para o alinhamento	
        text1 = TC.perform_conversion(text,
                    audio_file,
                    req_in,
                    req_out,
                    aligner)

        TC.align(text, audio_file, req_in, req_out, aligner=aligner, hmmdefs=hmmdefs)
        print("alinhamento feito")

        # Formata o arquivo de saida para o formato desejado
        TC.format_output(text1, audio_file, req_out, aligner)
        print("saidas geradas")

        self.alignment_tg = self.audio_file.replace(".wav", ".TextGrid")

        self.clean()

    #############################################################
    # Segment audio file and return a segment linked list
    #############################################################
    def segment_wav(self, wav, threshold_db):
      # Find gaps at a fine resolution:
      parts = librosa.effects.split(wav, top_db=threshold_db, frame_length=1024, hop_length=256)

      # Build up a linked list of segments:
      head = None
      for start, end in parts:
        segment = AudioSegment(start, end)
        if head is None:
          head = segment
        else:
          prev.set_next(segment)
        prev = segment
      return head

    #############################################################
    # Given an audio file, creates the best possible segment list 
    #############################################################
    def find_segments(self, filename, wav, sample_rate, min_duration, max_duration, max_gap_duration, threshold_db, wav_dest_dir):
      # Segment audio file
      segments = self.segment_wav(wav, threshold_db)

      # Convert to list
      result = []
      s = segments
      while s is not None:
        result.append(s)
        # Create a errors file
        if (s.duration(sample_rate) < min_duration and
            s.duration(sample_rate) > max_duration):
            with open(os.path.join(os.path.dirname(__file__), "erros.txt"), "a") as f:
                f.write(filename+"\n")
        s = s.next

      with open(wav_dest_dir+'/'+"silences.txt", "w") as sf:
        for r in result:
          sf.write(str(r.start/sample_rate) + ' ' + str(r.end/sample_rate) + "\n")

      return result

    #############################################################
    # Given an folder, creates a wav file alphabetical order dict  
    #############################################################
    def load_filenames(self, base_dir, orig):
      mappings = OrderedDict()
      for filepath in glob.glob(join(base_dir, orig + "/*.wav")):
        filename = filepath.split('/')[-1].split('.')[0]
        mappings[filename] = filepath
      return mappings

    #############################################################
    # Build best segments of wav files  
    #############################################################
    def find_silences(self, base_dir, orig, dest, sampling_rate, min_duration, max_duration, max_gap_duration, threshold, output_filename, output_filename_id):
      # Creates destination folder
      wav_dest_dir = os.path.join(base_dir, dest)
      os.makedirs(wav_dest_dir, exist_ok=True)
      # Initializes variables
      max_duration, mean_duration = 0, 0
      all_segments = []
      total_duration = 0
      filenames = self.load_filenames(base_dir, orig)
      for i, (file_id, filename) in enumerate(filenames.items()):
        print('Loading %s: %s (%d of %d)' % (file_id, filename, i+1, len(filenames)))
        wav, sample_rate = librosa.load(filename, sr=sampling_rate)
        print(' -> Loaded %.1f min of audio. Splitting...' % (len(wav) / sample_rate / 60))

        # Find best segments
        segments = self.find_segments(filename, wav, sample_rate, min_duration, max_duration,
          max_gap_duration, threshold, wav_dest_dir)
        duration = sum((s.duration(sample_rate) for s in segments))
        total_duration += duration

        # Create records for the segments
        output_filename = output_filename  if output_filename else file_id
        j = int(output_filename_id)
        for s in segments:
          all_segments.append(s)
          s.set_filename_and_id(filename, '%s-%04d' % (output_filename, j))
          j = j + 1

        print(' -> Segmented into %d parts (%.1f min, %.2f sec avg)' % (
          len(segments), duration / 60, duration / len(segments)))

        # Write segments to disk:
        for s in segments:
          #segment_wav = (wav[s.start:s.end] * 32767).astype(np.int16)
          segment_wav = (wav[s.start:s.end] * 32767).astype(np.int16)
          out_path = os.path.join(wav_dest_dir, '%s.wav' % s.id)
          #librosa.output.write_wav(out_path, segment_wav, sample_rate)
          #write(out_path, sample_rate, segment_wav)

          duration += len(segment_wav) / sample_rate
          duration_segment = len(segment_wav) / sample_rate
          if duration_segment > max_duration:
            max_duration = duration_segment

          men_duration = mean_duration + duration_segment
        print(' -> Wrote %d segment wav files' % len(segments))
        print(' -> Progress: %d segments, %.2f hours, %.2f sec avg' % (
          len(all_segments), total_duration / 3600, total_duration / len(all_segments)))

      print('Writing metadata for %d segments (%.2f hours)' % (len(all_segments), total_duration / 3600))
      with open(os.path.join(base_dir, 'segments.csv'), 'w') as f:
        for s in all_segments:
          f.write('%s|%s|%d|%d\n' % (s.id, s.filename, s.start, s.end))
      print('Mean: %f' %( mean_duration ))
      print('Max: %d' %(max_duration ))
      self.silences_file = self.path+"silences.txt"

    def predict_encoding(self, tg_path):
        '''Predict a file's encoding using chardet'''
        # Open the file as binary data
        with open(tg_path, 'rb') as f:
            # Join binary lines for specified number of lines
            rawdata = b''.join(f.readlines())

        return chardet.detect(rawdata)['encoding']

    def calculate_average_phone_duration(self, window_phones):
        s = 0
        for wp in window_phones:
            s += wp[1]
        return s / len(window_phones)

    def dsr_threshold_1(self, windows, boundaries_tier_1, delta1):
        # primeiro encontramos o maior e menor speech rates no turno
        max_sr_diff = 0
        last_sr = 0
        for w in windows:
            #print(w)
            if abs(w[1] - last_sr) > max_sr_diff:
                max_sr_diff = abs(w[1] - last_sr)
            last_sr = w[1]

        # se a diferença entre os speech rates das janelas consecutivas é > delta1 da maior diferença entre speech rates,
        #  identificamos como DSR.
        last_sr = 0
        dsrs_1 = []
        dsr_windows_1 = []
        # tempo da última fronteira
        last_boundary = 0
        for w in windows:
            print(abs(w[1] - last_sr), ", threshold is", delta1, "*", max_sr_diff, "=", delta1 * max_sr_diff)
            if abs(w[1] - last_sr) > delta1 * max_sr_diff:
                print("DSR!", w)
                print("adding boundary to tier 1")
                boundary = tgt.core.Interval(start_time=last_boundary, end_time=w[0][0][2])
                last_boundary = w[0][0][2]
                try:
                    boundaries_tier_1.add_interval(boundary)
                except:
                    print("overlap!")
                dsrs_1.append(w[0][0][2])
                dsr_windows_1.append(w)
            last_sr = w[1]

        return dsrs_1, dsr_windows_1

    def dsr_threshold_2(self, dsr_windows_1, boundaries_tier_2, delta2, interval_size):
        max_sr = 0
        min_sr = 9999
        for dsr in dsr_windows_1:
            #print(w)
            if dsr[1] > max_sr:
                max_sr = dsr[1]
            if dsr[1] < min_sr:
                min_sr = dsr[1]

        # se a diferença entre os speech rates das janelas consecutivas é > delta2 da maior diferença entre speech rates,
        #  identificamos como DSR.
        last_dsr = 0
        filtered = []
        for dsr in dsr_windows_1:
            if dsr[0][0][2] - last_dsr > interval_size:
                filtered.append(dsr)
            last_dsr = dsr[0][0][3]

        dsrs_2 = []
        last_sr = 0
        last_boundary = 0
        for w in filtered:
            if abs(w[1] - last_sr) > delta2 * (max_sr - min_sr):
                print("DSR!", w)
                print("adding boundary to tier 2")
                boundary = tgt.core.Interval(start_time=last_boundary, end_time=w[0][0][2])
                last_boundary = w[0][0][2]
                try:
                    boundaries_tier_2.add_interval(boundary)
                except:
                    print("overlap!")
                dsrs_2.append(w[0][0][2])
            last_sr = w[1]
        return dsrs_2

    def print_silences(self, sil_file, boundaries_tier_3, silence_threshold):
        with open(sil_file, "r") as sf:
            sils = sf.readlines()
        
        silences = []
        last_boundary = 0
        for s in sils:
            #print("intervalo de fala", s)
            interval = s.split()
            #print("trecho de silêncio entre", last_boundary, "e", float(interval[0]), "com duração", float(interval[0]) - last_boundary)
            if float(interval[0]) - last_boundary > silence_threshold:
                print("DSR!", interval[0])
                #print("adding boundary to tier 3")
                boundary = tgt.core.Interval(start_time=last_boundary, end_time=float(interval[0]))
                last_boundary = float(interval[1])
                try:
                    boundaries_tier_3.add_interval(boundary)
                except:
                    print("overlap!")
                silences.append(float(interval[0]))
        return silences

    def fill_boundaries_tier(self, timestamps, boundaries_tier):
        timestamps.sort()

        #print("timestamps:", timestamps)

        last_ts = timestamps[0]
        for ts in timestamps[1:]:
            #print("boundary:", last_ts, ts)
            boundary = tgt.core.Interval(start_time=last_ts, end_time=ts)
            boundaries_tier.add_interval(boundary)
            last_ts = ts

    def find_boundaries(self, locs_file, tg_file, sil_file, window_size, delta1, delta2, interval_size, silence_threshold, min_words_h2):
        tg = tgt.io.read_textgrid(tg_file, self.predict_encoding(tg_file), include_empty_intervals=False)
        tier_names = []
        boundaries_tier_1 = tgt.core.IntervalTier(start_time=tg.start_time, end_time=tg.end_time, name="fronteiras_heuristica_1", objects=None)
        boundaries_tier_2 = tgt.core.IntervalTier(start_time=tg.start_time, end_time=tg.end_time, name="fronteiras_heuristica_2", objects=None)
        boundaries_tier_3 = tgt.core.IntervalTier(start_time=tg.start_time, end_time=tg.end_time, name="fronteiras_heuristica_3", objects=None)
        boundaries_tier = tgt.core.IntervalTier(start_time=tg.start_time, end_time=tg.end_time, name="fronteiras_metodo", objects=None)
        names = tg.get_tier_names()
        fim = False

        # lemos as palavras no arquivo com locutores
        with open(locs_file, 'r') as lf:
            locs_and_words = lf.readlines()

            # lemos as palavras na coversão g2p
            Conv = Conversor()
            sentences = ""
            for lw in locs_and_words:
                w = lw.split(';')[1]
                sentences += ' ' + w
            #print("sentences:", sentences)
            g2p_words = Conv.convert_sentence(sentences)
            sentences = sentences.split()
            #print("sentences converted:", g2p_words)
            g2p_words = g2p_words.split()
            for pwi, pw in enumerate(g2p_words):
                # substituimos fonemas 'w' por 'v' (e 'y' por 'i') pois o alinhador joga fora (é uma solução tosca, mas o que dá pra fazer sem alterar o alinhador)
                g2p_words[pwi] = g2p_words[pwi].replace("w", "v")
                g2p_words[pwi] = g2p_words[pwi].replace("y", "i")
                
            print("g2p_words -> lista:", g2p_words)
            
        for name in names:
            tier = tg.get_tier_by_name(name)
            
            # índice para iterar pelas palavras convertidas via g2p
            i = 0
            curr_turn_start = 0.0
            curr_turn = ""
            window_phones = []
            tier_names = []
            windows = []
            all_timestamps = []
            constr = ""
            first_phone = True
            curr_word = g2p_words[0]
            curr_word_grapheme = sentences[0]
            curr_loc = locs_and_words[0].split(';')[0]
            turn_index = 0
            last_turn_start = 0

            # estrutura que guarda, em ordem, cada palavra do texto, o texto do turno atual até dada palavra, o locutor desse turno,
            #  o início do tempo da palavra, o início do tempo do turno e o final do tempo do turno até a palavra (final do tempo da palavra)
            turn_until_word = []
            for w in sentences:
                turn_until_word.append(["", "", "", 0.0, 0.0, 0.0])
            turn_until_word[0] = [curr_word_grapheme, curr_word_grapheme, curr_loc, 0.0, 0.0, 0.0]            

            for enum, interval in enumerate(tier.intervals):
                #print("fonema:", interval.text)
                if enum == 0 or enum == len(tier.intervals)-1:
                    continue
                if first_phone:
                    curr_window = [interval.start_time, interval.start_time + window_size]
                    first_phone = False
                
                constr += interval.text
                if len(constr) < 20:
                    print("curr_word:", curr_word, "; constr:", constr)
                
                # adicionamos à lista de fonemas da janela o fonema atual e sua duração caso esteja dentro do tempo da janela
                if interval.end_time < curr_window[1]:
                    window_phones.append([interval.text, interval.end_time - interval.start_time, interval.start_time, interval.end_time, curr_loc])
                
                # se os fonemas encontrados desde a última janela formam a próxima palavra, pulamos para a próxima janela
                if constr == curr_word:
                    curr_turn += ' ' + curr_word_grapheme
                    #print("window phones:", window_phones)
                    if window_phones:                
                        av = self.calculate_average_phone_duration(window_phones)
                        windows.append([window_phones, av])

                    # atualiza o turno atual até a palavra e o tempo do final do turno até agora
                    turn_until_word[i][1] = curr_turn
                    turn_until_word[i][5] = interval.end_time
                    
                    # se a palavra atual foi concluída, pulamos para a próxima
                    i += 1
                    try:
                        curr_word = g2p_words[i]
                        curr_word_grapheme = sentences[i]
                        #print(curr_word)

                        # atualiza a palavra atual no turno e seu tempo de início 
                        turn_until_word[i][0] = curr_word_grapheme
                        turn_until_word[i][3] = interval.end_time
                        turn_until_word[i][4] = last_turn_start
                        print("turn_until_word:", turn_until_word[i])
                    except:
                        print("lista de g2p acabou")
                        fim = True
                        #print("turno de", curr_loc, windows)

                        # primeira heurística
                        dsrs_1, dsr_windows_1 = self.dsr_threshold_1(windows, boundaries_tier_1, delta1)
                        # segunda heurística
                        dsrs_2 = self.dsr_threshold_2(dsr_windows_1, boundaries_tier_2, delta2, interval_size)

                        # junta todas as fronteiras identificadas pelas duas primeiras heuristicas aplicadas no turno em uma lista
                        timestamps = list(set(dsrs_1 + dsrs_2))
                        print("tamanho dsrs1:", len(dsrs_1))
                        print("tamanho dsrs2:", len(dsrs_2))
                        print("tamanho timestamps:", len(timestamps), "\n")
                        all_timestamps += timestamps

                        # limpa lista de janelas do turno
                        windows = []
                        break

                    print(g2p_words[i], ":", locs_and_words[i].split(';')[0])
                    # se há troca de turno chamamos as heurísticas para as janelas do turno
                    if locs_and_words[i].split(';')[0] != curr_loc:
                        print("turno de", curr_loc, windows)

                        # primeira heurística
                        dsrs_1, dsr_windows_1 = self.dsr_threshold_1(windows, boundaries_tier_1, delta1)
                        # segunda heurística
                        dsrs_2 = self.dsr_threshold_2(dsr_windows_1, boundaries_tier_2, delta2, interval_size)

                        # junta todas as fronteiras identificadas pelas duas primeiras heuristicas aplicadas no turno em uma lista
                        timestamps = list(set(dsrs_1 + dsrs_2))
                        print("tamanho dsrs1:", len(dsrs_1))
                        print("tamanho dsrs2:", len(dsrs_2))
                        print("tamanho timestamps:", len(timestamps), "\n")
                        all_timestamps += timestamps

                        # limpa lista de janelas do turno
                        windows = []

                        if curr_loc in tier_names:
                            loc_tb_tier = tg.get_tier_by_name("TB-"+curr_loc)
                            loc_ntb_tier = tg.get_tier_by_name("NTB-"+curr_loc)
                        else:
                            # Creates TB and NTB tiers for the speaker
                            loc_tb_tier = tgt.core.IntervalTier(start_time=tg.start_time, end_time=tg.end_time, name="TB-"+curr_loc, objects=None)
                            loc_ntb_tier = tgt.core.IntervalTier(start_time=tg.start_time, end_time=tg.end_time, name="NTB-"+curr_loc, objects=None)

                            # Adds the new tiers to the textgrid file
                            tg.add_tier(loc_tb_tier)
                            tg.add_tier(loc_ntb_tier)
                            tier_names.append(curr_loc)                   

                        # Resets the new tiers variables
                        loc_tb_tier = None
                        loc_ntb_tier = None
                        curr_turn = ""

                        # atualiza o tempo de início do próximo intervalo
                        curr_turn_start = interval.start_time
                        print("curr_turn_start", curr_turn_start)
                        # atualiza o começo do tempo do turno atual
                        turn_until_word[i][4] = curr_turn_start
                        last_turn_start = curr_turn_start

                    # atualiza locutor para a proxima palavra
                    curr_loc = locs_and_words[i].split(';')[0]

                    # atualiza o locutor do turno atual, o tempo do início do turno e o texto do turno até a antiga palavra
                    turn_until_word[i][2] = curr_loc

                    # limpamos a string que guarda a palavra sendo construída pelos fonemas
                    constr = ""        
                    # janelas de 300 ms
                    curr_window = [interval.end_time, interval.end_time + window_size]
                    # limpamos a lista de fonemas para a próxima janela
                    window_phones = []

            # terceira heurística
            silences = self.print_silences(sil_file, boundaries_tier_3, silence_threshold)

            # junta todas as fronteiras identificadas das primeiras heuristicas com a terceira
            all_timestamps = list(set(all_timestamps + silences))
        
            # preenche tier de boundaries juntando as 3 heurísticas
            self.fill_boundaries_tier(all_timestamps, boundaries_tier)
            
        #tg.add_tier(boundaries_tier_1)
        #tg.add_tier(boundaries_tier_2)
        #tg.add_tier(boundaries_tier_3)

        tg.add_tier(boundaries_tier)

        last_c = 0
        last_text = ""
        last_loc = turn_until_word[0][2]
        last_b = 0.0
                
        new_intervals = []
        # aqui vamos inserir as informações das fronteiras identificadas pelo método nas tiers correspondentes de cada turno no textgrid
        for b in boundaries_tier.intervals:
            #print("boundary:", b)
            # itera pelas palavras
            for c, t in enumerate(turn_until_word):
                #print("inicio de t:", t[3])
                # se o início da fronteira ocorre após o início da palavra atual, pegue o turno até essa palavra
                if b.start_time <= t[3]:
                    # se trocou de turno zera o último texto
                    if t[2] != last_loc:
                        # quando troca de turno adicionamos o que resta do texto do turno anterior ao intervalo anterior
                        nc = last_c
                        while turn_until_word[nc][2] == last_loc:
                            nc += 1
                        # pega o texto restante do turno anterior após o último trecho de texto para adicionar retroativamente no último intervalo
                        tail_text = turn_until_word[nc-1][1][len(last_text):]
                        #print("turno completo:", turn_until_word[nc-1][1])
                        #print("tail text", tail_text)
                        last_i_updated = [tgt.core.Interval(start_time=last_i[0].start_time, end_time=last_i[0].end_time, text=last_i[0].text+' '+tail_text), last_loc]
                        new_intervals[last_nic] = last_i_updated

                        #print("updated:", last_i_updated[0])

                        # cria intervalo com turno até a fronteira
                        i = [tgt.core.Interval(start_time=last_b, end_time=b.start_time, text=t[1]), t[2]]

                        last_text = ""
                    else:
                        # cria intervalo com turno até a fronteira usando o texto do trecho atual menos o anterior
                        i_text = t[1][len(last_text):]
                        i = [tgt.core.Interval(start_time=last_b, end_time=b.start_time, text=i_text), t[2]]

                    print("adicionando intervalo novo", i)

                    # adiciona intervalo nas duas camadas do turno
                    new_intervals.append(i)
    
                    # salva o turno até a fronteira atual e o locutor            
                    last_text = t[1]
                    last_loc = t[2]
                    last_b = b.start_time
                    last_i = i
                    last_c = c
                    last_nic = len(new_intervals)-1

                    # para de iterar pelas palavras para essa fronteira pois já foi encontrada
                    break

        first = True
        # incluindo o final do texto no textgrid
        if new_intervals[-1][0].end_time != tg.end_time:
            nc = last_c+1
            curr_i = [tgt.core.Interval(start_time=turn_until_word[last_c][4], end_time=turn_until_word[last_c][3], text=turn_until_word[last_c][1]), turn_until_word[last_c][2]]
            while turn_until_word[nc][3] < tg.end_time:
                if turn_until_word[nc][2] != last_loc:
                    if first:
                        curr_i[0].start_time = last_b
                        first = False
                    if curr_i[0].text != "" and curr_i[0].end_time != 0.0:
                            new_intervals.append(curr_i)
                #print("last loop", turn_until_word[nc], "last_loc=", last_loc)
                curr_i = [tgt.core.Interval(start_time=turn_until_word[nc][4], end_time=turn_until_word[nc][3], text=turn_until_word[nc][1]), turn_until_word[nc][2]]
                last_loc = turn_until_word[nc][2]
                nc += 1
                if nc >= len(turn_until_word):
                    try:
                        #print("last i", curr_i)
                        if curr_i[0].text != "" and curr_i[0].end_time != 0.0:
                            new_intervals.append(curr_i)
                    except:
                        print("overlap")
                    break

        # adiciona intervalos nas duas camadas do turno adequado
        for ni in new_intervals:
            tb_turn_tier = tg.get_tier_by_name("TB-"+ni[1])
            ntb_turn_tier =  tg.get_tier_by_name("NTB-"+ni[1])

            tb_turn_tier.add_interval(ni[0])
            ntb_turn_tier.add_interval(ni[0])

        #deleta tier com alinhamento fonético e tier original colapsada com fronteiras do metodo
        tg.delete_tier("labels")
        tg.delete_tier("fronteiras_metodo")

        # adiciona tier para comentários dos anotadores
        comments_tier = tgt.core.IntervalTier(start_time=tg.start_time, end_time=tg.end_time, name="comentarios-anotacao", objects=None)
        tg.add_tier(comments_tier)

        for name in tg.get_tier_names():
            print(name)

        tgt.io.write_to_file(tg, tg_file.replace(".TextGrid", "_novo.TextGrid"), format='long', encoding='utf-8')

    def ser(self, annot_tg, method_tg, boundary_type):
        if boundary_type not in ["TB", "NTB"]:
            print("boundary_type inválido")
            return 0

        Annot_tg = tgt.io.read_textgrid(annot_tg, self.predict_encoding(annot_tg), include_empty_intervals=True)
        Method_tg = tgt.io.read_textgrid(method_tg, self.predict_encoding(method_tg), include_empty_intervals=True)
        print("Method tg", Method_tg)
        
        agreement_tier = tgt.core.IntervalTier(start_time=Annot_tg.start_time, end_time=Annot_tg.end_time, name="concordancia", objects=None)

        names_annot = Annot_tg.get_tier_names()
        names_method = Method_tg.get_tier_names()
        
        method_boundaries = []
        for name in names_method:
            tier = Method_tg.get_tier_by_name(name)
            if "metodo" in name:
                #print("tier", name, "adicionada:", tier)
                for interval in tier.intervals:
                    #print("intervalo do metodo", interval)
                    method_boundaries.append([interval, '0'])
                Annot_tg.add_tier(tier)

        end_flag = False
        I = 0
        R = 0
        for name in names_annot:
            tier = Annot_tg.get_tier_by_name(name)
            if boundary_type in name:
                for interval in tier.intervals:
                    #print(method_boundaries)
                    if interval.start_time < method_boundaries[0][0].start_time or interval.start_time > method_boundaries[-1][0].end_time:
                        continue
                    for mb in method_boundaries:
                        if abs(interval.start_time - mb[0].start_time) < 0.01:
                            mb[1] = '1'
                        else:
                            R += 1
                        if mb[0].end_time == Method_tg.end_time:
                            if abs(interval.end_time - mb[0].end_time) < 0.01:
                                end_flag = True
        
        total = 0
        hits = 0
        for mb in method_boundaries:
            nb = tgt.core.Interval(start_time=mb[0].start_time, end_time=mb[0].end_time, text=mb[1])
            print("concordancia:", nb)
            if mb[1] == '1':
                hits += 1
            if mb[1] == '0':
                I += 1
            total += 1
            agreement_tier.add_interval(nb)

        if end_flag:
            hits += 1
        else:
            I += 1

        C = hits
        SER = (I+R)/(C+R)

        print("acertos:", hits, '/', total, '=', hits/total)
        print("métrica SER:", '(I+R)/(C+R)', SER)

        Annot_tg.add_tier(agreement_tier)
        tgt.io.write_to_file(Annot_tg, annot_tg.replace(".TextGrid", "_concat.TextGrid"), format='long', encoding='utf-8')
        return SER

def main():
    name = "SP_DID_089"
    segment_number = "2"

    path = "./" + name + "_segmentado/" + name + "_" + segment_number + "/"
    audio_file = path + name + "_clipped_" + segment_number + ".wav"
    alignment_tg = path + name + "_clipped_" + segment_number + ".TextGrid"
    locs_file = path + "locutores.txt"
    locs_file2 = path + "locutores_palavras.txt"
    annot_tg = name + "_segmentado/" + name + ".TextGrid"
    text_align = path + "locutores_palavras_align.txt"
    silences_file = path + "new_wavs/silences.txt"
    method_tg = path + name + "_clipped_" + segment_number + "_novo.TextGrid"
    wavs_path = path + "new_wavs/"

    Segmentation = AutomaticSegmentation(path, audio_file, locs_file)
    Segmentation.generate_words_file(locs_file)
    Segmentation.align(audio_file, text_align)
    Segmentation.find_silences("./", path, wavs_path, 22050, 0.3, 10.0, 5.0, 37.0, '', 1)
    Segmentation.find_boundaries(locs_file2, alignment_tg, silences_file, 0.3, 0.88, 0.70, 3, 0.3, 10)
    Segmentation.ser(annot_tg, method_tg, "NTB")

if __name__ == "__main__":
    main()
