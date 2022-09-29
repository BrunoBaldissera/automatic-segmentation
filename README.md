# automatic-segmentation

## Para usar o segmentador sem alterar os parâmetros do código internamente é esperada uma organização dos arquivos de áudio e texto conforme a seguir.

## Vamos simular a execução da ferramenta para o inquérito SP_DID_043 cortado em duas partes SP_DID_043_1 e SP_DID_043_2.

## A partir do diretório principal (clonando o repositório), ponha uma pasta com o nome do inquérito (SP_DID_043) contendo pastas com os segmentos (SP_DID_043_1 e SP_DID_043_2), um arquivo .TextGrid com a transcrição manual ("SP_DID_043.TextGrid") e dentro de cada subpasta de segmentos uma transcrição com arquivo "locutores.txt" e um arquivo com o áudio correspondente ao segmento (SP_DID_043_clipped_1 ou SP_DID_043_clipped_2):

## A organização da pasta se dá dessa forma:

#### automatic-segmentation/
#### automatic-segmentation/SP_DID_043.TextGrid
#### automatic-segmentation/SP_DID_043_1/
#### automatic-segmentation/SP_DID_043_1/SP_DID_043_clipped_1.wav
#### automatic-segmentation/SP_DID_043_1/locutores.txt
#### automatic-segmentation/SP_DID_043_2/
#### automatic-segmentation/SP_DID_043_2/SP_DID_043_clipped_2.wav
#### automatic-segmentation/SP_DID_043_2/locutores.txt

## 1) Agora é preciso editar as variáveis ao fim do arquivo Segmentador.py de acordo com o nome do inquérito e o número do segmento em que foi cortado

#### name = "SP_DID_043"
#### segment_number = "1"

## 2) python Segmentador.py
### Esse script gera uma série de arquivos que são usados internamente pelo método e um arquivo principal de saída (SP_DID_043_clipped_1_novo.TextGrid) com as fronteiras identificadas pelo método
