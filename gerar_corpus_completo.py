import json
import re
import urllib.request

# Espelhos permanentes do corpus Takahashi EVA completo (37.308 palavras)
URLS_CORPUS = [
    (
        'https://huggingface.co/AbstractPhil/sbert-voynich-translation/resolve/main/voynich_raw.txt'
    ),
    (
        'https://raw.githubusercontent.com/Oshiruko3/voynich-decipherment/main/data/voynich_raw.txt'
    ),
    (
        'https://raw.githubusercontent.com/voynich-data/eva/main/landini-takahashi.txt'
    ),
]

# Mapeamento padrão dos intervalos de fólios por secção do Voynich
SECTION_MAPPING = {
    'Botanical': range(1, 67),
    'Astronomical': range(67, 74),
    'Biological/Balneological': range(75, 85),
    'Cosmological': range(85, 87),
    'Pharmaceutical': range(87, 103),
    'Recipes/Text': range(103, 117),
}


def get_section_by_folio(folio_num):
  for section, folio_range in SECTION_MAPPING.items():
    if folio_num in folio_range:
      return section
  return 'Recipes/Text'


def download_corpus_lines():
  headers = {
      'User-Agent': (
          'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
          ' (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
      ),
      'Accept': 'text/plain,text/html,application/xhtml+xml,application/xml',
  }

  for url in URLS_CORPUS:
    try:
      print(f'Tentando baixar transcrição de:\n -> {url}')
      req = urllib.request.Request(url, headers=headers)
      with urllib.request.urlopen(req, timeout=20) as response:
        content = response.read().decode('utf-8', errors='ignore')
        lines = content.splitlines()
        if len(lines) > 100:
          print('✅ Download concluído com sucesso!')
          return lines
    except Exception as e:
      print(f'⚠️ Falha nesta fonte ({e}). Tentando o próximo espelho...\n')

  return None


def build_full_voynich_json():
  lines = download_corpus_lines()
  if not lines:
    print(
        '❌ Erro: Não foi possível conectar a nenhuma fonte online do Voynich.'
    )
    return

  corpus = {}

  # Regex para capturar marcações de fólio no formato EVA (ex: <f1r>, <f102v2>, f1r.P.1, etc.)
  line_pattern = re.compile(
      r'^(?:<)?f(\d+)([rv]\d*)?(?:\.[^>]+)?(?:>)?[\s,:\t]+(.*)$', re.IGNORECASE
  )

  print('Processando as 240 páginas e tokenizando a estrutura EVA...')

  for line in lines:
    line = line.strip()
    if not line or line.startswith('#'):
      continue

    match = line_pattern.match(line)
    if match:
      folio_num = int(match.group(1))
      side_suffix = match.group(2) or ''
      folio_id = f'f{folio_num}{side_suffix}'
      text_content = match.group(3)

      # Trata conectores e pontuações do padrão EVA
      text_clean = text_content.replace('.', ' ')
      text_clean = re.sub(r'\{[^}]*\}', '', text_clean)
      text_clean = re.sub(r'[!?.,\-+=*\$@<>%_;"\']', ' ', text_clean)

      tokens = [
          t.lower().strip()
          for t in text_clean.split()
          if t.isalpha() and len(t) > 1
      ]

      if not tokens:
        continue

      section = get_section_by_folio(folio_num)

      if section not in corpus:
        corpus[section] = {}

      if folio_id not in corpus[section]:
        corpus[section][folio_id] = []

      corpus[section][folio_id].extend(tokens)
    else:
      # Tratamento alternativo para arquivos ordenados por linhas diretas de texto
      raw_tokens = re.findall(r'\b[a-zA-Z]{2,}\b', line)
      if raw_tokens and 'Unassigned' in corpus:
        corpus['Unassigned']['f0_general'].extend(
            [t.lower() for t in raw_tokens]
        )

  # Salvando o arquivo local voyager
  output_filename = 'voynich_corpus.json'
  with open(output_filename, 'w', encoding='utf-8') as f:
    json.dump(corpus, f, indent=2, ensure_ascii=False)

  total_folios = sum(len(f) for f in corpus.values())
  total_tokens = sum(
      len(tokens) for f in corpus.values() for tokens in f.values()
  )

  print(f'\n🎉 Sucesso! Arquivo "{output_filename}" gerado localmente com:')
  print(f'   - {len(corpus)} Secções/Domínios')
  print(f'   - {total_folios} Fólios/Páginas mapeados')
  print(f'   - {total_tokens:,} Tokens morfológicos extraídos')


if __name__ == '__main__':
  build_full_voynich_json()