import math
import re
from collections import defaultdict


class VoynichSlotParser:

  def __init__(self):
    # Expressões regulares para mapeamento dos 16 slots morfológicos (S0 - S15)
    self.slot_patterns = [
        (0, r'^(qok|qot|qof)'),  # S0: Prefixos Primários
        (1, r'^(p|f|m|g)'),  # S1: Modificadores de Galpão/Gallows
        (2, r'^(d|s|y)'),  # S2: Prefixos Secundários
        (3, r'^(a|o)'),  # S3: Vogal de Ligação Inicial
        (4, r'(che|chy|ckh)'),  # S4: Core Relevante A
        (5, r'(sh|sho|shy)'),  # S5: Core Relevante B
        (6, r'(ok|ot|ol)'),  # S6: Núcleo de Estado
        (7, r'(k|t|p|f)'),  # S7: Consoantes Gallows Internas
        (8, r'(otar|oted|otaiin)'),  # S8: Marcadores Cíclicos/Graus
        (9, r'(aiin|aiiin|aiim)'),  # S9: Sequências de Hastes (Minims)
        (10, r'(ar|or|al)'),  # S10: Flexões Internas
        (11, r'(cthy|cpcth)'),  # S11: Extensões Operacionais
        (12, r'(ol|al|l)$'),  # S12: Sufixos de Fluidez/Canais
        (13, r'(ed|edy|ey)$'),  # S13: Terminalizadores de Estado
        (14, r'(dary|dar)$'),  # S14: Sufixos Astronômicos/Fase
        (15, r'(y|am|ain)$'),  # S15: Terminalizador Absoluto
    ]

  def parse(self, token):
    """Decompõe um token EVA em um vetor estruturado de 16 slots."""
    vector = [''] * 16
    work_token = token
    for slot_idx, pattern in self.slot_patterns:
      match = re.search(pattern, work_token)
      if match:
        vector[slot_idx] = match.group(0)
    return vector


class CategoricalNaiveBayesVoynich:

  def __init__(self, alpha=1.0):
    self.alpha = alpha  # Suavização de Laplace
    self.parser = VoynichSlotParser()
    self.domains = [
        'Botanical',
        'Pharmaceutical',
        'Balneological',
        'Astronomical',
    ]
    self.slot_counts = {
        d: [defaultdict(int) for _ in range(16)] for d in self.domains
    }
    self.domain_totals = {d: 0 for d in self.domains}

  def train(self, dataset):
    """Treina as matrizes de frequência por slot a partir do corpus."""
    for domain, tokens in dataset.items():
      for token in tokens:
        slots = self.parser.parse(token)
        self.domain_totals[domain] += 1
        for i, val in enumerate(slots):
          if val:
            self.slot_counts[domain][i][val] += 1

  def evaluate_sequence(self, tokens):
    """Calcula Log-Likelihood, Probabilidades Posteriores e Entropia de Shannon."""
    log_priors = {d: math.log(1.0 / len(self.domains)) for d in self.domains}
    log_scores = {d: log_priors[d] for d in self.domains}

    for domain in self.domains:
      N_domain = self.domain_totals[domain]
      for token in tokens:
        slots = self.parser.parse(token)
        for i, val in enumerate(slots):
          val_count = self.slot_counts[domain][i].get(val, 0)
          # Cardinalidade do vocabulário do slot
          vocab_size = (
              max(len(self.slot_counts[d][i]) for d in self.domains) or 1
          )

          # Formulação de Laplace: P(x_i | C_m) = (count + alpha) / (N + alpha * |A_i|)
          p_xi = (val_count + self.alpha) / (
              N_domain + self.alpha * vocab_size
          )
          log_scores[domain] += math.log(p_xi)

    # Estabilidade numérica: Subtração do valor máximo antes da exponenciação
    max_log = max(log_scores.values())
    unnorm_probs = {d: math.exp(log_scores[d] - max_log) for d in self.domains}
    total_mass = sum(unnorm_probs.values())
    posteriors = {d: unnorm_probs[d] / total_mass for d in self.domains}

    # Entropia Normalizada de Shannon: H = -sum(p * log2(p)) / log2(|C|)
    H = -sum(
        p * math.log2(p) for p in posteriors.values() if p > 0
    ) / math.log2(len(self.domains))

    return log_scores, posteriors, H


# ==============================================================================
# EXECUÇÃO DO TESTE NO NOVO FOLIO: f84r (Anatômico / Balneológico)
# ==============================================================================
if __name__ == '__main__':
  # Base de Treinamento (Amostra dos Folios f33v, f102v2, f78r, f72r1)
  corpus_train = {
      'Botanical': [
          'qokedy',
          'qokain',
          'qoky',
          'qokain',
          'chey',
          'qokaiin',
          'daraio',
          'qokey',
      ],
      'Pharmaceutical': [
          'checthy',
          'chey',
          'qokchey',
          'cheol',
          'qokcal',
          'checkhy',
          'chedy',
      ],
      'Balneological': [
          'qokol',
          'schol',
          'sody',
          'shesol',
          'qokal',
          'cheol',
          'qoksheol',
          'ykeol',
      ],
      'Astronomical': [
          'otardy',
          'otarak',
          'qokotar',
          'chedary',
          'otara',
          'dary',
          'otaiin',
      ],
  }

  # Instancia e Treina o Modelo
  model = CategoricalNaiveBayesVoynich(alpha=1.0)
  model.train(corpus_train)

  # Amostra do Folio f84r (Seção Anatômica/Dutos)
  tokens_f84r = [
      'qokol',
      'sheol',
      'cheol',
      'sody',
      'qokedy',
      'schol',
      'qokal',
      'ykeol',
  ]

  log_scores, posteriors, H = model.evaluate_sequence(tokens_f84r)

  print('=== RESULTADOS DA VALIDAÇÃO (FOLIO f84r) ===')
  for domain in model.domains:
    print(
        f'Domínio: {domain:15s} | Log-Likelihood: {log_scores[domain]:8.2f} |'
        f' Prob. Posterior: {posteriors[domain]*100:6.2f}%'
    )
  print(f'\nEntropia Normalizada de Shannon (H): {H:.4f}')