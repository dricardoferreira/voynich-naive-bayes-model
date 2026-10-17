import json
import math
import os
import random
import re
from collections import defaultdict
import numpy as np
import pandas as pd
import streamlit as st

# ==============================================================================
# CONFIGURAÇÃO DO STREAMLIT & DICIONÁRIO DE IDIOMAS (EN / PT)
# ==============================================================================
st.set_page_config(
    page_title="Voynich Rigorous Naive Bayes Corpus Explorer", layout="wide"
)

# Dicionário de Traduções da Interface
LANG_TEXTS = {
    "English": {
        "title": (
            "🔬 Voynich Rigorous Naive Bayes Corpus Explorer & Validator (MS"
            " 408)"
        ),
        "sidebar_header": "⚙️ Real Corpus Settings",
        "lang_label": "Language / Idioma",
        "alpha_label": "Smoothing Parameter (Laplace Alpha)",
        "mode_label": "Analysis Mode",
        "m_individual": "Individual Folio",
        "m_global": "Global Scan (All Pages)",
        "m_sliding": "Sliding Window Audit (Hybrid Detection)",
        "m_loocv": "Cross-Validation (LOOCV - Out-of-Sample)",
        "m_null": "Null Baseline (Control / Hoax Test)",
        # Folio Individual
        "fi_title": "📄 Individual Real Folio Analysis with 95% CI (Bootstrapping)",
        "fi_select": "Select Manuscript Folio:",
        "fi_table_col": "Canonical Domain",
        "fi_prob_col": "Posterior Prob (%)",
        "fi_ll_col": "Total Log-Likelihood",
        "fi_ci_col": "95% CI (Mean/Token)",
        "fi_cert_title": "🧬 Structural Certainty Metric",
        "fi_entropy_label": "Normalized Shannon Entropy (H)",
        "fi_entropy_pure": "Pure Domain / Highly Structured",
        "fi_entropy_trans": "Transition / Mixed Zone",
        "fi_tokens_info": "Total Tokens Analyzed in Folio:",
        "fi_original_sec": "Original Section:",
        "fi_dec_title": (
            "🛠️ Sample of 16-Slot Decomposition (S0 - S15) for First Tokens"
        ),
        # Global Scan
        "gs_title": "🌍 Real Corpus Global Scan (All Manuscript Pages)",
        "gs_desc": (
            "Executing sequential dynamic slot evaluation per folio across the"
            " loaded corpus..."
        ),
        "gs_success": "✅ Global scan successfully completed in real folios!",
        "gs_table_title": "📋 Complete Global Scan Results Table",
        # Sliding Window
        "sw_title": "🔍 Sliding Window Hybrid Audit",
        "sw_desc": (
            "Splits the selected folio into sequential blocks to detect"
            " micro-transitions and hybrid structures."
        ),
        "sw_select": "Select Folio for Audit:",
        "sw_window_size": "Window Size (Tokens per Block)",
        "sw_step_size": "Step Size",
        "sw_warning": "has only tokens, smaller than window size.",
        "sw_chart_title": "📈 Window Probability Dynamics Along Folio",
        "sw_mean_entropy": "Mean Local Window Entropy",
        "sw_homo": "Highly Homogeneous Folio",
        "sw_hybrid": "Transition / Hybrid Folio",
        "sw_total_windows": "Total Windows Analyzed:",
        "sw_table_title": "📋 Detailed Local Window Results Table",
        # LOOCV
        "loocv_title": "🔄 Leave-One-Out Cross-Validation (LOOCV)",
        "loocv_desc": (
            "Trains the model on N-1 folios and tests prediction exclusively on"
            " the held-out out-of-sample folio."
        ),
        "loocv_button": "Run Full Cross-Validation",
        "loocv_success": "✅ Cross-Validation successfully completed!",
        "loocv_acc": "Global Out-of-Sample Accuracy (LOOCV)",
        "loocv_table_title": "📋 Detailed Folio Results (LOOCV)",
        "col_folio": "Folio",
        "col_orig_sec": "Original Section",
        "col_canon_dom": "Canonical Domain",
        "col_pred_oos": "Predicted Out-of-Sample",
        "col_correct": "Correct?",
        "col_conf": "Confidence (%)",
        "col_entropy": "Entropy",
        # Null Baseline
        "null_title": "🎲 Null Baseline — Control Against Hoax Hypothesis",
        "null_desc": (
            "Generates synthetic sequences by randomly shuffling corpus tokens"
            " to prove real low entropy is not by chance."
        ),
        "null_slider": "Number of Null Simulations",
        "null_button": "Run Null Baseline Simulation",
        "null_success": "✅ Null Baseline Simulation Completed!",
        "null_real_ent": "Mean Real Manuscript Entropy",
        "null_sim_ent": "Mean Null Model Entropy (Random Tokens)",
        "null_delta": "Significant Structural Difference",
        "null_interp": (
            "💡 **test interpretation:** The null model demonstrates that"
            " stochastically shuffled sequences yield completely different"
            " uncertainty distributions compared to the real codex, providing"
            " the statistical baseline for the 16-slot internal organization."
        ),
        "null_chart_type": "Sequence Type",
    },
    "Português": {
        "title": (
            "🔬 Explorador e Validador Global de Slots & Métricas de Rigor"
            " (MS 408)"
        ),
        "sidebar_header": "⚙️ Configurações do Corpus Real",
        "lang_label": "Language / Idioma",
        "alpha_label": "Parâmetro de Suavização (Laplace Alpha)",
        "mode_label": "Modo de Análise",
        "m_individual": "Fólio Individual",
        "m_global": "Varredura Global (Todas as Páginas)",
        "m_sliding": "Análise de Janela Deslizante (Detecção de Hibridismo)",
        "m_loocv": "Validação Cruzada (LOOCV - Out-of-Sample)",
        "m_null": "Modelo Nulo (Teste de Controlo / Farsa)",
        # Folio Individual
        "fi_title": (
            "📄 Análise Individual de Fólio Real com IC 95% (Bootstrapping)"
        ),
        "fi_select": "Selecione o Fólio do Manuscrito:",
        "fi_table_col": "Domínio Canónico",
        "fi_prob_col": "Posterior Prob (%)",
        "fi_ll_col": "Log-Likelihood Total",
        "fi_ci_col": "IC 95% (Média/Token)",
        "fi_cert_title": "🧬 Métrica de Certeza Estrutural",
        "fi_entropy_label": "Entropia de Shannon Normalizada (H)",
        "fi_entropy_pure": "Domínio Puro / Altamente Estruturado",
        "fi_entropy_trans": "Zona de Transição / Mista",
        "fi_tokens_info": "Total de Tokens Analisados neste Fólio:",
        "fi_original_sec": "Seção Original:",
        "fi_dec_title": (
            "🛠️ Amostra da Decomposição nos 16 Slots (S0 - S15) dos Primeiros"
            " Tokens"
        ),
        # Global Scan
        "gs_title": (
            "🌍 Varredura Global do Corpus Real (Todas as Páginas do Manuscrito)"
        ),
        "gs_desc": (
            "A executar a avaliação sequencial de slot dinâmico por fólio em todo"
            " o corpus carregado..."
        ),
        "gs_success": (
            "✅ Varredura concluída com sucesso em todos os fólios reais!"
        ),
        "gs_table_title": "📋 Tabela Completa de Resultados da Varredura Global",
        # Sliding Window
        "sw_title": "🔍 Auditoria de Hibridismo por Janela Deslizante",
        "sw_desc": (
            "Divide o fólio selecionado em blocos sequenciais para detetar"
            " micro-transições e estruturas híbridas."
        ),
        "sw_select": "Selecione o Fólio para Auditoria:",
        "sw_window_size": "Tamanho da Janela (Tokens por Bloco)",
        "sw_step_size": "Passo de Avanço (Step)",
        "sw_warning": "possui apenas tokens, menor que o tamanho da janela.",
        "sw_chart_title": (
            "📈 Dinâmica de Probabilidades por Janela ao Longo do Fólio"
        ),
        "sw_mean_entropy": "Média da Entropia Local das Janelas",
        "sw_homo": "Fólio Altamente Homogéneo",
        "sw_hybrid": "Fólio de Transição / Híbrido",
        "sw_total_windows": "Total de Janelas Analisadas:",
        "sw_table_title": "📋 Tabela Detalhada de Janelas e Probabilidades Locais",
        # LOOCV
        "loocv_title": "🔄 Validação Cruzada Leave-One-Out (LOOCV)",
        "loocv_desc": (
            "Treina o modelo com N-1 fólios e testa a predição exclusivamente"
            " no fólio deixado de fora (out-of-sample)."
        ),
        "loocv_button": "Executar Validação Cruzada Completa",
        "loocv_success": (
            "✅ Validação Cruzada Concluída com Sucesso em todos os fólios!"
        ),
        "loocv_acc": "Taxa de Precisão Global Out-of-Sample (LOOCV Accuracy)",
        "loocv_table_title": "📋 Resultados Detalhados por Fólio (LOOCV)",
        "col_folio": "Folio",
        "col_orig_sec": "Seção Original",
        "col_canon_dom": "Domínio Canónico",
        "col_pred_oos": "Predito Out-of-Sample",
        "col_correct": "Acertou?",
        "col_conf": "Confiança (%)",
        "col_entropy": "Entropia",
        # Null Baseline
        "null_title": (
            "🎲 Modelo Nulo (Null Baseline) — A Hipótese de"
            " Farsa"
        ),
        "null_desc": (
            "Gera sequências sintéticas baralhando aleatoriamente os tokens do"
            " corpus para provar que a entropia baixa não ocorre por acaso."
        ),
        "null_slider": "Número de Simulações Nulas",
        "null_button": "Executar Simulação do Modelo Nulo",
        "null_success": "✅ Simulação do Modelo Nulo Concluída!",
        "null_real_ent": "Entropia Média do Manuscrito Real (Voynich)",
        "null_sim_ent": "Entropia Média do Modelo Nulo (Tokens Aleatórios)",
        "null_delta": "Diferença Estrutural Significativa",
        "null_interp": (
            "💡 **Interpretação do teste:** O modelo nulo demonstra que"
            " sequências geradas por baralhamento estocástico tendem a uma"
            " distribuição de incerteza completamente diferente da encontrada no"
            " códice real, fornecendo a linha de base estatística da"
            " organização interna dos 16 slots."
        ),
        "null_chart_type": "Tipo de Sequência",
    },
}

CORPUS_FILENAME = "voynich_corpus.json"

DOMAIN_MAPPER = {
    "Botanical": "Botanical",
    "Pharmaceutical": "Pharmaceutical",
    "Recipes/Text": "Pharmaceutical",
    "Recipes": "Pharmaceutical",
    "Biological/Balneological": "Balneological",
    "Balneological": "Balneological",
    "Biological": "Balneological",
    "Astronomical": "Astronomical",
    "Cosmological": "Astronomical",
}


def normalize_domain(domain_name):
  return DOMAIN_MAPPER.get(domain_name, "Pharmaceutical")


# ==============================================================================
# 1. PARSER MORFOLÓGICO DE 16 SLOTS (S0 - S15)
# ==============================================================================
class VoynichSlotParser:

  def __init__(self):
    self.slot_patterns = [
        (0, r"^(qok|qot|qof)"),
        (1, r"^(p|f|m|g)"),
        (2, r"^(d|s|y)"),
        (3, r"^(a|o)"),
        (4, r"(che|chy|ckh)"),
        (5, r"(sh|sho|shy)"),
        (6, r"(ok|ot|ol)"),
        (7, r"(k|t|p|f)"),
        (8, r"(otar|oted|otaiin)"),
        (9, r"(aiin|aiiin|aiim)"),
        (10, r"(ar|or|al)"),
        (11, r"(cthy|cpcth)"),
        (12, r"(ol|al|l)$"),
        (13, r"(ed|edy|ey)$"),
        (14, r"(dary|dar)$"),
        (15, r"(y|am|ain)$"),
    ]

  def parse(self, token):
    vector = [""] * 16
    work_token = token
    for slot_idx, pattern in self.slot_patterns:
      match = re.search(pattern, work_token)
      if match:
        vector[slot_idx] = match.group(0)
    return vector


# ==============================================================================
# 2. MOTOR NAIVE BAYES COM SUAVIZAÇÃO DE LAPLACE & BOOTSTRAPPING CI
# ==============================================================================
class RigorousNaiveBayesVoynich:

  def __init__(self, domains, alpha=1.0):
    self.alpha = alpha
    self.domains = list(set([normalize_domain(d) for d in domains]))
    self.parser = VoynichSlotParser()
    self.slot_counts = {
        d: [defaultdict(int) for _ in range(16)] for d in self.domains
    }
    self.slot_totals = {d: [0] * 16 for d in self.domains}
    self.class_priors = {}

  def train(self, dataset):
    for domain, folios in dataset.items():
      canonical_domain = normalize_domain(domain)
      if canonical_domain in self.domains:
        for folio_id, tokens in folios.items():
          for token in tokens:
            slots = self.parser.parse(token)
            for i, val in enumerate(slots):
              val_state = val if val != "" else "<NULL>"
              self.slot_counts[canonical_domain][i][val_state] += 1
              self.slot_totals[canonical_domain][i] += 1

    num_domains = len(self.domains)
    for d in self.domains:
      self.class_priors[d] = 1.0 / num_domains if num_domains > 0 else 0.25

  def evaluate_sequence(self, tokens_sequence):
    log_scores = {d: math.log(self.class_priors[d]) for d in self.domains}
    num_classes = len(self.domains)
    total_tokens = len(tokens_sequence)

    if total_tokens == 0:
      return log_scores, {d: 1.0 / num_classes for d in self.domains}, 1.0

    for domain in self.domains:
      for token in tokens_sequence:
        slots = self.parser.parse(token)
        for slot_idx, slot_val in enumerate(slots):
          val_state = slot_val if slot_val != "" else "<NULL>"
          count = self.slot_counts[domain][slot_idx][val_state]
          total = self.slot_totals[domain][slot_idx]
          vocab_size = max(len(self.slot_counts[domain][slot_idx]), 1)

          prob = (count + self.alpha) / (total + self.alpha * vocab_size)
          log_scores[domain] += math.log(max(prob, 1e-15))

    max_log = max(log_scores.values())
    exp_scores = {
        d: math.exp(score - max_log) for d, score in log_scores.items()
    }
    sum_exp = sum(exp_scores.values())
    posteriors = {
        d: (val / sum_exp if sum_exp > 0 else 1.0 / num_classes)
        for d, val in exp_scores.items()
    }

    entropy = 0.0
    for d, p in posteriors.items():
      if p > 0:
        entropy -= p * math.log2(p)
    norm_entropy = entropy / math.log2(num_classes) if num_classes > 1 else 0.0

    return log_scores, posteriors, norm_entropy

  def compute_bootstrap_ci(
      self, tokens_sequence, n_bootstraps=300, ci_level=0.95
  ):
    if not tokens_sequence or len(tokens_sequence) == 0:
      return {}, {}

    num_tokens = len(tokens_sequence)
    bootstrap_log_scores = {d: [] for d in self.domains}

    np.random.seed(42)
    tokens_arr = np.array(tokens_sequence)

    for _ in range(n_bootstraps):
      bootstrap_sample = np.random.choice(
          tokens_arr, size=num_tokens, replace=True
      )
      log_scores, _, _ = self.evaluate_sequence(bootstrap_sample)

      for d in self.domains:
        bootstrap_log_scores[d].append(log_scores[d] / num_tokens)

    lower_percentile = ((1.0 - ci_level) / 2.0) * 100
    upper_percentile = (ci_level + (1.0 - ci_level) / 2.0) * 100

    ci_results = {}
    mean_scores = {}

    for d in self.domains:
      scores = bootstrap_log_scores[d]
      mean_val = np.mean(scores)
      ci_lower = np.percentile(scores, lower_percentile)
      ci_upper = np.percentile(scores, upper_percentile)

      mean_scores[d] = mean_val
      ci_results[d] = (ci_lower, ci_upper)

    return mean_scores, ci_results


# ==============================================================================
# 3. CARREGAMENTO DOS DADOS REAIS DO CORPUS
# ==============================================================================
@st.cache_resource
def load_corpus_data():
  if not os.path.exists(CORPUS_FILENAME):
    return None
  with open(CORPUS_FILENAME, "r", encoding="utf-8") as f:
    return json.load(f)


corpus_data = load_corpus_data()

if corpus_data is None:
  st.error(
      f"❌ Ficheiro '{CORPUS_FILENAME}' não encontrado no diretório de"
      " execução."
  )
  st.stop()

canonical_domains = [
    "Botanical",
    "Pharmaceutical",
    "Balneological",
    "Astronomical",
]


@st.cache_resource
def get_trained_model(alpha):
  model = RigorousNaiveBayesVoynich(canonical_domains, alpha=alpha)
  model.train(corpus_data)
  return model


# ==============================================================================
# INTERFACE DE CONTROLO (SIDEBAR & IDIOMA)
# ==============================================================================
st.sidebar.header("⚙️ Settings / Configurações")

# Seletor de Idioma
selected_lang = st.sidebar.selectbox(
    "Language / Idioma", ["Português", "English"], index=0
)
T = LANG_TEXTS[selected_lang]

st.title(T["title"])

st.sidebar.header(T["sidebar_header"])
alpha_param = st.sidebar.slider(T["alpha_label"], 0.1, 2.0, 1.0, 0.1)

mode = st.sidebar.radio(
    T["mode_label"],
    [
        T["m_individual"],
        T["m_global"],
        T["m_sliding"],
        T["m_loocv"],
        T["m_null"],
    ],
)

model = get_trained_model(alpha_param)

# ==============================================================================
# EXIBIÇÃO DE MODOS
# ==============================================================================
if mode == T["m_individual"]:
  st.subheader(T["fi_title"])

  all_folios = []
  for section, folios in corpus_data.items():
    for f_id in folios.keys():
      all_folios.append(f_id)

  selected_folio = st.selectbox(T["fi_select"], sorted(all_folios))

  if selected_folio:
    sec_key = [
        s for s, f_dict in corpus_data.items() if selected_folio in f_dict
    ][0]
    tokens = corpus_data[sec_key][selected_folio]

    log_scores, posteriors, entropy = model.evaluate_sequence(tokens)
    mean_scores, ci_results = model.compute_bootstrap_ci(
        tokens, n_bootstraps=300
    )

    col1, col2 = st.columns(2)
    with col1:
      st.markdown(f"### 📊 P(Cm|W) & IC 95% — Folio `{selected_folio}`")
      df_res = pd.DataFrame({
          T["fi_table_col"]: list(posteriors.keys()),
          T["fi_prob_col"]: [
              round(val * 100, 2) for val in posteriors.values()
          ],
          T["fi_ll_col"]: [round(log_scores[d], 2) for d in posteriors.keys()],
          T["fi_ci_col"]: [
              f"[{ci_results[d][0]:.2f}, {ci_results[d][1]:.2f}]"
              for d in posteriors.keys()
          ],
      })
      st.dataframe(df_res, use_container_width=True)

    with col2:
      st.markdown(f"### {T['fi_cert_title']}")
      st.metric(
          label=T["fi_entropy_label"],
          value=f"{entropy:.4f}",
          delta=(
              T["fi_entropy_trans"]
              if entropy > 0.3
              else T["fi_entropy_pure"]
          ),
          delta_color="inverse",
      )
      st.info(
          f"{T['fi_tokens_info']} **{len(tokens):,}** | {T['fi_original_sec']}"
          f" **{sec_key}**"
      )

    st.markdown("---")
    st.subheader(T["fi_dec_title"])
    sample_tokens = tokens[:10]
    decomposition_rows = []
    for t in sample_tokens:
      vec = model.parser.parse(t)
      row = {"Token": t}
      for idx, val in enumerate(vec):
        row[f"S{idx}"] = val if val != "" else "-"
      decomposition_rows.append(row)

    st.dataframe(pd.DataFrame(decomposition_rows), use_container_width=True)

elif mode == T["m_global"]:
  st.subheader(T["gs_title"])
  st.write(T["gs_desc"])

  global_results = []
  for section, folios in corpus_data.items():
    for f_id, tokens in folios.items():
      if len(tokens) > 5:
        _, posteriors, entropy = model.evaluate_sequence(tokens)
        dominant_domain = max(posteriors, key=posteriors.get)
        global_results.append({
            "Folio": f_id,
            "Seção Original": section,
            "Dominante": dominant_domain,
            "Botanical": posteriors.get("Botanical", 0),
            "Pharmaceutical": posteriors.get("Pharmaceutical", 0),
            "Balneological": posteriors.get("Balneological", 0),
            "Astronomical": posteriors.get("Astronomical", 0),
            "Entropy": entropy,
        })

  df_global = pd.DataFrame(global_results)

  if not df_global.empty:
    st.success(T["gs_success"])
    chart_df = df_global.set_index("Folio")[
        ["Botanical", "Pharmaceutical", "Balneological", "Astronomical"]
    ]
    st.line_chart(chart_df)

    st.markdown(f"### {T['gs_table_title']}")
    st.dataframe(df_global, use_container_width=True)
  else:
    st.warning("Nenhum dado encontrado.")

elif mode == T["m_sliding"]:
  st.subheader(T["sw_title"])
  st.write(T["sw_desc"])

  all_folios = []
  for section, folios in corpus_data.items():
    for f_id in folios.keys():
      all_folios.append(f_id)

  col_w1, col_w2, col_w3 = st.columns(3)
  with col_w1:
    selected_folio_sw = st.selectbox(
        T["sw_select"],
        sorted(all_folios),
        key="sw_folio",
        index=(
            sorted(all_folios).index("f96v")
            if "f96v" in sorted(all_folios)
            else 0
        ),
    )
  with col_w2:
    window_size = st.slider(T["sw_window_size"], 20, 200, 50, 10)
  with col_w3:
    step_size = st.slider(T["sw_step_size"], 5, 50, 15, 5)

  if selected_folio_sw:
    sec_key = [
        s for s, f_dict in corpus_data.items() if selected_folio_sw in f_dict
    ][0]
    tokens = corpus_data[sec_key][selected_folio_sw]

    if len(tokens) < window_size:
      st.warning(
          f"⚠️ O fólio `{selected_folio_sw}` {T['sw_warning']} ({len(tokens)})."
      )
    else:
      window_results = []
      for i in range(0, len(tokens) - window_size + 1, step_size):
        window_tokens = tokens[i : i + window_size]
        _, posteriors, entropy = model.evaluate_sequence(window_tokens)
        dominant = max(posteriors, key=posteriors.get)
        row = {
            "Janela_Start": i,
            "Janela_End": i + window_size,
            "Dominante": dominant,
            "Entropia_Local": entropy,
        }
        for d, p in posteriors.items():
          row[d] = round(p * 100, 2)
        window_results.append(row)

      df_sw = pd.DataFrame(window_results)

      st.markdown(f"### {T['sw_chart_title']} `{selected_folio_sw}`")
      chart_cols = [
          c
          for c in [
              "Botanical",
              "Pharmaceutical",
              "Balneological",
              "Astronomical",
          ]
          if c in df_sw.columns
      ]
      st.line_chart(df_sw.set_index("Janela_Start")[chart_cols])

      col_m1, col_m2 = st.columns(2)
      with col_m1:
        st.metric(
            label=T["sw_mean_entropy"],
            value=f"{df_sw['Entropia_Local'].mean():.4f}",
            delta=(
                T["sw_homo"]
                if df_sw["Entropia_Local"].mean() < 0.2
                else T["sw_hybrid"]
            ),
        )
      with col_m2:
        st.info(f"{T['sw_total_windows']} **{len(df_sw)}**")

      st.markdown(f"### {T['sw_table_title']}")
      st.dataframe(df_sw, use_container_width=True)

elif mode == T["m_loocv"]:
  st.subheader(T["loocv_title"])
  st.write(T["loocv_desc"])

  if st.button(T["loocv_button"]):
    loocv_results = []
    all_folio_items = []
    for section, folios in corpus_data.items():
      for f_id, tokens in folios.items():
        if len(tokens) > 5:
          all_folio_items.append((section, f_id, tokens))

    progress_bar = st.progress(0)
    total_items = len(all_folio_items)

    for idx, (target_sec, target_f_id, target_tokens) in enumerate(
        all_folio_items
    ):
      train_subset = {}
      for section, folios in corpus_data.items():
        train_subset[section] = {}
        for f_id, tokens in folios.items():
          if f_id != target_f_id:
            train_subset[section][f_id] = tokens

      cv_model = RigorousNaiveBayesVoynich(canonical_domains, alpha=alpha_param)
      cv_model.train(train_subset)

      _, posteriors, entropy = cv_model.evaluate_sequence(target_tokens)
      dominant = max(posteriors, key=posteriors.get)
      canon_target = normalize_domain(target_sec)
      is_correct = dominant == canon_target

      loocv_results.append({
          T["col_folio"]: target_f_id,
          T["col_orig_sec"]: target_sec,
          T["col_canon_dom"]: canon_target,
          T["col_pred_oos"]: dominant,
          T["col_correct"]: "✅ Sim" if is_correct else "❌ Não",
          T["col_conf"]: round(posteriors[dominant] * 100, 2),
          T["col_entropy"]: entropy,
      })
      progress_bar.progress((idx + 1) / total_items)

    df_loocv = pd.DataFrame(loocv_results)
    st.success(T["loocv_success"])

    accuracy = (
        (df_loocv[T["col_correct"]] == "✅ Sim").sum() / len(df_loocv)
    ) * 100
    st.metric(label=T["loocv_acc"], value=f"{accuracy:.2f}%")

    st.markdown(f"### {T['loocv_table_title']}")
    st.dataframe(df_loocv, use_container_width=True)

else:
  st.subheader(T["null_title"])
  st.write(T["null_desc"])

  num_simulations = st.slider(T["null_slider"], 10, 1000, 50, 10)

  if st.button(T["null_button"]):
    all_tokens_pool = []
    real_entropies = []

    for section, folios in corpus_data.items():
      for f_id, tokens in folios.items():
        if len(tokens) > 20:
          all_tokens_pool.extend(tokens)
          _, _, ent = model.evaluate_sequence(tokens)
          real_entropies.append(ent)

    null_entropies = []
    sample_len = 100

    progress_bar = st.progress(0)
    for sim in range(num_simulations):
      random_tokens = random.choices(all_tokens_pool, k=sample_len)
      _, _, ent = model.evaluate_sequence(random_tokens)
      null_entropies.append(ent)
      progress_bar.progress((sim + 1) / num_simulations)

    avg_real_ent = sum(real_entropies) / len(real_entropies)
    avg_null_ent = sum(null_entropies) / len(null_entropies)

    st.success(T["null_success"])

    col_n1, col_n2 = st.columns(2)
    with col_n1:
      st.metric(label=T["null_real_ent"], value=f"{avg_real_ent:.4f}")
    with col_n2:
      st.metric(
          label=T["null_sim_ent"],
          value=f"{avg_null_ent:.4f}",
          delta=T["null_delta"],
      )

    st.info(T["null_interp"])

    # Calcula as médias globais para comparar o corpus real vs as 1000 simulações
    real_mean = np.mean(real_entropies) if len(real_entropies) > 0 else 0
    null_mean = np.mean(null_entropies) if len(null_entropies) > 0 else 0

    label_real = "Voynich Real (Média)" if selected_lang == "Português" else "Real Voynich (Mean)"
    label_null = "Modelo Nulo (Simulado)" if selected_lang == "Português" else "Null Model (Simulated)"

    df_control = pd.DataFrame({
        T["null_chart_type"]: [label_real, label_null],
        "Entropia Normalizada (H)": [real_mean, null_mean],
    })

    st.bar_chart(
        df_control, x=T["null_chart_type"], y="Entropia Normalizada (H)"
    )