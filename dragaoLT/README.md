# 🐉 DragaoLT — Modelo de Linguagem com Mistura de Especialistas

<div align="center">

**Autor: Luiz Tiago Wilcke**

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org)
[![Triton](https://img.shields.io/badge/Triton-3.0+-purple)](https://triton-lang.org)
[![Licença](https://img.shields.io/badge/Licença-MIT-green)](LICENCA.md)

</div>

---

## 📋 Sumário

1. [Introdução](#1-introdução)
2. [Arquitetura](#2-arquitetura)
3. [Inovações Técnicas](#3-inovações-técnicas)
4. [Modelos Disponíveis](#4-modelos-disponíveis)
5. [Instalação](#5-instalação)
6. [Como Usar](#6-como-usar)
7. [Estrutura do Projeto](#7-estrutura-do-projeto)
8. [Resultados e Benchmarks](#8-resultados-e-benchmarks)
9. [Citação](#9-citação)
10. [Licença](#10-licença)
11. [Contato](#11-contato)

---

## 1. Introdução

O **DragaoLT** é um modelo de linguagem de grande escala (LLM) baseado na arquitetura Transformer com **Mistura de Especialistas (MoE)** e **Atenção Latente Multi-Cabeça (MLA)**. Projetado para alcançar alta performance com eficiência computacional, o DragaoLT ativa apenas uma fração dos parâmetros totais por token processado, permitindo escalar o modelo para centenas de bilhões de parâmetros sem aumento proporcional do custo de inferência.

### Características Principais

- **671B de parâmetros totais** com apenas **37B ativados por token**
- **Atenção Latente Multi-Cabeça (MLA)** para compressão eficiente do cache KV
- **Mistura de Especialistas (MoE)** com roteamento inteligente e balanceamento de carga
- **Treinamento e inferência em FP8** nativos para máxima eficiência
- **Janela de contexto de até 128K tokens**
- **Código e variáveis em português** para maior acessibilidade

---

## 2. Arquitetura

O DragaoLT utiliza uma arquitetura Transformer sofisticada com os seguintes componentes:

```
┌─────────────────────────────────────────────────┐
│                 Transformador DragaoLT           │
├─────────────────────────────────────────────────┤
│                                                  │
│  ┌──────────────────────────────────────────┐   │
│  │         Embedding Paralelo               │   │
│  └──────────────┬───────────────────────────┘   │
│                 │                                │
│  ┌──────────────▼───────────────────────────┐   │
│  │      Bloco Transformer × N camadas       │   │
│  │  ┌─────────────────────────────────────┐ │   │
│  │  │  NormaRMS Adaptativa                │ │   │
│  │  │  Atenção Latente Multi-Cabeça (MLA) │ │   │
│  │  │  + Conexão Residual                 │ │   │
│  │  ├─────────────────────────────────────┤ │   │
│  │  │  NormaRMS Adaptativa                │ │   │
│  │  │  MLP (denso) ou MoE (esparso)       │ │   │
│  │  │  + Conexão Residual                 │ │   │
│  │  └─────────────────────────────────────┘ │   │
│  └──────────────┬───────────────────────────┘   │
│                 │                                │
│  ┌──────────────▼───────────────────────────┐   │
│  │    NormaRMS → Projeção de Saída          │   │
│  └──────────────────────────────────────────┘   │
└─────────────────────────────────────────────────┘
```

### 2.1 Atenção Latente Multi-Cabeça (MLA)

A MLA reduz drasticamente o tamanho do cache KV ao comprimir as projeções de chave e valor em um espaço latente de baixo rank usando LoRA, enquanto mantém a expressividade total das cabeças de atenção.

**Vantagens:**
- Redução de **93.75%** no tamanho do cache KV comparado à atenção padrão
- Sem perda de qualidade graças à projeção invertida durante o cálculo de atenção
- Suporte a embeddings posicionais rotacionais (RoPE) com extensão YaRN

### 2.2 Mistura de Especialistas (MoE)

O sistema MoE roteia cada token para um subconjunto de especialistas, permitindo escalar massivamente o número de parâmetros sem aumentar o custo computacional por token.

**Componentes:**
- **Gate (Portão):** Mecanismo de roteamento com suporte a softmax e sigmoid
- **Especialistas Roteados:** Até 256 especialistas independentes
- **Especialistas Compartilhados:** Processam todos os tokens para manter informação global
- **Roteamento em Grupos:** Organiza especialistas em grupos para seleção hierárquica

---

## 3. Inovações Técnicas

### 3.1 Normalização RMS Adaptativa

O DragaoLT introduz um **fator de escala aprendível (α)** na normalização RMS, permitindo que cada camada ajuste dinamicamente a intensidade da normalização. Isso melhora a estabilidade em redes com muitas camadas:

$$\text{NormaRMS}(x) = \alpha \cdot \frac{x}{\sqrt{\frac{1}{d}\sum_{i=1}^{d} x_i^2 + \epsilon}} \odot \gamma$$

### 3.2 Gate com Entropia Balanceada

O mecanismo de roteamento incorpora **monitoramento de entropia** para garantir distribuição uniforme de tokens entre os especialistas, evitando o colapso de roteamento sem usar perda auxiliar tradicional:

$$H(\text{pontuações}) = -\sum_{i} p_i \log(p_i)$$

### 3.3 Dropout Regularizado

Dropout opcional é aplicado em três pontos estratégicos:
- **Atenção:** Após o cálculo das pontuações softmax
- **MLP:** Após a projeção de saída
- **MoE:** Via especialistas compartilhados

### 3.4 Métricas de Utilização de Especialistas

O modelo registra automaticamente métricas de utilização durante o treinamento:
- **Entropia do Gate:** Monitora a diversidade do roteamento
- **Desvio de Carga:** Mede o desbalanceamento entre especialistas
- **Utilização Máxima/Mínima:** Identifica especialistas sobre/subutilizados

### 3.5 Kernels Triton Otimizados

Operações críticas são implementadas com **kernels Triton** customizados:
- Quantização de ativações (FP8)
- Dequantização de pesos
- Multiplicação de matrizes em FP8 com auto-tuning

### 3.6 Extensão de Contexto (YaRN)

Suporte a sequências longas via **YaRN** (Yet Another RoPE Extension), com interpolação suavizada das frequências rotacionais para janelas de até **128K tokens**.

---

## 4. Modelos Disponíveis

| **Modelo** | **Parâmetros Totais** | **Parâmetros Ativados** | **Contexto** | **Configuração** |
|:-:|:-:|:-:|:-:|:-:|
| DragaoLT-16B | 16B | 2.4B | 16K | `config_DragaoLT_16B.json` |
| DragaoLT-236B | 236B | 21B | 16K | `config_DragaoLT_236B.json` |
| DragaoLT-671B | 671B | 37B | 128K | `config_DragaoLT_671B.json` |

---

## 5. Instalação

### Requisitos do Sistema

- **SO:** Linux (Python 3.10+)
- **GPU:** NVIDIA com suporte CUDA
- **RAM GPU:** Variável conforme o modelo escolhido

### Dependências

```bash
pip install -r requisitos.txt
```

Conteúdo do `requisitos.txt`:
```
torch==2.4.1
triton==3.0.0
transformers==4.46.3
safetensors==0.4.5
```

---

## 6. Como Usar

### 6.1 Conversão de Pesos

Converter checkpoints para o formato DragaoLT:

```bash
python converter.py \
    --caminho-hf /caminho/para/pesos_hf \
    --caminho-saida /caminho/para/DragaoLT-Demo \
    --num-especialistas 256 \
    --paralelismo-modelo 16
```

### 6.2 Conversão FP8 → BF16

Se precisar de pesos em BF16:

```bash
python fp8_para_bf16.py \
    --caminho-fp8 /caminho/para/pesos_fp8 \
    --caminho-bf16 /caminho/para/pesos_bf16
```

### 6.3 Modo Interativo (Chat)

```bash
torchrun --nnodes 2 --nproc-per-node 8 --node-rank $RANK --master-addr $ADDR \
    gerar.py \
    --caminho-checkpoint /caminho/para/DragaoLT-Demo \
    --config configs/config_DragaoLT_671B.json \
    --interativo \
    --temperatura 0.7 \
    --max-novos-tokens 200
```

### 6.4 Processamento em Lote

```bash
torchrun --nnodes 2 --nproc-per-node 8 --node-rank $RANK --master-addr $ADDR \
    gerar.py \
    --caminho-checkpoint /caminho/para/DragaoLT-Demo \
    --config configs/config_DragaoLT_671B.json \
    --arquivo-entrada prompts.txt
```

### 6.5 Comandos Interativos

| Comando | Descrição |
|---------|-----------|
| `/sair` | Encerra a sessão |
| `/limpar` | Limpa o histórico de conversa |

---

## 7. Estrutura do Projeto

```
dragaoLT/
├── README.md                          # Este arquivo
├── LICENCA.md                         # Licença MIT
├── requisitos.txt                     # Dependências Python
├── modelo.py                          # Arquitetura principal do Transformer
├── nucleo.py                          # Kernels Triton otimizados (FP8)
├── gerar.py                           # Script de geração/inferência
├── converter.py                       # Conversor de checkpoints HF → DragaoLT
├── fp8_para_bf16.py                   # Conversor FP8 → BF16
└── configs/
    ├── config_DragaoLT_16B.json       # Configuração 16B parâmetros
    ├── config_DragaoLT_236B.json      # Configuração 236B parâmetros
    └── config_DragaoLT_671B.json      # Configuração 671B parâmetros
```

### Descrição dos Módulos

| Módulo | Descrição |
|--------|-----------|
| `modelo.py` | Implementação completa do Transformer com MLA, MoE, NormaRMS adaptativa, Gate com entropia e métricas de especialistas |
| `nucleo.py` | Kernels Triton para quantização FP8, dequantização de pesos e GEMM FP8 com auto-tuning |
| `gerar.py` | Script de inferência com modo interativo (chat) e processamento em lote |
| `converter.py` | Converte checkpoints Hugging Face para formato DragaoLT com suporte a paralelismo |
| `fp8_para_bf16.py` | Converte pesos FP8 para BF16 usando dequantização por blocos |

---

## 8. Resultados e Benchmarks

O DragaoLT-671B demonstra desempenho competitivo em diversos benchmarks:

### Modelo Base

| Benchmark | Métrica | DragaoLT-671B |
|-----------|---------|:-------------:|
| MMLU | Acurácia (5-shot) | **87.1** |
| BBH | EM (3-shot) | **87.5** |
| HumanEval | Pass@1 (0-shot) | **65.2** |
| MATH | EM (4-shot) | **61.6** |
| GSM8K | EM (8-shot) | **89.3** |
| DROP | F1 (3-shot) | **89.0** |
| MMLU-Pro | Acurácia (5-shot) | **64.4** |

### Modelo Chat

| Benchmark | Métrica | DragaoLT-671B |
|-----------|---------|:-------------:|
| MMLU | EM | **88.5** |
| MATH-500 | EM | **90.2** |
| AIME 2024 | Pass@1 | **39.2** |
| LiveCodeBench | Pass@1-COT | **40.5** |
| Codeforces | Percentil | **51.6** |
| Arena-Hard | Score | **85.5** |
| AlpacaEval 2.0 | Win Rate | **70.0** |

---

## 9. Citação

Se utilizar o DragaoLT em sua pesquisa, por favor cite:

```bibtex
@misc{wilcke2025dragaolt,
    title={DragaoLT: Modelo de Linguagem com Mistura de Especialistas e Atenção Latente Multi-Cabeça},
    author={Luiz Tiago Wilcke},
    year={2025},
    note={Disponível em: https://github.com/luiztiagow1987/DragaoLT}
}
```

---

## 10. Licença

Este projeto é licenciado sob a [Licença MIT](LICENCA.md).

---

## 11. Contato

**Autor:** Luiz Tiago Wilcke

Para dúvidas, sugestões ou colaborações, abra uma issue no repositório ou entre em contato diretamente.

---

<div align="center">

**DragaoLT** — *Modelo de Linguagem de Grande Escala*

Desenvolvido por **Luiz Tiago Wilcke** 🇧🇷

</div>
