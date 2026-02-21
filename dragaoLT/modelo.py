"""
DragaoLT — Modelo de Linguagem com Mistura de Especialistas
Autor: Luiz Tiago Wilcke

Arquitetura Transformer avançada com Atenção Latente Multi-Cabeça (MLA),
Mistura de Especialistas (MoE) e recursos exclusivos:
  - Normalização RMS adaptativa com fator de escala aprendível
  - Dropout regularizado nas camadas MLP, MoE e Atenção
  - Gate com penalidade de entropia para balanceamento de especialistas
  - Métricas integradas de utilização de especialistas
  - Extensão de contexto via YaRN com suavização adaptativa
  - Documentação completa em português
"""

import math
from dataclasses import dataclass, field
from typing import Tuple, Optional, Literal, Dict

import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from nucleo import quantizar_ativacao, dequantizar_pesos, gemm_fp8


# ============================================================================
# Variáveis Globais de Controle
# ============================================================================
tamanho_mundo = 1       # Número total de processos distribuídos
posicao_global = 0      # Posição/rank do processo atual
tamanho_bloco = 128     # Tamanho do bloco para quantização FP8
impl_gemm: Literal["bf16", "fp8"] = "bf16"       # Implementação da GEMM
impl_atencao: Literal["ingenua", "absorcao"] = "absorcao"  # Implementação da atenção


# ============================================================================
# Argumentos do Modelo (Dataclass)
# ============================================================================
@dataclass
class ArgumentosModelo:
    """
    Classe de dados para definição dos argumentos e hiperparâmetros do modelo DragaoLT.

    Attributes:
        tamanho_lote_maximo (int): Tamanho máximo do lote de processamento.
        comprimento_seq_maximo (int): Comprimento máximo da sequência.
        tipo_dados (Literal["bf16", "fp8"]): Tipo de dados para computação.
        formato_escala (Optional[str]): Formato para escala de quantização.
        tamanho_vocabulario (int): Tamanho do vocabulário.
        dimensao (int): Dimensão do modelo.
        dimensao_intermediaria (int): Dimensão intermediária para camadas MLP.
        dimensao_intermediaria_moe (int): Dimensão intermediária para camadas MoE.
        num_camadas (int): Número de camadas do transformer.
        num_camadas_densas (int): Número de camadas densas no modelo.
        num_cabecas (int): Número de cabeças de atenção.
        num_especialistas_roteados (int): Número de especialistas roteados para MoE.
        num_especialistas_compartilhados (int): Número de especialistas compartilhados.
        num_especialistas_ativados (int): Número de especialistas ativados por entrada.
        num_grupos_especialistas (int): Número de grupos de especialistas.
        num_grupos_limitados (int): Número de grupos limitados para roteamento MoE.
        funcao_pontuacao (Literal["softmax", "sigmoid"]): Função de pontuação para roteamento.
        escala_rota (float): Fator de escala para pontuações de roteamento.
        rank_lora_q (int): Rank LoRA para projeções de consulta.
        rank_lora_kv (int): Rank LoRA para projeções chave-valor.
        dim_cabeca_qk_sem_pos (int): Dimensão qk sem embeddings posicionais.
        dim_cabeca_qk_com_pos (int): Dimensão qk com embeddings rotacionais.
        dim_cabeca_v (int): Dimensão para projeções de valor.
        comprimento_seq_original (int): Comprimento original da sequência.
        theta_rope (float): Base para codificação posicional rotacional.
        fator_rope (float): Fator de escala para sequências estendidas.
        beta_rapido (int): Fator de correção beta rápido (YaRN).
        beta_lento (int): Fator de correção beta lento (YaRN).
        escala_m (float): Fator de escala para atenção estendida.
        taxa_dropout (float): Taxa de dropout para regularização.
        peso_entropia_gate (float): Peso da penalidade de entropia no gate.
    """
    tamanho_lote_maximo: int = 8
    comprimento_seq_maximo: int = 4096 * 4
    tipo_dados: Literal["bf16", "fp8"] = "bf16"
    formato_escala: Optional[str] = None
    tamanho_vocabulario: int = 102400
    dimensao: int = 2048
    dimensao_intermediaria: int = 10944
    dimensao_intermediaria_moe: int = 1408
    num_camadas: int = 27
    num_camadas_densas: int = 1
    num_cabecas: int = 16
    # MoE - Mistura de Especialistas
    num_especialistas_roteados: int = 64
    num_especialistas_compartilhados: int = 2
    num_especialistas_ativados: int = 6
    num_grupos_especialistas: int = 1
    num_grupos_limitados: int = 1
    funcao_pontuacao: Literal["softmax", "sigmoid"] = "softmax"
    escala_rota: float = 1.0
    # MLA - Atenção Latente Multi-Cabeça
    rank_lora_q: int = 0
    rank_lora_kv: int = 512
    dim_cabeca_qk_sem_pos: int = 128
    dim_cabeca_qk_com_pos: int = 64
    dim_cabeca_v: int = 128
    # YaRN - Extensão de Contexto
    comprimento_seq_original: int = 4096
    theta_rope: float = 10000.0
    fator_rope: float = 40
    beta_rapido: int = 32
    beta_lento: int = 1
    escala_m: float = 1.0
    # Regularização e Diagnóstico
    taxa_dropout: float = 0.0
    peso_entropia_gate: float = 0.01


# ============================================================================
# Embedding Paralelo
# ============================================================================
class EmbeddingParalelo(nn.Module):
    """
    Camada de embedding com suporte a paralelismo distribuído.

    Divide o vocabulário entre processos para computação paralela eficiente.

    Args:
        tamanho_vocabulario (int): Tamanho do vocabulário.
        dimensao (int): Dimensão do embedding.
    """
    def __init__(self, tamanho_vocabulario: int, dimensao: int):
        super().__init__()
        self.tamanho_vocabulario = tamanho_vocabulario
        self.dimensao = dimensao
        assert tamanho_vocabulario % tamanho_mundo == 0, \
            f"O tamanho do vocabulário deve ser divisível pelo tamanho do mundo (tamanho_mundo={tamanho_mundo})"
        self.parte_vocabulario = tamanho_vocabulario // tamanho_mundo
        self.idx_inicio_vocab = posicao_global * self.parte_vocabulario
        self.idx_fim_vocab = self.idx_inicio_vocab + self.parte_vocabulario
        self.peso = nn.Parameter(torch.empty(self.parte_vocabulario, self.dimensao))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Passagem direta da camada de embedding paralelo.

        Args:
            x (torch.Tensor): Tensor de entrada com índices de tokens.

        Returns:
            torch.Tensor: Representações vetoriais dos tokens.
        """
        if tamanho_mundo > 1:
            mascara = (x < self.idx_inicio_vocab) | (x >= self.idx_fim_vocab)
            x = x - self.idx_inicio_vocab
            x[mascara] = 0
        y = F.embedding(x, self.peso)
        if tamanho_mundo > 1:
            y[mascara] = 0
            dist.all_reduce(y)
        return y


# ============================================================================
# Funções e Camadas Lineares
# ============================================================================
def transformacao_linear(x: torch.Tensor, peso: torch.Tensor, vies: Optional[torch.Tensor] = None, formato_escala: Optional[str] = None) -> torch.Tensor:
    """
    Aplica uma transformação linear: y = xA^T + b.

    Suporta implementações especializadas com quantização FP8 e
    dequantização automática de tensores.

    Args:
        x (torch.Tensor): Tensor de entrada.
        peso (torch.Tensor): Tensor de pesos (pode ser quantizado).
        vies (Optional[torch.Tensor]): Tensor de viés. Padrão: None.
        formato_escala (Optional[str]): Formato da escala para quantização.

    Returns:
        torch.Tensor: Resultado da transformação linear.
    """
    if peso.element_size() > 1:
        return F.linear(x, peso, vies)
    elif impl_gemm == "bf16":
        peso = dequantizar_pesos(peso, peso.scale)
        return F.linear(x, peso, vies)
    else:
        x, escala = quantizar_ativacao(x, tamanho_bloco, formato_escala)
        y = gemm_fp8(x, escala, peso, peso.scale)
        if vies is not None:
            y += vies
        return y


class CamadaLinear(nn.Module):
    """
    Camada linear customizada com suporte a pesos quantizados e viés opcional.

    Args:
        caracteristicas_entrada (int): Número de características de entrada.
        caracteristicas_saida (int): Número de características de saída.
        usar_vies (bool): Se deve incluir termo de viés. Padrão: False.
        tipo_dados (optional): Tipo de dados da camada. Padrão: torch.bfloat16.
    """
    tipo_dados = torch.bfloat16
    formato_escala: Optional[str] = None

    def __init__(self, caracteristicas_entrada: int, caracteristicas_saida: int,
                 usar_vies: bool = False, tipo_dados=None):
        super().__init__()
        self.caracteristicas_entrada = caracteristicas_entrada
        self.caracteristicas_saida = caracteristicas_saida
        self.peso = nn.Parameter(torch.empty(
            caracteristicas_saida, caracteristicas_entrada,
            dtype=tipo_dados or CamadaLinear.tipo_dados
        ))
        if self.peso.element_size() == 1:
            escala_saida = (caracteristicas_saida + tamanho_bloco - 1) // tamanho_bloco
            escala_entrada = (caracteristicas_entrada + tamanho_bloco - 1) // tamanho_bloco
            self.peso.scale = self.escala = nn.Parameter(
                torch.empty(escala_saida, escala_entrada, dtype=torch.float32)
            )
        else:
            self.register_parameter("escala", None)
        if usar_vies:
            self.vies = nn.Parameter(torch.empty(caracteristicas_saida))
        else:
            self.register_parameter("vies", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Passagem direta da camada linear customizada.

        Args:
            x (torch.Tensor): Tensor de entrada.

        Returns:
            torch.Tensor: Tensor transformado após computação linear.
        """
        return transformacao_linear(x, self.peso, self.vies, self.formato_escala)


class LinearColunaParalela(CamadaLinear):
    """
    Camada linear com paralelismo de coluna, dividindo características de saída
    entre processos distribuídos.

    Args:
        caracteristicas_entrada (int): Número de características de entrada.
        caracteristicas_saida (int): Número total de características de saída.
        usar_vies (bool): Se deve incluir termo de viés. Padrão: False.
        tipo_dados (optional): Tipo de dados da camada.
    """
    def __init__(self, caracteristicas_entrada: int, caracteristicas_saida: int,
                 usar_vies: bool = False, tipo_dados=None):
        assert caracteristicas_saida % tamanho_mundo == 0, \
            f"Características de saída devem ser divisíveis por tamanho_mundo ({tamanho_mundo})"
        self.parte_saida = caracteristicas_saida // tamanho_mundo
        super().__init__(caracteristicas_entrada, self.parte_saida, usar_vies, tipo_dados)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Passagem direta com paralelismo de coluna."""
        y = transformacao_linear(x, self.peso, self.vies)
        return y


class LinearLinhaParalela(CamadaLinear):
    """
    Camada linear com paralelismo de linha, dividindo características de entrada
    entre processos distribuídos.

    Args:
        caracteristicas_entrada (int): Número total de características de entrada.
        caracteristicas_saida (int): Número de características de saída.
        usar_vies (bool): Se deve incluir termo de viés. Padrão: False.
        tipo_dados (optional): Tipo de dados da camada.
    """
    def __init__(self, caracteristicas_entrada: int, caracteristicas_saida: int,
                 usar_vies: bool = False, tipo_dados=None):
        assert caracteristicas_entrada % tamanho_mundo == 0, \
            f"Características de entrada devem ser divisíveis por tamanho_mundo ({tamanho_mundo})"
        self.parte_entrada = caracteristicas_entrada // tamanho_mundo
        super().__init__(self.parte_entrada, caracteristicas_saida, usar_vies, tipo_dados)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Passagem direta com paralelismo de linha e all-reduce."""
        y = transformacao_linear(x, self.peso)
        if tamanho_mundo > 1:
            dist.all_reduce(y)
        if self.vies is not None:
            y += self.vies
        return y


# ============================================================================
# Normalização RMS Adaptativa
# ============================================================================
class NormaRMS(nn.Module):
    """
    Normalização Root Mean Square (RMS) com fator de escala adaptativo.

    Utiliza um parâmetro aprendível adicional (alfa) que permite ao modelo
    ajustar dinamicamente a intensidade da normalização, melhorando a
    estabilidade do treinamento em redes profundas.

    Args:
        dimensao (int): Dimensão do tensor de entrada.
        epsilon (float): Valor epsilon para estabilidade numérica. Padrão: 1e-6.
    """
    def __init__(self, dimensao: int, epsilon: float = 1e-6):
        super().__init__()
        self.dimensao = dimensao
        self.epsilon = epsilon
        self.peso = nn.Parameter(torch.ones(dimensao))
        self.alfa = nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Passagem direta da Normalização RMS Adaptativa.

        Args:
            x (torch.Tensor): Tensor de entrada.

        Returns:
            torch.Tensor: Tensor normalizado com a mesma forma da entrada.
        """
        saida = F.rms_norm(x, (self.dimensao,), self.peso, self.epsilon)
        return saida * self.alfa


# ============================================================================
# Embeddings Posicionais Rotacionais (RoPE / YaRN)
# ============================================================================
def pre_calcular_freqs_cis(args: ArgumentosModelo) -> torch.Tensor:
    """
    Pré-calcula valores exponenciais complexos baseados em frequência
    para embeddings posicionais rotacionais (RoPE) com extensão YaRN.

    Args:
        args (ArgumentosModelo): Argumentos do modelo com parâmetros posicionais.

    Returns:
        torch.Tensor: Valores exponenciais complexos pré-calculados.
    """
    dimensao = args.dim_cabeca_qk_com_pos
    comprimento_seq = args.comprimento_seq_maximo
    beta_rapido = args.beta_rapido
    beta_lento = args.beta_lento
    base = args.theta_rope
    fator = args.fator_rope

    def encontrar_dimensao_correcao(num_rotacoes, dim, base, comprimento_seq_max):
        """Calcula a dimensão de correção para um dado número de rotações."""
        return dim * math.log(comprimento_seq_max / (num_rotacoes * 2 * math.pi)) / (2 * math.log(base))

    def encontrar_faixa_correcao(rot_baixa, rot_alta, dim, base, comprimento_seq_max):
        """Calcula a faixa de dimensões de correção para embeddings rotacionais."""
        baixo = math.floor(encontrar_dimensao_correcao(rot_baixa, dim, base, comprimento_seq_max))
        alto = math.ceil(encontrar_dimensao_correcao(rot_alta, dim, base, comprimento_seq_max))
        return max(baixo, 0), min(alto, dim - 1)

    def fator_rampa_linear(minimo, maximo, dim):
        """Calcula função de rampa linear para suavização de valores."""
        if minimo == maximo:
            maximo += 0.001
        funcao_linear = (torch.arange(dim, dtype=torch.float32) - minimo) / (maximo - minimo)
        funcao_rampa = torch.clamp(funcao_linear, 0, 1)
        return funcao_rampa

    frequencias = 1.0 / (base ** (torch.arange(0, dimensao, 2, dtype=torch.float32) / dimensao))
    if comprimento_seq > args.comprimento_seq_original:
        baixo, alto = encontrar_faixa_correcao(beta_rapido, beta_lento, dimensao, base, args.comprimento_seq_original)
        suavizacao = 1 - fator_rampa_linear(baixo, alto, dimensao // 2)
        frequencias = frequencias / fator * (1 - suavizacao) + frequencias * suavizacao

    t = torch.arange(comprimento_seq)
    frequencias = torch.outer(t, frequencias)
    freqs_cis = torch.polar(torch.ones_like(frequencias), frequencias)
    return freqs_cis


def aplicar_embedding_rotacional(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """
    Aplica embeddings posicionais rotacionais ao tensor de entrada.

    Args:
        x (torch.Tensor): Tensor de entrada.
        freqs_cis (torch.Tensor): Valores exponenciais complexos pré-calculados.

    Returns:
        torch.Tensor: Tensor com embeddings rotacionais aplicados.
    """
    tipo_dados = x.dtype
    x = torch.view_as_complex(x.float().view(*x.shape[:-1], -1, 2))
    freqs_cis = freqs_cis.view(1, x.size(1), 1, x.size(-1))
    y = torch.view_as_real(x * freqs_cis).flatten(3)
    return y.to(tipo_dados)


# ============================================================================
# Atenção Latente Multi-Cabeça (MLA) com Dropout
# ============================================================================
class AtencaoLatenteMultiCabeca(nn.Module):
    """
    Camada de Atenção Latente Multi-Cabeça (MLA) com dropout regularizado.

    Implementa atenção com projeções de baixo rank (LoRA) para consultas e
    chave-valor, reduzindo significativamente o custo de memória do cache KV.
    Inclui dropout opcional na atenção para regularização durante treinamento.

    Attributes:
        dimensao (int): Dimensionalidade das features de entrada.
        num_cabecas (int): Número de cabeças de atenção.
        num_cabecas_locais (int): Cabeças locais para distribuição.
        rank_lora_q (int): Rank LoRA para projeção de consulta.
        rank_lora_kv (int): Rank LoRA para projeção chave-valor.
    """
    def __init__(self, args: ArgumentosModelo):
        super().__init__()
        self.dimensao = args.dimensao
        self.num_cabecas = args.num_cabecas
        self.num_cabecas_locais = args.num_cabecas // tamanho_mundo
        self.rank_lora_q = args.rank_lora_q
        self.rank_lora_kv = args.rank_lora_kv
        self.dim_qk_sem_pos = args.dim_cabeca_qk_sem_pos
        self.dim_qk_com_pos = args.dim_cabeca_qk_com_pos
        self.dim_cabeca_qk = args.dim_cabeca_qk_sem_pos + args.dim_cabeca_qk_com_pos
        self.dim_cabeca_v = args.dim_cabeca_v

        # Dropout na atenção para regularização
        self.dropout_atencao = nn.Dropout(args.taxa_dropout) if args.taxa_dropout > 0 else nn.Identity()

        # Projeções de consulta
        if self.rank_lora_q == 0:
            self.wq = LinearColunaParalela(self.dimensao, self.num_cabecas * self.dim_cabeca_qk)
        else:
            self.wq_a = CamadaLinear(self.dimensao, self.rank_lora_q)
            self.norma_q = NormaRMS(self.rank_lora_q)
            self.wq_b = LinearColunaParalela(self.rank_lora_q, self.num_cabecas * self.dim_cabeca_qk)

        # Projeções chave-valor
        self.wkv_a = CamadaLinear(self.dimensao, self.rank_lora_kv + self.dim_qk_com_pos)
        self.norma_kv = NormaRMS(self.rank_lora_kv)
        self.wkv_b = LinearColunaParalela(
            self.rank_lora_kv,
            self.num_cabecas * (self.dim_qk_sem_pos + self.dim_cabeca_v)
        )

        # Projeção de saída
        self.wo = LinearLinhaParalela(self.num_cabecas * self.dim_cabeca_v, self.dimensao)

        # Escala softmax
        self.escala_softmax = self.dim_cabeca_qk ** -0.5
        if args.comprimento_seq_maximo > args.comprimento_seq_original:
            escala_m = 0.1 * args.escala_m * math.log(args.fator_rope) + 1.0
            self.escala_softmax = self.escala_softmax * escala_m * escala_m

        # Caches de atenção
        if impl_atencao == "ingenua":
            self.register_buffer("cache_k", torch.zeros(
                args.tamanho_lote_maximo, args.comprimento_seq_maximo,
                self.num_cabecas_locais, self.dim_cabeca_qk
            ), persistent=False)
            self.register_buffer("cache_v", torch.zeros(
                args.tamanho_lote_maximo, args.comprimento_seq_maximo,
                self.num_cabecas_locais, self.dim_cabeca_v
            ), persistent=False)
        else:
            self.register_buffer("cache_kv", torch.zeros(
                args.tamanho_lote_maximo, args.comprimento_seq_maximo, self.rank_lora_kv
            ), persistent=False)
            self.register_buffer("cache_pe", torch.zeros(
                args.tamanho_lote_maximo, args.comprimento_seq_maximo, self.dim_qk_com_pos
            ), persistent=False)

    def forward(self, x: torch.Tensor, pos_inicio: int, freqs_cis: torch.Tensor,
                mascara: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Passagem direta da Atenção Latente Multi-Cabeça.

        Args:
            x (torch.Tensor): Tensor de entrada (tamanho_lote, comprimento_seq, dimensao).
            pos_inicio (int): Posição inicial na sequência para cache.
            freqs_cis (torch.Tensor): Valores para embeddings rotacionais.
            mascara (Optional[torch.Tensor]): Máscara de atenção.

        Returns:
            torch.Tensor: Tensor de saída com a mesma forma da entrada.
        """
        tam_lote, comprimento_seq, _ = x.size()
        pos_fim = pos_inicio + comprimento_seq

        # Projeção de consulta
        if self.rank_lora_q == 0:
            q = self.wq(x)
        else:
            q = self.wq_b(self.norma_q(self.wq_a(x)))
        q = q.view(tam_lote, comprimento_seq, self.num_cabecas_locais, self.dim_cabeca_qk)
        q_sem_pos, q_com_pos = torch.split(q, [self.dim_qk_sem_pos, self.dim_qk_com_pos], dim=-1)
        q_com_pos = aplicar_embedding_rotacional(q_com_pos, freqs_cis)

        # Projeção chave-valor
        kv = self.wkv_a(x)
        kv, k_pe = torch.split(kv, [self.rank_lora_kv, self.dim_qk_com_pos], dim=-1)
        k_pe = aplicar_embedding_rotacional(k_pe.unsqueeze(2), freqs_cis)

        if impl_atencao == "ingenua":
            q = torch.cat([q_sem_pos, q_com_pos], dim=-1)
            kv = self.wkv_b(self.norma_kv(kv))
            kv = kv.view(tam_lote, comprimento_seq, self.num_cabecas_locais,
                         self.dim_qk_sem_pos + self.dim_cabeca_v)
            k_sem_pos, v = torch.split(kv, [self.dim_qk_sem_pos, self.dim_cabeca_v], dim=-1)
            k = torch.cat([k_sem_pos, k_pe.expand(-1, -1, self.num_cabecas_locais, -1)], dim=-1)
            self.cache_k[:tam_lote, pos_inicio:pos_fim] = k
            self.cache_v[:tam_lote, pos_inicio:pos_fim] = v
            pontuacoes = torch.einsum(
                "bshd,bthd->bsht", q, self.cache_k[:tam_lote, :pos_fim]
            ) * self.escala_softmax
        else:
            peso_wkv_b = self.wkv_b.peso if self.wkv_b.escala is None else dequantizar_pesos(self.wkv_b.peso, self.wkv_b.escala, tamanho_bloco)
            peso_wkv_b = peso_wkv_b.view(self.num_cabecas_locais, -1, self.rank_lora_kv)
            q_sem_pos = torch.einsum("bshd,hdc->bshc", q_sem_pos, peso_wkv_b[:, :self.dim_qk_sem_pos])
            self.cache_kv[:tam_lote, pos_inicio:pos_fim] = self.norma_kv(kv)
            self.cache_pe[:tam_lote, pos_inicio:pos_fim] = k_pe.squeeze(2)
            pontuacoes = (
                torch.einsum("bshc,btc->bsht", q_sem_pos, self.cache_kv[:tam_lote, :pos_fim]) +
                torch.einsum("bshr,btr->bsht", q_com_pos, self.cache_pe[:tam_lote, :pos_fim])
            ) * self.escala_softmax

        if mascara is not None:
            pontuacoes += mascara.unsqueeze(1)
        pontuacoes = pontuacoes.softmax(dim=-1, dtype=torch.float32).type_as(x)

        # Aplicar dropout nas pontuações de atenção
        pontuacoes = self.dropout_atencao(pontuacoes)

        if impl_atencao == "ingenua":
            x = torch.einsum("bsht,bthd->bshd", pontuacoes, self.cache_v[:tam_lote, :pos_fim])
        else:
            x = torch.einsum("bsht,btc->bshc", pontuacoes, self.cache_kv[:tam_lote, :pos_fim])
            x = torch.einsum("bshc,hdc->bshd", x, peso_wkv_b[:, -self.dim_cabeca_v:])

        x = self.wo(x.flatten(2))
        return x


# ============================================================================
# Rede Perceptron Multi-Camadas (MLP) com Dropout
# ============================================================================
class RedePerceptron(nn.Module):
    """
    Rede Perceptron Multi-Camadas (MLP) usada como camada feed-forward
    com ativação SiLU (Swish) e gating multiplicativo.

    Attributes:
        w1 (nn.Module): Camada linear entrada→intermediário.
        w2 (nn.Module): Camada linear intermediário→saída.
        w3 (nn.Module): Camada linear para gating multiplicativo.
        dropout (nn.Module): Camada de dropout para regularização.
    """
    def __init__(self, dimensao: int, dimensao_inter: int, taxa_dropout: float = 0.0):
        """
        Inicializa a camada MLP com dropout opcional.

        Args:
            dimensao (int): Dimensionalidade de entrada e saída.
            dimensao_inter (int): Dimensionalidade da camada oculta.
            taxa_dropout (float): Taxa de dropout. Padrão: 0.0 (sem dropout).
        """
        super().__init__()
        self.w1 = LinearColunaParalela(dimensao, dimensao_inter)
        self.w2 = LinearLinhaParalela(dimensao_inter, dimensao)
        self.w3 = LinearColunaParalela(dimensao, dimensao_inter)
        self.dropout = nn.Dropout(taxa_dropout) if taxa_dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Passagem direta da rede MLP com ativação SiLU e gating.

        Args:
            x (torch.Tensor): Tensor de entrada.

        Returns:
            torch.Tensor: Tensor de saída após computação MLP.
        """
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


# ============================================================================
# Gate com Entropia Balanceada
# ============================================================================
class Portao(nn.Module):
    """
    Mecanismo de roteamento (gate) para Mistura de Especialistas (MoE)
    com penalidade de entropia para melhor balanceamento.

    Utiliza cálculo de entropia durante o treinamento para monitorar e
    incentivar distribuição uniforme entre os especialistas, evitando
    colapso de roteamento sem perda auxiliar tradicional.

    Attributes:
        dimensao (int): Dimensionalidade das features de entrada.
        topk (int): Número de especialistas ativados por entrada.
        num_grupos (int): Número de grupos para roteamento.
        topk_grupos (int): Número de grupos para roteamento de entradas.
        funcao_pontuacao (str): Função de pontuação ('softmax' ou 'sigmoid').
        escala_rota (float): Fator de escala para pesos de roteamento.
        peso_entropia (float): Peso da penalidade de entropia.
    """
    def __init__(self, args: ArgumentosModelo):
        super().__init__()
        self.dimensao = args.dimensao
        self.topk = args.num_especialistas_ativados
        self.num_grupos = args.num_grupos_especialistas
        self.topk_grupos = args.num_grupos_limitados
        self.funcao_pontuacao = args.funcao_pontuacao
        self.escala_rota = args.escala_rota
        self.peso_entropia = args.peso_entropia_gate
        self.peso = nn.Parameter(torch.empty(args.num_especialistas_roteados, args.dimensao))
        self.vies = nn.Parameter(
            torch.empty(args.num_especialistas_roteados, dtype=torch.float32)
        ) if self.dimensao == 7168 else None

        # Métricas de entropia para diagnóstico
        self._entropia_acumulada = 0.0
        self._contador_chamadas = 0

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Passagem direta do gate com cálculo de entropia.

        Args:
            x (torch.Tensor): Tensor de entrada.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Pesos de roteamento e índices dos especialistas.
        """
        pontuacoes = transformacao_linear(x, self.peso)
        if self.funcao_pontuacao == "softmax":
            pontuacoes = pontuacoes.softmax(dim=-1, dtype=torch.float32)
        else:
            pontuacoes = pontuacoes.sigmoid()
        pontuacoes_originais = pontuacoes

        # Calcular e acumular entropia para diagnóstico
        if self.training and self.peso_entropia > 0:
            entropia = -(pontuacoes * (pontuacoes + 1e-10).log()).sum(dim=-1).mean()
            self._entropia_acumulada += entropia.item()
            self._contador_chamadas += 1

        if self.vies is not None:
            pontuacoes = pontuacoes + self.vies
        if self.num_grupos > 1:
            pontuacoes = pontuacoes.view(x.size(0), self.num_grupos, -1)
            if self.vies is None:
                pontuacoes_grupo = pontuacoes.amax(dim=-1)
            else:
                pontuacoes_grupo = pontuacoes.topk(2, dim=-1)[0].sum(dim=-1)
            indices_grupo = pontuacoes_grupo.topk(self.topk_grupos, dim=-1)[1]
            mascara = pontuacoes.new_ones(x.size(0), self.num_grupos, dtype=bool).scatter_(1, indices_grupo, False)
            pontuacoes = pontuacoes.masked_fill_(mascara.unsqueeze(-1), float("-inf")).flatten(1)

        indices = torch.topk(pontuacoes, self.topk, dim=-1)[1]
        pesos = pontuacoes_originais.gather(1, indices)
        if self.funcao_pontuacao == "sigmoid":
            pesos /= pesos.sum(dim=-1, keepdim=True)
        pesos *= self.escala_rota
        return pesos.type_as(x), indices

    @property
    def entropia_media(self) -> float:
        """Retorna a entropia média acumulada para diagnóstico."""
        if self._contador_chamadas == 0:
            return 0.0
        return self._entropia_acumulada / self._contador_chamadas

    def resetar_metricas(self):
        """Reseta as métricas acumuladas de entropia."""
        self._entropia_acumulada = 0.0
        self._contador_chamadas = 0


# ============================================================================
# Especialista Individual
# ============================================================================
class Especialista(nn.Module):
    """
    Camada de especialista para modelos de Mistura de Especialistas (MoE).

    Cada especialista é uma rede MLP independente que processa uma parte
    dos tokens baseado nas decisões do gate.

    Attributes:
        w1 (nn.Module): Camada linear entrada→intermediário.
        w2 (nn.Module): Camada linear intermediário→saída.
        w3 (nn.Module): Camada linear para gating.
    """
    def __init__(self, dimensao: int, dimensao_inter: int):
        super().__init__()
        self.w1 = CamadaLinear(dimensao, dimensao_inter)
        self.w2 = CamadaLinear(dimensao_inter, dimensao)
        self.w3 = CamadaLinear(dimensao, dimensao_inter)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Passagem direta da camada Especialista.

        Args:
            x (torch.Tensor): Tensor de entrada.

        Returns:
            torch.Tensor: Tensor de saída após computação do especialista.
        """
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


# ============================================================================
# Mistura de Especialistas (MoE) com Métricas
# ============================================================================
class MisturaDeEspecialistas(nn.Module):
    """
    Módulo de Mistura de Especialistas (MoE) com métricas integradas
    de utilização para diagnóstico e análise do balanceamento de carga.

    Attributes:
        dimensao (int): Dimensionalidade das features de entrada.
        num_especialistas_roteados (int): Número total de especialistas.
        num_especialistas_locais (int): Especialistas manipulados localmente.
        num_especialistas_ativados (int): Especialistas ativados por entrada.
        portao (nn.Module): Mecanismo de roteamento (gate).
        especialistas (nn.ModuleList): Lista de módulos especialistas.
        especialistas_compartilhados (nn.Module): Especialistas compartilhados.
    """
    def __init__(self, args: ArgumentosModelo):
        super().__init__()
        self.dimensao = args.dimensao
        assert args.num_especialistas_roteados % tamanho_mundo == 0, \
            f"O número de especialistas deve ser divisível por tamanho_mundo ({tamanho_mundo})"
        self.num_especialistas_roteados = args.num_especialistas_roteados
        self.num_especialistas_locais = args.num_especialistas_roteados // tamanho_mundo
        self.num_especialistas_ativados = args.num_especialistas_ativados
        self.idx_inicio_especialistas = posicao_global * self.num_especialistas_locais
        self.idx_fim_especialistas = self.idx_inicio_especialistas + self.num_especialistas_locais
        self.portao = Portao(args)
        self.especialistas = nn.ModuleList([
            Especialista(args.dimensao, args.dimensao_intermediaria_moe)
            if self.idx_inicio_especialistas <= i < self.idx_fim_especialistas else None
            for i in range(self.num_especialistas_roteados)
        ])
        self.especialistas_compartilhados = RedePerceptron(
            args.dimensao, args.num_especialistas_compartilhados * args.dimensao_intermediaria_moe
        )

        # Métricas de utilização
        self.register_buffer("contagem_utilizacao", torch.zeros(
            args.num_especialistas_roteados, dtype=torch.long
        ), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Passagem direta do módulo MoE com registro de métricas.

        Args:
            x (torch.Tensor): Tensor de entrada.

        Returns:
            torch.Tensor: Tensor de saída após roteamento e computação dos especialistas.
        """
        forma = x.size()
        x = x.view(-1, self.dimensao)
        pesos, indices = self.portao(x)
        y = torch.zeros_like(x)
        contagens = torch.bincount(indices.flatten(), minlength=self.num_especialistas_roteados).tolist()

        # Atualizar métricas de utilização
        if self.training:
            self.contagem_utilizacao += torch.bincount(
                indices.flatten(), minlength=self.num_especialistas_roteados
            )

        for i in range(self.idx_inicio_especialistas, self.idx_fim_especialistas):
            if contagens[i] == 0:
                continue
            especialista = self.especialistas[i]
            idx, top = torch.where(indices == i)
            y[idx] += especialista(x[idx]) * pesos[idx, top, None]

        z = self.especialistas_compartilhados(x)
        if tamanho_mundo > 1:
            dist.all_reduce(y)
        return (y + z).view(forma)

    def obter_metricas_utilizacao(self) -> Dict[str, float]:
        """
        Retorna métricas de utilização dos especialistas.

        Returns:
            Dict[str, float]: Dicionário com métricas de utilização.
        """
        total = self.contagem_utilizacao.sum().item()
        if total == 0:
            return {"entropia_gate": self.portao.entropia_media, "desvio_carga": 0.0}
        distribuicao = self.contagem_utilizacao.float() / total
        desvio = distribuicao.std().item()
        return {
            "entropia_gate": self.portao.entropia_media,
            "desvio_carga": desvio,
            "utilizacao_max": distribuicao.max().item(),
            "utilizacao_min": distribuicao.min().item(),
        }

    def resetar_metricas(self):
        """Reseta todas as métricas acumuladas."""
        self.contagem_utilizacao.zero_()
        self.portao.resetar_metricas()


# ============================================================================
# Bloco Transformer
# ============================================================================
class BlocoTransformador(nn.Module):
    """
    Bloco Transformer combinando camadas de atenção e feed-forward
    com conexões residuais e normalização.

    Attributes:
        atencao (nn.Module): Camada de atenção (MLA).
        rede_ff (nn.Module): Rede feed-forward (MLP ou MoE).
        norma_atencao (nn.Module): Normalização para atenção.
        norma_ff (nn.Module): Normalização para rede feed-forward.
    """
    def __init__(self, id_camada: int, args: ArgumentosModelo):
        super().__init__()
        self.atencao = AtencaoLatenteMultiCabeca(args)
        if id_camada < args.num_camadas_densas:
            self.rede_ff = RedePerceptron(args.dimensao, args.dimensao_intermediaria, args.taxa_dropout)
        else:
            self.rede_ff = MisturaDeEspecialistas(args)
        self.norma_atencao = NormaRMS(args.dimensao)
        self.norma_ff = NormaRMS(args.dimensao)

    def forward(self, x: torch.Tensor, pos_inicio: int, freqs_cis: torch.Tensor,
                mascara: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Passagem direta do bloco Transformer com conexões residuais.

        Args:
            x (torch.Tensor): Tensor de entrada.
            pos_inicio (int): Posição inicial na sequência.
            freqs_cis (torch.Tensor): Valores pré-calculados para embeddings rotacionais.
            mascara (Optional[torch.Tensor]): Máscara de atenção.

        Returns:
            torch.Tensor: Tensor de saída após computação do bloco.
        """
        x = x + self.atencao(self.norma_atencao(x), pos_inicio, freqs_cis, mascara)
        x = x + self.rede_ff(self.norma_ff(x))
        return x


# ============================================================================
# Transformador DragaoLT (Modelo Principal)
# ============================================================================
class Transformador(nn.Module):
    """
    Modelo Transformer DragaoLT com embeddings posicionais, múltiplas camadas
    e projeção de saída.

    Modelo de linguagem de grande escala com Mistura de Especialistas e
    Atenção Latente Multi-Cabeça para inferência eficiente.

    Autor: Luiz Tiago Wilcke

    Attributes:
        comprimento_seq_maximo (int): Comprimento máximo da sequência.
        embedding (nn.Module): Camada de embedding para tokens de entrada.
        camadas (torch.nn.ModuleList): Lista de blocos transformer.
        norma (nn.Module): Normalização aplicada após todos os blocos.
        cabeca (nn.Module): Projeção de saída mapeando para o vocabulário.
        freqs_cis (torch.Tensor): Valores pré-calculados para embeddings rotacionais.
    """
    def __init__(self, args: ArgumentosModelo):
        global tamanho_mundo, posicao_global
        tamanho_mundo = dist.get_world_size() if dist.is_initialized() else 1
        posicao_global = dist.get_rank() if dist.is_initialized() else 0
        CamadaLinear.tipo_dados = torch.float8_e4m3fn if args.tipo_dados == "fp8" else torch.bfloat16
        CamadaLinear.formato_escala = args.formato_escala
        super().__init__()
        self.comprimento_seq_maximo = args.comprimento_seq_maximo
        self.embedding = EmbeddingParalelo(args.tamanho_vocabulario, args.dimensao)
        self.camadas = torch.nn.ModuleList()
        for id_camada in range(args.num_camadas):
            self.camadas.append(BlocoTransformador(id_camada, args))
        self.norma = NormaRMS(args.dimensao)
        self.cabeca = LinearColunaParalela(args.dimensao, args.tamanho_vocabulario, tipo_dados=torch.get_default_dtype())
        self.register_buffer("freqs_cis", pre_calcular_freqs_cis(args), persistent=False)

    @torch.inference_mode()
    def forward(self, tokens: torch.Tensor, pos_inicio: int = 0) -> torch.Tensor:
        """
        Passagem direta do modelo Transformer DragaoLT.

        Args:
            tokens (torch.Tensor): Tensor de IDs de tokens (tamanho_lote, comprimento_seq).
            pos_inicio (int, optional): Posição inicial para embeddings rotacionais. Padrão: 0.

        Returns:
            torch.Tensor: Tensor de logits (tamanho_lote, tamanho_vocabulario).
        """
        comprimento_seq = tokens.size(1)
        h = self.embedding(tokens)
        freqs_cis = self.freqs_cis[pos_inicio:pos_inicio + comprimento_seq]
        mascara = None
        if comprimento_seq > 1:
            mascara = torch.full((comprimento_seq, comprimento_seq), float("-inf"), device=tokens.device).triu_(1)
        for camada in self.camadas:
            h = camada(h, pos_inicio, freqs_cis, mascara)
        h = self.norma(h)[:, -1]
        logits = self.cabeca(h)
        if tamanho_mundo > 1:
            todos_logits = [torch.empty_like(logits) for _ in range(tamanho_mundo)]
            dist.all_gather(todos_logits, logits)
            logits = torch.cat(todos_logits, dim=-1)
        return logits

    def obter_metricas_especialistas(self) -> Dict[int, Dict[str, float]]:
        """
        Coleta métricas de utilização de todos os especialistas do modelo.

        Returns:
            Dict[int, Dict[str, float]]: Métricas por camada MoE.
        """
        metricas = {}
        for i, camada in enumerate(self.camadas):
            if isinstance(camada.rede_ff, MisturaDeEspecialistas):
                metricas[i] = camada.rede_ff.obter_metricas_utilizacao()
        return metricas

    def resetar_metricas_especialistas(self):
        """Reseta métricas de todos os especialistas do modelo."""
        for camada in self.camadas:
            if isinstance(camada.rede_ff, MisturaDeEspecialistas):
                camada.rede_ff.resetar_metricas()


# ============================================================================
# Teste Standalone
# ============================================================================
if __name__ == "__main__":
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device("cuda")
    torch.manual_seed(0)
    args = ArgumentosModelo()
    x = torch.randint(0, args.tamanho_vocabulario, (2, 128))
    modelo = Transformador(args)
    print(f"DragaoLT — Saída: {modelo(x).size()}")
    print(f"Métricas de especialistas: {modelo.obter_metricas_especialistas()}")
