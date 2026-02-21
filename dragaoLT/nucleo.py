"""
DragaoLT — Kernels Triton para Operações de Alta Performance
Autor: Luiz Tiago Wilcke

Módulo de kernels Triton para quantização FP8, dequantização de pesos
e multiplicação de matrizes em precisão FP8 com fatores de escala.
"""

from typing import Tuple, Optional

import torch
import triton
import triton.language as tl
from triton import Config


@triton.jit
def nucleo_quantizacao_ativacao(ptr_entrada, ptr_saida, ptr_escala, TAMANHO_BLOCO: tl.constexpr, formato_escala: tl.constexpr):
    """
    Kernel Triton para quantização de ativações em blocos.

    Quantiza o tensor de entrada `ptr_entrada` e armazena o resultado em `ptr_saida`
    com o fator de escala correspondente em `ptr_escala`.

    Args:
        ptr_entrada (triton.Pointer): Ponteiro para o tensor de entrada.
        ptr_saida (triton.Pointer): Ponteiro para o tensor de saída quantizado.
        ptr_escala (triton.Pointer): Ponteiro para os fatores de escala.
        TAMANHO_BLOCO (tl.constexpr): Tamanho do bloco para processamento.
        formato_escala (tl.constexpr): Formato da escala (ex: "ue8m0").
    """
    id_programa = tl.program_id(axis=0)
    deslocamentos = id_programa * TAMANHO_BLOCO + tl.arange(0, TAMANHO_BLOCO)
    entrada = tl.load(ptr_entrada + deslocamentos).to(tl.float32)
    valor_maximo_abs = tl.max(tl.abs(entrada))
    valor_maximo_abs = tl.maximum(valor_maximo_abs, 1e-4)
    escala = valor_maximo_abs / 448.0
    if formato_escala == "ue8m0":
        expoente = tl.math.ceil(tl.math.log2(escala))
        escala = tl.math.exp2(expoente)
    saida = entrada / escala
    saida = saida.to(ptr_saida.dtype.element_ty)
    tl.store(ptr_saida + deslocamentos, saida)
    tl.store(ptr_escala + id_programa, escala)


def quantizar_ativacao(entrada: torch.Tensor, tamanho_bloco: int = 128, formato_escala: Optional[str] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantiza o tensor de entrada usando quantização por blocos.

    Aplica quantização FP8 (float8_e4m3fn) ao tensor de entrada, dividindo-o
    em blocos e calculando um fator de escala por bloco.

    Args:
        entrada (torch.Tensor): Tensor de entrada. Deve ser contíguo e sua última
            dimensão deve ser divisível por `tamanho_bloco`.
        tamanho_bloco (int): Tamanho dos blocos para quantização. Padrão: 128.
        formato_escala (Optional[str]): Formato da escala. Padrão: None.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tupla contendo:
            - Tensor quantizado com dtype `torch.float8_e4m3fn`.
            - Tensor de fatores de escala com dtype `torch.float32`.
    """
    assert entrada.is_contiguous(), 'O tensor de entrada deve ser contíguo'
    assert entrada.size(-1) % tamanho_bloco == 0, f'A última dimensão deve ser divisível por tamanho_bloco ({tamanho_bloco})'
    saida = torch.empty_like(entrada, dtype=torch.float8_e4m3fn)
    escala = entrada.new_empty(*entrada.size()[:-1], entrada.size(-1) // tamanho_bloco, dtype=torch.float32)
    grade = lambda meta: (triton.cdiv(entrada.numel(), meta['TAMANHO_BLOCO']),)
    nucleo_quantizacao_ativacao[grade](entrada, saida, escala, TAMANHO_BLOCO=tamanho_bloco, formato_escala=formato_escala)
    return saida, escala


@triton.jit
def nucleo_dequantizacao_pesos(ptr_entrada, ptr_escala, ptr_saida, M, N, TAMANHO_BLOCO: tl.constexpr):
    """
    Kernel Triton para dequantização de pesos com fatores de escala.

    Dequantiza os pesos multiplicando-os pelos fatores de escala correspondentes
    e armazena o resultado no buffer de saída.

    Args:
        ptr_entrada (tl.pointer): Ponteiro para os pesos quantizados.
        ptr_escala (tl.pointer): Ponteiro para os fatores de escala.
        ptr_saida (tl.pointer): Ponteiro para o buffer de saída.
        M (int): Número de linhas na matriz de pesos.
        N (int): Número de colunas na matriz de pesos.
        TAMANHO_BLOCO (tl.constexpr): Tamanho do bloco para tiling.
    """
    id_m = tl.program_id(axis=0)
    id_n = tl.program_id(axis=1)
    n_blocos = tl.cdiv(N, TAMANHO_BLOCO)
    deslocamentos_m = id_m * TAMANHO_BLOCO + tl.arange(0, TAMANHO_BLOCO)
    deslocamentos_n = id_n * TAMANHO_BLOCO + tl.arange(0, TAMANHO_BLOCO)
    deslocamentos = deslocamentos_m[:, None] * N + deslocamentos_n[None, :]
    mascara = (deslocamentos_m[:, None] < M) & (deslocamentos_n[None, :] < N)
    entrada = tl.load(ptr_entrada + deslocamentos, mask=mascara).to(tl.float32)
    escala = tl.load(ptr_escala + id_m * n_blocos + id_n)
    saida = entrada * escala
    tl.store(ptr_saida + deslocamentos, saida, mask=mascara)


def dequantizar_pesos(entrada: torch.Tensor, escala: torch.Tensor, tamanho_bloco: int = 128) -> torch.Tensor:
    """
    Dequantiza o tensor de pesos usando o tensor de escala fornecido.

    Args:
        entrada (torch.Tensor): Tensor de pesos quantizados com forma (M, N).
        escala (torch.Tensor): Tensor de escala com forma (M//tamanho_bloco, N//tamanho_bloco).
        tamanho_bloco (int): Tamanho do bloco para dequantização. Padrão: 128.

    Returns:
        torch.Tensor: Tensor de pesos dequantizados com a mesma forma que `entrada`.

    Raises:
        AssertionError: Se `entrada` ou `escala` não forem contíguos ou tiverem dimensões != 2.
    """
    assert entrada.is_contiguous() and escala.is_contiguous(), 'Os tensores de entrada devem ser contíguos'
    assert entrada.dim() == 2 and escala.dim() == 2, 'Os tensores de entrada devem ter 2 dimensões'
    M, N = entrada.size()
    saida = torch.empty_like(entrada, dtype=torch.get_default_dtype())
    grade = lambda meta: (triton.cdiv(M, meta['TAMANHO_BLOCO']), triton.cdiv(N, meta['TAMANHO_BLOCO']))
    nucleo_dequantizacao_pesos[grade](entrada, escala, saida, M, N, TAMANHO_BLOCO=tamanho_bloco)
    return saida


# Configurações para auto-tuning do kernel GEMM FP8
configs_gemm_fp8 = [
    Config({'TAMANHO_BLOCO_M': bloco_m, 'TAMANHO_BLOCO_N': bloco_n, 'TAMANHO_BLOCO_K': 128},
           num_stages=num_estagios, num_warps=8)
    for bloco_m in [16, 32, 64]
    for bloco_n in [32, 64, 128]
    for num_estagios in [3, 4, 5, 6]
]


@triton.autotune(configs=configs_gemm_fp8, key=['N', 'K'])
@triton.jit
def nucleo_gemm_fp8(ptr_a, ptr_b, ptr_c,
                    ptr_escala_a, ptr_escala_b,
                    M, N: tl.constexpr, K: tl.constexpr,
                    TAMANHO_BLOCO_M: tl.constexpr,
                    TAMANHO_BLOCO_N: tl.constexpr,
                    TAMANHO_BLOCO_K: tl.constexpr):
    """
    Kernel Triton para multiplicação de matrizes em precisão FP8 com fatores de escala.

    Realiza a operação C = A @ B^T com escalonamento por blocos, onde A e B
    estão em formato FP8 com fatores de escala associados.

    Args:
        ptr_a (tl.tensor): Ponteiro para a matriz A de entrada.
        ptr_b (tl.tensor): Ponteiro para a matriz B de entrada.
        ptr_c (tl.tensor): Ponteiro para a matriz C de saída.
        ptr_escala_a (tl.tensor): Ponteiro para os fatores de escala de A.
        ptr_escala_b (tl.tensor): Ponteiro para os fatores de escala de B.
        M (int): Número de linhas em A e C.
        N (tl.constexpr): Número de colunas em B e C.
        K (tl.constexpr): Número de colunas em A e linhas em B.
        TAMANHO_BLOCO_M (tl.constexpr): Tamanho do bloco na dimensão M.
        TAMANHO_BLOCO_N (tl.constexpr): Tamanho do bloco na dimensão N.
        TAMANHO_BLOCO_K (tl.constexpr): Tamanho do bloco na dimensão K.
    """
    id_m = tl.program_id(axis=0)
    id_n = tl.program_id(axis=1)
    k_blocos = tl.cdiv(K, TAMANHO_BLOCO_K)
    deslocamentos_m = (id_m * TAMANHO_BLOCO_M + tl.arange(0, TAMANHO_BLOCO_M)) % M
    deslocamentos_n = (id_n * TAMANHO_BLOCO_N + tl.arange(0, TAMANHO_BLOCO_N)) % N
    deslocamentos_k = tl.arange(0, TAMANHO_BLOCO_K)
    ptrs_a = ptr_a + deslocamentos_m[:, None] * K + deslocamentos_k[None, :]
    ptrs_b = ptr_b + deslocamentos_n[None, :] * K + deslocamentos_k[:, None]
    ptrs_escala_a = ptr_escala_a + deslocamentos_m * k_blocos
    ptrs_escala_b = ptr_escala_b + (deslocamentos_n // TAMANHO_BLOCO_K) * k_blocos

    acumulador = tl.zeros((TAMANHO_BLOCO_M, TAMANHO_BLOCO_N), dtype=tl.float32)
    for i in range(k_blocos):
        a = tl.load(ptrs_a, mask=deslocamentos_k[None, :] < K - i * TAMANHO_BLOCO_K, other=0.0)
        b = tl.load(ptrs_b, mask=deslocamentos_k[:, None] < K - i * TAMANHO_BLOCO_K, other=0.0)
        escala_a = tl.load(ptrs_escala_a)
        escala_b = tl.load(ptrs_escala_b)
        acumulador += tl.dot(a, b) * escala_a[:, None] * escala_b[None, :]
        ptrs_a += TAMANHO_BLOCO_K
        ptrs_b += TAMANHO_BLOCO_K
        ptrs_escala_a += 1
        ptrs_escala_b += 1
    c = acumulador.to(ptr_c.dtype.element_ty)
    deslocamentos_m = id_m * TAMANHO_BLOCO_M + tl.arange(0, TAMANHO_BLOCO_M)
    deslocamentos_n = id_n * TAMANHO_BLOCO_N + tl.arange(0, TAMANHO_BLOCO_N)
    ptrs_c = ptr_c + deslocamentos_m[:, None] * N + deslocamentos_n[None, :]
    mascara = (deslocamentos_m[:, None] < M) & (deslocamentos_n[None, :] < N)
    tl.store(ptrs_c, c, mask=mascara)


def gemm_fp8(a: torch.Tensor, escala_a: torch.Tensor, b: torch.Tensor, escala_b: torch.Tensor) -> torch.Tensor:
    """
    Realiza multiplicação de matrizes em precisão FP8.

    Executa a operação C = A @ B^T utilizando aritmética FP8 com
    fatores de escala por bloco para manter a precisão numérica.

    Args:
        a (torch.Tensor): Primeira matriz de entrada (deve ser contígua).
        escala_a (torch.Tensor): Fator de escala para a matriz A (deve ser contíguo).
        b (torch.Tensor): Segunda matriz de entrada (deve ser contígua).
        escala_b (torch.Tensor): Fator de escala para a matriz B (deve ser contíguo).

    Returns:
        torch.Tensor: Resultado da multiplicação de matrizes.
    """
    assert a.is_contiguous() and b.is_contiguous(), 'Os tensores de entrada devem ser contíguos'
    assert escala_a.is_contiguous() and escala_b.is_contiguous(), 'Os tensores de escala devem ser contíguos'
    K = a.size(-1)
    M = a.numel() // K
    N = b.size(0)
    c = a.new_empty(*a.size()[:-1], N, dtype=torch.get_default_dtype())
    grade = lambda META: (triton.cdiv(M, META['TAMANHO_BLOCO_M']), triton.cdiv(N, META['TAMANHO_BLOCO_N']))
    nucleo_gemm_fp8[grade](a, b, c, escala_a, escala_b, M, N, K)
    return c
