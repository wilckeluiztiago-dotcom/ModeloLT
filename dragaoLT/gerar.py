"""
DragaoLT — Script de Geração e Inferência
Autor: Luiz Tiago Wilcke

Script para geração de texto utilizando o modelo DragaoLT.
Suporta modo interativo (chat) e processamento em lote.
"""

import os
import json
from argparse import ArgumentParser
from typing import List

import torch
import torch.distributed as dist
from transformers import AutoTokenizer
from safetensors.torch import load_model

from modelo import Transformador, ArgumentosModelo


def amostrar(logits: torch.Tensor, temperatura: float = 1.0) -> torch.Tensor:
    """
    Amostra um token a partir dos logits usando escalonamento por temperatura.

    Aplica a técnica de amostragem Gumbel-max para seleção estocástica de tokens,
    controlada pela temperatura para ajustar a diversidade da geração.

    Args:
        logits (torch.Tensor): Tensor de logits para predição de tokens.
        temperatura (float, optional): Temperatura para escalonamento. Padrão: 1.0.
            Valores mais altos produzem texto mais diverso/criativo.
            Valores mais baixos produzem texto mais determinístico.

    Returns:
        torch.Tensor: Token amostrado.
    """
    logits = logits / max(temperatura, 1e-5)
    probabilidades = torch.softmax(logits, dim=-1)
    return probabilidades.div_(torch.empty_like(probabilidades).exponential_(1)).argmax(dim=-1)


@torch.inference_mode()
def gerar(
    modelo: Transformador,
    tokens_prompt: List[List[int]],
    max_novos_tokens: int,
    id_fim_sequencia: int,
    temperatura: float = 1.0
) -> List[List[int]]:
    """
    Gera novos tokens baseados nos tokens de prompt fornecidos.

    Utiliza o modelo DragaoLT para geração autoregressiva de tokens,
    com suporte a múltiplos prompts simultâneos (batching).

    Args:
        modelo (Transformador): Modelo DragaoLT para geração.
        tokens_prompt (List[List[int]]): Lista de listas com tokens do prompt.
        max_novos_tokens (int): Número máximo de novos tokens a gerar.
        id_fim_sequencia (int): ID do token de fim de sequência.
        temperatura (float, optional): Temperatura para amostragem. Padrão: 1.0.

    Returns:
        List[List[int]]: Lista de listas com os tokens gerados para cada sequência.
    """
    comprimentos_prompt = [len(t) for t in tokens_prompt]
    assert max(comprimentos_prompt) <= modelo.comprimento_seq_maximo, \
        f"Comprimento do prompt excede o máximo da sequência ({modelo.comprimento_seq_maximo})"

    comprimento_total = min(modelo.comprimento_seq_maximo, max_novos_tokens + max(comprimentos_prompt))
    tokens = torch.full((len(tokens_prompt), comprimento_total), -1, dtype=torch.long, device="cuda")
    for i, t in enumerate(tokens_prompt):
        tokens[i, :len(t)] = torch.tensor(t, dtype=torch.long, device="cuda")

    pos_anterior = 0
    finalizado = torch.tensor([False] * len(tokens_prompt), device="cuda")
    mascara_prompt = tokens != -1

    for pos_atual in range(min(comprimentos_prompt), comprimento_total):
        logits = modelo.forward(tokens[:, pos_anterior:pos_atual], pos_anterior)
        if temperatura > 0:
            proximo_token = amostrar(logits, temperatura)
        else:
            proximo_token = logits.argmax(dim=-1)
        proximo_token = torch.where(mascara_prompt[:, pos_atual], tokens[:, pos_atual], proximo_token)
        tokens[:, pos_atual] = proximo_token
        finalizado |= torch.logical_and(~mascara_prompt[:, pos_atual], proximo_token == id_fim_sequencia)
        pos_anterior = pos_atual
        if finalizado.all():
            break

    tokens_completados = []
    for i, toks in enumerate(tokens.tolist()):
        toks = toks[comprimentos_prompt[i]:comprimentos_prompt[i] + max_novos_tokens]
        if id_fim_sequencia in toks:
            toks = toks[:toks.index(id_fim_sequencia)]
        tokens_completados.append(toks)
    return tokens_completados


def principal(
    caminho_checkpoint: str,
    config: str,
    arquivo_entrada: str = "",
    interativo: bool = True,
    max_novos_tokens: int = 100,
    temperatura: float = 1.0,
) -> None:
    """
    Função principal para carregar o modelo DragaoLT e realizar geração de texto.

    Suporta dois modos de operação:
    - Modo interativo: conversa em tempo real via terminal
    - Modo em lote: processa múltiplos prompts de um arquivo

    Args:
        caminho_checkpoint (str): Caminho para o diretório do checkpoint do modelo.
        config (str): Caminho para o arquivo de configuração JSON.
        arquivo_entrada (str, optional): Arquivo com prompts para processamento em lote.
        interativo (bool, optional): Modo interativo (chat). Padrão: True.
        max_novos_tokens (int, optional): Máximo de novos tokens. Padrão: 100.
        temperatura (float, optional): Temperatura para amostragem. Padrão: 1.0.
    """
    tamanho_mundo = int(os.getenv("WORLD_SIZE", "1"))
    posicao = int(os.getenv("RANK", "0"))
    posicao_local = int(os.getenv("LOCAL_RANK", "0"))
    if tamanho_mundo > 1:
        dist.init_process_group("nccl")
    global print
    if posicao != 0:
        print = lambda *_, **__: None

    torch.cuda.set_device(posicao_local)
    torch.set_default_dtype(torch.bfloat16)
    torch.set_num_threads(8)
    torch.manual_seed(965)

    with open(config) as f:
        args = ArgumentosModelo(**json.load(f))
    print(f"DragaoLT — Configuração carregada: {args}")

    with torch.device("cuda"):
        modelo = Transformador(args)
    tokenizador = AutoTokenizer.from_pretrained(caminho_checkpoint)

    # Aquecimento do modelo
    tokenizador.decode(gerar(modelo, [tokenizador.encode("DragaoLT")], 2, -1, 1.0)[0])
    load_model(modelo, os.path.join(caminho_checkpoint, f"model{posicao}-mp{tamanho_mundo}.safetensors"))

    if interativo:
        print("=" * 60)
        print("  DragaoLT — Modelo de Linguagem")
        print("  Autor: Luiz Tiago Wilcke")
        print("  Comandos: /sair para sair, /limpar para limpar histórico")
        print("=" * 60)
        mensagens = []
        while True:
            if tamanho_mundo == 1:
                prompt = input(">>> ")
            elif posicao == 0:
                prompt = input(">>> ")
                objetos = [prompt]
                dist.broadcast_object_list(objetos, 0)
            else:
                objetos = [None]
                dist.broadcast_object_list(objetos, 0)
                prompt = objetos[0]

            if prompt == "/sair":
                print("Encerrando DragaoLT. Até a próxima!")
                break
            elif prompt == "/limpar":
                mensagens.clear()
                print("Histórico limpo.")
                continue

            mensagens.append({"role": "user", "content": prompt})
            tokens_prompt = tokenizador.apply_chat_template(mensagens, add_generation_prompt=True)
            tokens_completados = gerar(modelo, [tokens_prompt], max_novos_tokens, tokenizador.eos_token_id, temperatura)
            completacao = tokenizador.decode(tokens_completados[0], skip_special_tokens=True)
            print(completacao)
            mensagens.append({"role": "assistant", "content": completacao})
    else:
        with open(arquivo_entrada) as f:
            prompts = [linha.strip() for linha in f.readlines()]
        assert len(prompts) <= args.tamanho_lote_maximo, \
            f"Número de prompts excede o tamanho máximo do lote ({args.tamanho_lote_maximo})"
        tokens_prompt = [
            tokenizador.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True)
            for prompt in prompts
        ]
        tokens_completados = gerar(modelo, tokens_prompt, max_novos_tokens, tokenizador.eos_token_id, temperatura)
        completacoes = tokenizador.batch_decode(tokens_completados, skip_special_tokens=True)
        for prompt, completacao in zip(prompts, completacoes):
            print(f"Prompt: {prompt}")
            print(f"Resposta: {completacao}")
            print()

    if tamanho_mundo > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    """
    Interface de linha de comando para geração de texto distribuída com DragaoLT.

    Argumentos:
        --caminho-checkpoint (str): Caminho para o diretório do checkpoint.
        --config (str): Caminho para o arquivo de configuração JSON.
        --arquivo-entrada (str, optional): Arquivo com prompts para lote.
        --interativo (bool, optional): Ativa modo interativo (chat).
        --max-novos-tokens (int, optional): Máximo de novos tokens. Padrão: 200.
        --temperatura (float, optional): Temperatura para amostragem. Padrão: 0.2.
    """
    parser = ArgumentParser(description="DragaoLT — Gerador de Texto | Autor: Luiz Tiago Wilcke")
    parser.add_argument("--caminho-checkpoint", type=str, required=True,
                        help="Caminho para o diretório do checkpoint do modelo")
    parser.add_argument("--config", type=str, required=True,
                        help="Caminho para o arquivo de configuração JSON")
    parser.add_argument("--arquivo-entrada", type=str, default="",
                        help="Arquivo com prompts para processamento em lote")
    parser.add_argument("--interativo", action="store_true",
                        help="Ativa modo interativo (chat)")
    parser.add_argument("--max-novos-tokens", type=int, default=200,
                        help="Número máximo de novos tokens a gerar")
    parser.add_argument("--temperatura", type=float, default=0.2,
                        help="Temperatura para amostragem (0 = determinístico)")
    args = parser.parse_args()
    assert args.arquivo_entrada or args.interativo, \
        "É necessário especificar --arquivo-entrada ou --interativo"
    principal(
        args.caminho_checkpoint, args.config, args.arquivo_entrada,
        args.interativo, args.max_novos_tokens, args.temperatura
    )
