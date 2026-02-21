"""
DragaoLT — Conversor de Checkpoints
Autor: Luiz Tiago Wilcke

Converte checkpoints do formato Hugging Face para o formato DragaoLT,
com suporte a paralelismo de modelo (model parallelism).
"""

import os
import shutil
from argparse import ArgumentParser
from glob import glob
from tqdm import tqdm, trange

import torch
from safetensors.torch import safe_open, save_file


# Mapeamento de nomes HuggingFace → DragaoLT
mapeamento = {
    "embed_tokens": ("embedding", 0),
    "input_layernorm": ("norma_atencao", None),
    "post_attention_layernorm": ("norma_ff", None),
    "q_proj": ("wq", 0),
    "q_a_proj": ("wq_a", None),
    "q_a_layernorm": ("norma_q", None),
    "q_b_proj": ("wq_b", 0),
    "kv_a_proj_with_mqa": ("wkv_a", None),
    "kv_a_layernorm": ("norma_kv", None),
    "kv_b_proj": ("wkv_b", 0),
    "o_proj": ("wo", 1),
    "gate": ("portao", None),
    "gate_proj": ("w1", 0),
    "down_proj": ("w2", 1),
    "up_proj": ("w3", 0),
    "norm": ("norma", None),
    "lm_head": ("cabeca", 0),
    "scale": ("escala", None),
}


def principal(caminho_hf: str, caminho_saida: str, num_especialistas: int, mp: int) -> None:
    """
    Converte e salva arquivos de checkpoint no formato DragaoLT.

    Realiza a conversão dos pesos do formato Hugging Face para o formato
    interno do DragaoLT, com suporte a particionamento para paralelismo
    de modelo (model parallelism).

    Args:
        caminho_hf (str): Caminho para o diretório com checkpoints de entrada.
        caminho_saida (str): Caminho para salvar os checkpoints convertidos.
        num_especialistas (int): Número total de especialistas no modelo.
        mp (int): Fator de paralelismo do modelo.
    """
    torch.set_num_threads(8)
    num_especialistas_locais = num_especialistas // mp
    dicts_estados = [{} for _ in range(mp)]

    for caminho_arquivo in tqdm(glob(os.path.join(caminho_hf, "*.safetensors")),
                                 desc="Convertendo checkpoints"):
        with safe_open(caminho_arquivo, framework="pt", device="cpu") as f:
            for nome in f.keys():
                if "model.layers.61" in nome:
                    continue
                parametro: torch.Tensor = f.get_tensor(nome)
                if nome.startswith("model."):
                    nome = nome[len("model."):]
                nome = nome.replace("self_attn", "atencao")
                nome = nome.replace("mlp", "rede_ff")
                nome = nome.replace("weight_scale_inv", "escala")
                nome = nome.replace("e_score_correction_bias", "vies")
                chave = nome.split(".")[-2]
                assert chave in mapeamento, f"Chave '{chave}' não encontrada no mapeamento"
                nova_chave, dim = mapeamento[chave]
                nome = nome.replace(chave, nova_chave)
                for i in range(mp):
                    novo_parametro = parametro
                    if "especialistas" in nome and "especialistas_compartilhados" not in nome:
                        idx = int(nome.split(".")[-3])
                        if idx < i * num_especialistas_locais or idx >= (i + 1) * num_especialistas_locais:
                            continue
                    elif dim is not None:
                        assert parametro.size(dim) % mp == 0, \
                            f"A dimensão {dim} deve ser divisível por {mp}"
                        tamanho_fatia = parametro.size(dim) // mp
                        novo_parametro = parametro.narrow(dim, i * tamanho_fatia, tamanho_fatia).contiguous()
                    dicts_estados[i][nome] = novo_parametro

    os.makedirs(caminho_saida, exist_ok=True)

    for i in trange(mp, desc="Salvando checkpoints"):
        save_file(dicts_estados[i], os.path.join(caminho_saida, f"model{i}-mp{mp}.safetensors"))

    for caminho_arquivo in glob(os.path.join(caminho_hf, "*token*")):
        novo_caminho = os.path.join(caminho_saida, os.path.basename(caminho_arquivo))
        shutil.copyfile(caminho_arquivo, novo_caminho)

    print(f"\nConversão concluída! Checkpoints salvos em: {caminho_saida}")


if __name__ == "__main__":
    parser = ArgumentParser(description="DragaoLT — Conversor de Checkpoints | Autor: Luiz Tiago Wilcke")
    parser.add_argument("--caminho-hf", type=str, required=True,
                        help="Caminho para os checkpoints Hugging Face")
    parser.add_argument("--caminho-saida", type=str, required=True,
                        help="Caminho para salvar os checkpoints convertidos")
    parser.add_argument("--num-especialistas", type=int, required=True,
                        help="Número total de especialistas no modelo")
    parser.add_argument("--paralelismo-modelo", type=int, required=True,
                        help="Fator de paralelismo do modelo")
    args = parser.parse_args()
    assert args.num_especialistas % args.paralelismo_modelo == 0, \
        "O número de especialistas deve ser divisível pelo paralelismo do modelo"
    principal(args.caminho_hf, args.caminho_saida, args.num_especialistas, args.paralelismo_modelo)
