"""
DragaoLT — Conversor FP8 para BF16
Autor: Luiz Tiago Wilcke

Converte pesos FP8 (float8_e4m3fn) para BF16 (bfloat16) usando
dequantização por blocos com kernels Triton otimizados.
"""

import os
import json
from argparse import ArgumentParser
from glob import glob
from tqdm import tqdm

import torch
from safetensors.torch import load_file, save_file

from nucleo import dequantizar_pesos


def principal(caminho_fp8: str, caminho_bf16: str) -> None:
    """
    Converte pesos FP8 para BF16 e salva os pesos convertidos.

    Lê os pesos FP8 do diretório especificado, aplica dequantização
    usando fatores de escala, e salva os pesos BF16 resultantes.

    Args:
        caminho_fp8 (str): Caminho para o diretório com pesos FP8.
        caminho_bf16 (str): Caminho para salvar os pesos BF16 convertidos.

    Raises:
        KeyError: Se um tensor de escala necessário estiver ausente.
    """
    torch.set_default_dtype(torch.bfloat16)
    os.makedirs(caminho_bf16, exist_ok=True)

    arquivo_indice = os.path.join(caminho_fp8, "model.safetensors.index.json")
    with open(arquivo_indice, "r") as f:
        indice_modelo = json.load(f)
    mapa_pesos = indice_modelo["weight_map"]

    # Cache de arquivos safetensor carregados
    arquivos_carregados = {}
    nomes_pesos_fp8 = []

    def obter_tensor(nome_tensor: str) -> torch.Tensor:
        """
        Recupera um tensor dos arquivos safetensor em cache ou do disco.

        Args:
            nome_tensor (str): Nome do tensor a recuperar.

        Returns:
            torch.Tensor: Tensor recuperado.

        Raises:
            KeyError: Se o tensor não existir no arquivo safetensor.
        """
        nome_arquivo = mapa_pesos[nome_tensor]
        if nome_arquivo not in arquivos_carregados:
            caminho_arquivo = os.path.join(caminho_fp8, nome_arquivo)
            arquivos_carregados[nome_arquivo] = load_file(caminho_arquivo, device="cuda")
        return arquivos_carregados[nome_arquivo][nome_tensor]

    arquivos_safetensor = sorted(glob(os.path.join(caminho_fp8, "*.safetensors")))

    for arquivo_safetensor in tqdm(arquivos_safetensor, desc="Convertendo FP8 → BF16"):
        nome_arquivo = os.path.basename(arquivo_safetensor)
        dict_estado_atual = load_file(arquivo_safetensor, device="cuda")
        arquivos_carregados[nome_arquivo] = dict_estado_atual

        novo_dict_estado = {}
        for nome_peso, peso in dict_estado_atual.items():
            if nome_peso.endswith("_scale_inv"):
                continue
            elif peso.element_size() == 1:  # Peso FP8
                nome_escala_inv = f"{nome_peso}_scale_inv"
                try:
                    escala_inv = obter_tensor(nome_escala_inv)
                    nomes_pesos_fp8.append(nome_peso)
                    novo_dict_estado[nome_peso] = dequantizar_pesos(peso, escala_inv)
                except KeyError:
                    print(f"Aviso: Tensor de escala ausente para {nome_peso}, pulando conversão")
                    novo_dict_estado[nome_peso] = peso
            else:
                novo_dict_estado[nome_peso] = peso

        novo_arquivo = os.path.join(caminho_bf16, nome_arquivo)
        save_file(novo_dict_estado, novo_arquivo)

        # Gerenciamento de memória: manter apenas os 2 arquivos mais recentes
        if len(arquivos_carregados) > 2:
            arquivo_mais_antigo = next(iter(arquivos_carregados))
            del arquivos_carregados[arquivo_mais_antigo]
            torch.cuda.empty_cache()

    # Atualizar índice do modelo
    novo_arquivo_indice = os.path.join(caminho_bf16, "model.safetensors.index.json")
    for nome_peso in nomes_pesos_fp8:
        nome_escala_inv = f"{nome_peso}_scale_inv"
        if nome_escala_inv in mapa_pesos:
            mapa_pesos.pop(nome_escala_inv)
    with open(novo_arquivo_indice, "w") as f:
        json.dump({"metadata": {}, "weight_map": mapa_pesos}, f, indent=2)

    print(f"\nConversão FP8→BF16 concluída! Pesos salvos em: {caminho_bf16}")


if __name__ == "__main__":
    parser = ArgumentParser(description="DragaoLT — Conversor FP8→BF16 | Autor: Luiz Tiago Wilcke")
    parser.add_argument("--caminho-fp8", type=str, required=True,
                        help="Caminho para o diretório com pesos FP8")
    parser.add_argument("--caminho-bf16", type=str, required=True,
                        help="Caminho para salvar os pesos BF16 convertidos")
    args = parser.parse_args()
    principal(args.caminho_fp8, args.caminho_bf16)
