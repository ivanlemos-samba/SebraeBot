import re
from pathlib import Path

PASTA_ENTRADA = Path("./legendas")
PASTA_SAIDA = Path("./transcricao-texto")

PASTA_SAIDA.mkdir(exist_ok=True)

for arquivo_srt in PASTA_ENTRADA.glob("*.srt"):

    linhas_texto = []

    with open(arquivo_srt, "r", encoding="utf-8-sig") as f:

        for linha in f:
            linha = linha.strip()

            if not linha:
                continue

            # Remove numeração do bloco
            if linha.isdigit():
                continue

            # Remove timestamps
            if "-->" in linha:
                continue

            linhas_texto.append(linha)

    texto_final = " ".join(linhas_texto)

    # Normaliza espaços
    texto_final = re.sub(r"\s+", " ", texto_final)

    arquivo_saida = PASTA_SAIDA / f"{arquivo_srt.stem}.txt"

    with open(arquivo_saida, "w", encoding="utf-8") as f:
        f.write(texto_final)

    print(f"✅ {arquivo_srt.name}")

print("🎉 Conversão concluída")