# -*- coding: utf-8 -*-
"""
API FastAPI para o Sistema de Matching WorkSphere.

Expõe endpoints para interagir com a lógica de matching.
"""

from fastapi import FastAPI, HTTPException, Path, Query
from typing import List, Dict, Union, Optional
import uvicorn
import logging

# Configura logging básico
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Importa a função de matching e os dados necessários
try:
    # A função inicializar_matcher() é chamada quando matching_logic é importado.
    from matching_logic import encontrar_matches_para_trabalhador, trabalhadores, inicializar_matcher
    logger.info("Módulo matching_logic importado com sucesso.")
except ImportError as e:
    logger.error(f"Erro ao importar matching_logic: {e}")
    # Define funções dummy ou lança erro para indicar falha crítica
    def encontrar_matches_para_trabalhador(*args, **kwargs):
        return {"erro": "Falha crítica ao carregar a lógica de matching."}
    trabalhadores = []
    def inicializar_matcher():
        pass
except Exception as e:
    logger.error(f"Erro inesperado durante a importação ou inicialização: {e}")
    raise

# Cria a instância da aplicação FastAPI
app = FastAPI(
    title="WorkSphere Matching API",
    description="API para encontrar matches entre trabalhadores e vagas usando TF-IDF.",
    version="0.1.0"
)

# --- Eventos de Ciclo de Vida da Aplicação ---
@app.on_event("startup")
async def startup_event():
    """Função executada na inicialização da API."""
    logger.info("Iniciando a API WorkSphere Matching...")
    # Garante que o matcher seja inicializado se ainda não foi
    # (embora já deva ter sido na importação)
    try:
        inicializar_matcher() # Chama novamente para garantir ou logar status
        logger.info("Verificação de inicialização do Matcher concluída.")
    except Exception as e:
        logger.error(f"Erro durante a inicialização no startup: {e}")

# --- Endpoints da API ---

# Modelo de resposta para documentação (usando Dict/List por simplicidade agora)
# from pydantic import BaseModel
# class MatchResponse(BaseModel):
#     id_vaga: str
#     titulo_vaga: str
#     empresa: str
#     similaridade: float

@app.get(
    "/match/trabalhador/{id_trabalhador}",
    response_model=Union[List[Dict[str, Union[str, float]]], Dict[str, str]], # Define o tipo de retorno esperado
    summary="Encontra vagas compatíveis para um trabalhador",
    tags=["Matching"] # Agrupa endpoints na documentação interativa (/docs)
)
async def get_matches_para_trabalhador(
    id_trabalhador: str = Path(..., title="ID do Trabalhador", description="O ID único do trabalhador (ex: 't1', 't2')."),
    top_n: Optional[int] = Query(3, title="Número de Matches", description="Quantidade máxima de vagas similares a retornar.", ge=1)
):
    """
    Recebe o ID de um trabalhador e retorna uma lista das `top_n` vagas mais compatíveis.

    A compatibilidade é calculada usando similaridade de cosseno sobre vetores TF-IDF
    gerados a partir das habilidades/experiência do trabalhador e requisitos/descrição das vagas.

    - **id_trabalhador**: ID do trabalhador (obrigatório).
    - **top_n**: Número máximo de resultados (opcional, padrão 3).
    """
    logger.info(f"Recebida requisição de match para trabalhador ID: {id_trabalhador}, top_n: {top_n}")

    # Verifica se o trabalhador existe nos dados carregados
    if not any(t["id"] == id_trabalhador for t in trabalhadores):
        logger.warning(f"Trabalhador com ID 	'{id_trabalhador}	' não encontrado.")
        raise HTTPException(status_code=404, detail=f"Trabalhador com ID 	'{id_trabalhador}	' não encontrado.")

    # Chama a função de lógica de matching
    try:
        matches = encontrar_matches_para_trabalhador(id_trabalhador, top_n=top_n)
        logger.info(f"Matches encontrados para {id_trabalhador}: {len(matches) if isinstance(matches, list) else 'Erro'}")

        # Verifica se houve erro interno retornado pela lógica
        if isinstance(matches, dict) and "erro" in matches:
            logger.error(f"Erro interno ao buscar matches para {id_trabalhador}: {matches['erro']}")
            raise HTTPException(status_code=500, detail=matches["erro"]) # Erro interno do servidor

        # Retorna os matches encontrados (pode ser uma lista vazia)
        return matches

    except HTTPException as http_exc:
        # Re-levanta exceções HTTP para que o FastAPI as trate
        raise http_exc
    except Exception as e:
        # Captura qualquer outro erro inesperado durante o processamento
        logger.exception(f"Erro inesperado ao processar matches para {id_trabalhador}: {e}")
        raise HTTPException(status_code=500, detail="Ocorreu um erro interno inesperado no servidor.")

@app.get("/health", summary="Verifica a saúde da API", tags=["Status"])
async def health_check():
    """Endpoint simples para verificar se a API está rodando."""
    return {"status": "ok"}

# --- Execução para Desenvolvimento Local ---
if __name__ == "__main__":
    # Roda o servidor Uvicorn. 'api:app' refere-se ao arquivo api.py e à instância app.
    # host="0.0.0.0" permite acesso de fora do sandbox/container.
    # reload=True reinicia o servidor automaticamente quando o código é alterado (ótimo para dev).
    logger.info("Iniciando servidor Uvicorn para desenvolvimento local em http://0.0.0.0:8000")
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)

