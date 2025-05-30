# -*- coding: utf-8 -*-
"""
Lógica do Sistema de Matching Trabalhador-Vaga usando TF-IDF e Similaridade de Cosseno.

Este módulo contém as funções para pré-processar dados, calcular a matriz
de similaridade e encontrar matches entre trabalhadores e vagas.
"""

import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np # Import numpy for checking NaN/None in similarity matrix

# --- Dados de Exemplo ---
# Em um sistema real, estes dados viriam de um banco de dados ou outra fonte.
trabalhadores = [
    {
        "id": "t1",
        "nome": "Ana Silva",
        "habilidades": "Python, Machine Learning, Análise de Dados, SQL, Pandas, Scikit-learn",
        "experiencia": "Cientista de Dados Júnior com 2 anos de experiência em projetos de classificação e regressão. Experiência com visualização de dados usando Matplotlib e Seaborn."
    },
    {
        "id": "t2",
        "nome": "Bruno Costa",
        "habilidades": "JavaScript, React, Node.js, HTML, CSS, MongoDB",
        "experiencia": "Desenvolvedor Web Full-Stack com 5 anos de experiência na criação de aplicações web responsivas e escaláveis. Experiência com APIs RESTful."
    },
    {
        "id": "t3",
        "nome": "Carla Dias",
        "habilidades": "Java, Spring Boot, Microserviços, Docker, Kubernetes, AWS",
        "experiencia": "Engenheira de Software Sênior com 8 anos de experiência em desenvolvimento backend e arquitetura de sistemas distribuídos. Foco em soluções cloud-native."
    },
    {
        "id": "t4",
        "nome": "Daniel Souza",
        "habilidades": "Python, Django, Flask, PostgreSQL, Testes Unitários",
        "experiencia": "Desenvolvedor Backend Pleno com 3 anos de experiência em desenvolvimento de APIs web com Python. Conhecimento em bancos de dados relacionais."
    }
]

vagas = [
    {
        "id": "v1",
        "titulo": "Cientista de Dados Pleno",
        "empresa": "InovaTech",
        "requisitos_habilidades": "Python, R, Machine Learning, Deep Learning, SQL, Spark",
        "descricao": "Procuramos um Cientista de Dados para desenvolver modelos preditivos e analisar grandes volumes de dados. Necessário experiência com bibliotecas como Scikit-learn, TensorFlow ou PyTorch."
    },
    {
        "id": "v2",
        "titulo": "Desenvolvedor Frontend Júnior",
        "empresa": "WebSoluções",
        "requisitos_habilidades": "HTML, CSS, JavaScript, React, Git",
        "descricao": "Vaga para Desenvolvedor Frontend para atuar na criação de interfaces de usuário interativas e modernas. Desejável conhecimento em algum framework JS como React ou Vue."
    },
    {
        "id": "v3",
        "titulo": "Engenheiro de DevOps Sênior",
        "empresa": "CloudExperts",
        "requisitos_habilidades": "AWS, Azure, Docker, Kubernetes, Terraform, CI/CD, Python",
        "descricao": "Buscamos Engenheiro de DevOps experiente para automatizar infraestrutura e processos de deploy em ambiente de nuvem. Forte conhecimento em ferramentas de orquestração e IaC."
    },
    {
        "id": "v4",
        "titulo": "Desenvolvedor Python Backend",
        "empresa": "CodeBuilders",
        "requisitos_habilidades": "Python, Django, Flask, REST API, SQL, Docker",
        "descricao": "Oportunidade para Desenvolvedor Python Backend para construir e manter APIs robustas e escaláveis. Experiência com frameworks web e bancos de dados é essencial."
    }
]

# --- Pré-processamento de Texto ---
def preprocessar_texto(texto):
    """Limpa e normaliza o texto para análise."""
    if not isinstance(texto, str):
        return ""
    texto = texto.lower()
    texto = re.sub(r'\W+', ' ', texto)
    texto = re.sub(r'\s+', ' ', texto).strip()
    return texto

# --- Criação dos Documentos para TF-IDF ---
def criar_documento(item, tipo):
    """Combina campos relevantes em um único documento de texto."""
    if tipo == 'trabalhador':
        texto_habilidades = preprocessar_texto(item.get('habilidades', ''))
        texto_experiencia = preprocessar_texto(item.get('experiencia', ''))
        return f"{texto_habilidades} {texto_experiencia}".strip()
    elif tipo == 'vaga':
        texto_requisitos = preprocessar_texto(item.get('requisitos_habilidades', ''))
        texto_descricao = preprocessar_texto(item.get('descricao', ''))
        return f"{texto_requisitos} {texto_descricao}".strip()
    return ""

# --- Vetorização TF-IDF e Cálculo de Similaridade ---
# Global variables to store the fitted vectorizer and similarity matrix
# to avoid recalculating them on every API request.
vectorizer = TfidfVectorizer()
matriz_similaridade_global = None
tfidf_matrix_global = None
docs_trabalhadores_global = []
docs_vagas_global = []

def inicializar_matcher():
    """Prepara os dados e calcula a matriz de similaridade inicial."""
    global vectorizer, matriz_similaridade_global, tfidf_matrix_global, docs_trabalhadores_global, docs_vagas_global

    print("Inicializando o matcher...")
    docs_trabalhadores_global = [criar_documento(t, 'trabalhador') for t in trabalhadores]
    docs_vagas_global = [criar_documento(v, 'vaga') for v in vagas]

    if not docs_trabalhadores_global or not docs_vagas_global:
        print("Erro: Listas de documentos de trabalhadores ou vagas estão vazias durante a inicialização.")
        return

    todos_docs = docs_trabalhadores_global + docs_vagas_global
    try:
        tfidf_matrix_global = vectorizer.fit_transform(todos_docs)
        tfidf_trabalhadores = tfidf_matrix_global[:len(docs_trabalhadores_global)]
        tfidf_vagas = tfidf_matrix_global[len(docs_trabalhadores_global):]
        matriz_similaridade_global = cosine_similarity(tfidf_trabalhadores, tfidf_vagas)
        print("Matcher inicializado com sucesso.")
    except ValueError as e:
        print(f"Erro ao ajustar o vetorizador TF-IDF durante a inicialização: {e}")
        matriz_similaridade_global = None

# --- Função de Matching ---
def encontrar_matches_para_trabalhador(id_trabalhador, top_n=3):
    """Encontra as 'top_n' vagas mais similares para um dado trabalhador."""
    global matriz_similaridade_global

    if matriz_similaridade_global is None or not isinstance(matriz_similaridade_global, np.ndarray):
        print("Erro: Matriz de similaridade não foi inicializada corretamente.")
        # Tenta inicializar se ainda não foi
        inicializar_matcher()
        if matriz_similaridade_global is None or not isinstance(matriz_similaridade_global, np.ndarray):
             return {"erro": "Falha ao inicializar a matriz de similaridade."}

    try:
        indice_trabalhador = next(i for i, t in enumerate(trabalhadores) if t['id'] == id_trabalhador)
    except StopIteration:
        print(f"Erro: Trabalhador com ID '{id_trabalhador}' não encontrado.")
        return {"erro": f"Trabalhador com ID '{id_trabalhador}' não encontrado."}

    if indice_trabalhador >= matriz_similaridade_global.shape[0]:
         print(f"Erro: Índice do trabalhador ({indice_trabalhador}) fora dos limites da matriz de similaridade ({matriz_similaridade_global.shape[0]}).")
         return {"erro": "Índice do trabalhador fora dos limites da matriz."}

    similaridades_vagas = matriz_similaridade_global[indice_trabalhador]

    # Verifica se similaridades_vagas é válido
    if similaridades_vagas is None or not isinstance(similaridades_vagas, np.ndarray):
        print(f"Erro: Vetor de similaridade inválido para o trabalhador {id_trabalhador}.")
        return {"erro": "Vetor de similaridade inválido."}

    # Cria pares (índice_vaga, similaridade) e ordena
    matches_ordenados = sorted(enumerate(similaridades_vagas), key=lambda item: item[1], reverse=True)

    resultado_matches = []
    for indice_vaga, similaridade in matches_ordenados[:top_n]:
        # Check for NaN or invalid similarity values
        if np.isnan(similaridade) or similaridade <= 0:
            continue
        if indice_vaga < len(vagas):
            vaga = vagas[indice_vaga]
            resultado_matches.append({
                "id_vaga": vaga['id'],
                "titulo_vaga": vaga['titulo'],
                "empresa": vaga['empresa'],
                "similaridade": round(float(similaridade), 4) # Converte para float nativo
            })
        else:
             print(f"Aviso: Índice de vaga {indice_vaga} fora dos limites da lista de vagas.")

    return resultado_matches

# Inicializa o matcher quando o módulo é carregado
# Em uma aplicação real, isso pode ser feito de forma mais robusta (ex: lazy loading)
inicializar_matcher()

