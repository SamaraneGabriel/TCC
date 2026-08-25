from typing import Optional
from pydantic import BaseModel


class Experiencia(BaseModel):
    cargo: str
    empresa: str
    duracao_meses: Optional[int] = None
    descricao: Optional[str] = None
    skills_utilizadas: list[str] = []


class Formacao(BaseModel):
    grau: str  # "graduação", "pós-graduação", "mestrado", "doutorado", "técnico"
    curso: str
    instituicao: str
    ano_conclusao: Optional[int] = None
    em_andamento: bool = False


class Certificacao(BaseModel):
    nome: str
    instituicao: Optional[str] = None
    ano: Optional[int] = None


class CurriculoEstruturado(BaseModel):
    nome: str
    email: Optional[str] = None
    telefone: Optional[str] = None
    localidade: Optional[str] = None

    resumo_profissional: Optional[str] = None

    senioridade_estimada: Optional[str] = None  # "junior", "pleno", "senior", "especialista"
    anos_experiencia_estimados: Optional[float] = None
    area_principal: Optional[str] = None

    skills_tecnicas: list[str] = []
    skills_comportamentais: list[str] = []
    idiomas: list[str] = []

    experiencias: list[Experiencia] = []
    formacoes: list[Formacao] = []
    certificacoes: list[Certificacao] = []

    pretensao_salarial: Optional[str] = None
    disponibilidade: Optional[str] = None
    modelo_trabalho_preferido: Optional[str] = None  # "remoto", "híbrido", "presencial"
