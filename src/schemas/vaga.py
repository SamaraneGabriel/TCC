from typing import Optional
from pydantic import BaseModel


class Requisito(BaseModel):
    descricao: str
    obrigatorio: bool = True


class VagaEstruturada(BaseModel):
    titulo: str
    empresa: Optional[str] = None
    localidade: Optional[str] = None
    modelo_trabalho: Optional[str] = None  # "remoto", "híbrido", "presencial"

    area: Optional[str] = None  # "Engenharia de Software", "Data Science", etc.
    senioridade: Optional[str] = None  # "junior", "pleno", "senior"

    descricao: Optional[str] = None
    responsabilidades: list[str] = []

    skills_obrigatorias: list[str] = []
    skills_desejadas: list[str] = []
    requisitos: list[Requisito] = []

    experiencia_minima_anos: Optional[float] = None
    formacao_minima: Optional[str] = None  # "graduação", "pós-graduação"

    faixa_salarial: Optional[str] = None
    beneficios: list[str] = []
    regime_contratacao: Optional[str] = None  # "CLT", "PJ", "estágio"
