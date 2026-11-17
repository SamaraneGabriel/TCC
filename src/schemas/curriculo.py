from typing import Optional
from pydantic import BaseModel, field_validator


class Experiencia(BaseModel):
    cargo: Optional[str] = None
    empresa: Optional[str] = None
    duracao_meses: Optional[int] = None
    descricao: Optional[str] = None
    skills_utilizadas: list[str] = []

    @field_validator("duracao_meses", mode="before")
    @classmethod
    def _arredondar_duracao(cls, v):
        # O LLM devolve frações quando o currículo diz "um mês e meio" — arredonda em vez de rejeitar.
        return round(v) if isinstance(v, float) else v


class Formacao(BaseModel):
    grau: Optional[str] = None  # "graduação", "pós-graduação", "mestrado", "doutorado", "técnico"
    curso: Optional[str] = None
    instituicao: Optional[str] = None
    ano_conclusao: Optional[int] = None
    em_andamento: bool = False


class Certificacao(BaseModel):
    nome: Optional[str] = None
    instituicao: Optional[str] = None
    ano: Optional[int] = None


class CurriculoEstruturado(BaseModel):
    nome: Optional[str] = None
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
