from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker


class Base(DeclarativeBase):
    pass


def _connect_args(database_url: str) -> dict[str, object]:
    if database_url.startswith("sqlite"):
        return {"check_same_thread": False}
    return {}


def init_database(database_url: str) -> sessionmaker:
    from house_price_prediction.infrastructure.db import models  # noqa: F401

    engine = create_engine(
        database_url,
        future=True,
        connect_args=_connect_args(database_url),
    )
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False)