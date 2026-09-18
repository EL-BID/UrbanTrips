"""
Pure SQL helper utilities — no DB connections, no side effects.
"""

import re


def is_date_string(input_str: str) -> bool:
    pattern = re.compile(r"^\d{4}-\d{2}-\d{2}$")
    return bool(pattern.match(input_str))


def check_date_type(day_type: str) -> None:
    """Validate that day_type is 'weekday', 'weekend', or a YYYY-MM-DD date string."""
    if not ((day_type in ("weekday", "weekend")) or is_date_string(day_type)):
        raise Exception("dat_type debe ser `weekday`, `weekend` o fecha 'YYYY-MM-DD'")


def create_days_sql_filter(dias, col: str = "dia", prefix: str = " AND ") -> str:
    """
    Return a WHERE clause fragment restricting to a list of days.

    Empty or None means no filter: the whole run gets read. Acotar los días es
    lo que hace viable procesar un mes desde el dashboard, donde el usuario
    elige qué días le interesan en vez de barrer toda la base.
    """
    if not dias:
        return ""
    for dia in dias:
        if not is_date_string(str(dia)):
            raise Exception(f"El día '{dia}' no tiene formato 'YYYY-MM-DD'")
    dias_str = ", ".join(f"'{dia}'" for dia in sorted(set(map(str, dias))))
    return f"{prefix}{col} IN ({dias_str})"


def create_line_ids_sql_filter(line_ids) -> str:
    """Return a WHERE clause fragment filtering by id_linea."""
    if line_ids is not None:
        if isinstance(line_ids, int):
            line_ids = [line_ids]
        lines_str = ",".join(map(str, line_ids))
        return f" where id_linea in ({lines_str})"
    return " where id_linea is not NULL"


def create_branch_ids_sql_filter(branch_ids) -> str:
    """Return a WHERE clause fragment filtering by id_ramal."""
    if branch_ids is not None:
        if isinstance(branch_ids, int):
            branch_ids = [branch_ids]
        branches_str = ",".join(map(str, branch_ids))
        return f" where id_ramal in ({branches_str})"
    return " where id_ramal is not NULL"
