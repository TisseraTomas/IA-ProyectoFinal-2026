"""
Ingesta del Reglamento del Alumno (CRUC IUA, Títulos I-X) -> JSON de chunks atómicos.

Pipeline 100 % determinístico (sin LLM): el texto de cada chunk se COPIA del PDF,
por lo que no puede haber resúmenes ni excepciones omitidas.

Uso:
    python -m src.ingest_reglamento --pdf data/source/Reglamento_Alumno_IUA_-_I_al_X.pdf \
                                    --out data/reglamento_chunks.json

Requiere `pdftotext` (poppler-utils) en el PATH, o bien pasar --txt con el texto ya extraído.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ROMAN = {"I": 1, "II": 2, "III": 3, "IV": 4, "V": 5, "VI": 6, "VII": 7, "VIII": 8, "IX": 9, "X": 10}
ROMAN_INV = {v: k for k, v in ROMAN.items()}

# Artículos (secciones numeradas) del reglamento, en orden, por título.
KNOWN_IDS = {
    1: ["1.1", "1.2", "1.3", "1.4", "1.5"],
    2: ["2.1", "2.2", "2.3", "2.4", "2.4.1", "2.4.1.a", "2.4.1.b", "2.4.1.c", "2.4.2", "2.5"],
    3: ["3.1", "3.2"],
    4: ["4.1", "4.2", "4.2.1", "4.2.2", "4.3", "4.3.1", "4.3.2", "4.4", "4.4.1", "4.4.2",
        "4.5", "4.6", "4.6.1", "4.6.2", "4.6.3", "4.6.4", "4.6.5", "4.6.6"],
    5: ["5.1", "5.2", "5.3"],
    6: ["6.1", "6.2", "6.3", "6.4", "6.5"],
    7: ["7.1", "7.2", "7.2.1", "7.2.2", "7.2.2.1", "7.2.2.2", "7.2.2.3", "7.2.3",
        "7.3", "7.3.1", "7.3.2", "7.4", "7.4.1", "7.4.2", "7.4.3"],
}
ALL_IDS = {i for ids in KNOWN_IDS.values() for i in ids}

# Títulos sin artículos numerados: se parten por párrafo con estos marcadores.
UNNUMBERED_TITLES = {8: "VIII", 9: "IX", 10: "X"}

# Títulos cuya introducción es contenido normativo (se indexa). El resto de las
# introducciones son frases del tipo "En el presente apartado se establecen...".
KEEP_INTRO = {4, 5}
INTRO_END = {4: r"^Condiciones estudiantiles en la carrera y sus"}  # corta el índice-tabla

ACT = ("actuaciones", r"^Actuaciones habilitadas")
_PAIR = ("requisitos_para_rendir", r"^Requisitos para rendir")
_MODAL = ("modalidad_y_nota", r"^Modalidad de evaluación")
# id -> (nombre de la primera parte, [(nombre, regex de inicio de la parte), ...])
SPLITS = {
    "2.3": ("condiciones", [("legalizacion_documentacion", r"^La documentación expedida"),
                            ("tribunal_y_decision", r"^El estudio de equivalencias")]),
    "2.4.1.c": ("documentacion", [("proceso_de_evaluacion", r"^La documentación debe ser entregada")]),
    "4.1": ("requisitos", [ACT, ("convenio_cero_materias", r"^Los estudiantes con convenio")]),
    **{i: ("situacion_y_requisitos", [ACT]) for i in
       ["4.2.1", "4.2.2", "4.3.1", "4.3.2", "4.4.1", "4.4.2", "4.5",
        "4.6.1", "4.6.2", "4.6.3", "4.6.4", "4.6.5"]},
    "7.2.2.1": ("condiciones_regularidad", [("plazo_vigencia", r"^Plazo de vigencia"),
                                            ("perdida_regularidad", r"^Situaciones que generan la pérdida")]),
    "7.2.2.2": ("condiciones_promocion", [("plazo_vigencia", r"^Plazo de vigencia"),
                                          ("perdida_categoria", r"^Pérdida de la categoría")]),
    "7.4.1": ("requisitos_inscripcion", [_PAIR, _MODAL]),
    "7.4.2": ("requisitos", [_MODAL]),  # "rigen iguales requisitos..." queda junto a la inscripción
    "7.4.3": ("topes_maximos", [("requisitos", r"^Requisitos de inscripción"), _MODAL]),
    "VIII": ("p1", [("p2", r"^El estudiante o un familiar"), ("p3", r"^El CRUC IUA contempla")]),
    "IX": ("p1", [("p2", r"^Los estudiantes civiles"), ("p3", r"^Es aplicable a todos"),
                  ("p4", r"^Las sanciones disciplinarias")]),
    "X": ("p1", [("p2", r"^La Facultad dará curso"), ("p3", r"^El egresado podrá")]),
}

TITLE_RE = re.compile(r"^T[ÍI]TULO\s+([IVX]+)\s*[–-]\s*(.*)$")
ID_RE = re.compile(r"^(\d+(?:\.\d+)*(?:\.[a-z])?)\.?\s*(.*)$")
NOISE = {"", ".", "🔍"}


# --------------------------------------------------------------------------- utilidades
def clean_line(line: str) -> str:
    """Quita form-feed e íconos 🔍; zero-width space -> espacio; colapsa espacios."""
    line = line.replace("\f", "").replace("\u200b", " ").replace("🔍", "")
    return re.sub(r"[ \t]+", " ", line).strip()


def extract_text(pdf: Path) -> str:
    out = subprocess.run(["pdftotext", "-raw", str(pdf), "-"], capture_output=True, text=True, check=True)
    return out.stdout


ROW_RE = re.compile(
    r"^(Actuaciones habilitadas|Solicitar (reincorporación|licencias|baja|cambio de carrera)|"
    r"Reinscripción anual|Inscripción/reinscripción a materias|Cursar materias|"
    r"Rendir (actividades evaluativas|trabajo final)|Regularizar/promocionar|"
    r"Inscripción (y rendir finales|a trabajo final)|Trámites (de egreso|administrativos|académicos)|"
    r"Participar (como Ayudante|en actividades de representación)|Recibir becas)"
)


ITEM_RE = re.compile(r"^[a-z]( y [a-z])?\)")  # incisos a), b), 'a y b)'


def join_wrapped(lines: list[str], table: bool = False) -> str:
    """Une líneas cortadas por el ancho de página; conserva saltos de bullets/párrafos.

    - Texto normal: se une sólo si la línea previa es 'larga', no cierra la oración y la
      siguiente continúa en minúscula/dígito/paréntesis.
    - Tablas de actuaciones: cada fila empieza con una etiqueta conocida; lo demás es continuación.
    """
    out: list[str] = []
    for ln in lines:
        if table:
            join = bool(out) and not ROW_RE.match(ln)
        else:
            join = (bool(out) and len(out[-1]) >= 78
                    and not out[-1].endswith((".", ":", ";", "?", "✅", "❌"))
                    and not ITEM_RE.match(ln)
                    and (ln[0].islower() or ln[0] in "(“\"" or ln[0].isdigit()))
        if join:
            out[-1] += " " + ln
        else:
            out.append(ln)
    return "\n".join(out)


def collapse(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


# --------------------------------------------------------------------------- parseo
def parse(raw: str):
    lines = [clean_line(l) for l in raw.split("\n")]

    # 1) ubicar TÍTULOS y candidatos a encabezado
    titles: list[tuple[int, int, str]] = []  # (idx, num, nombre)
    cands: dict[str, list[int]] = {}
    cur = 0
    for i, ln in enumerate(lines):
        m = TITLE_RE.match(ln)
        if m:
            cur = ROMAN[m.group(1)]
            titles.append((i, cur, f"TÍTULO {m.group(1)} – {m.group(2).strip()}"))
            continue
        m = ID_RE.match(ln)
        if m and m.group(1) in ALL_IDS and int(m.group(1).split(".")[0]) == cur:
            cands.setdefault(m.group(1), []).append(i)

    # 2) el encabezado real es la ÚLTIMA aparición (las anteriores son el índice del título)
    anchors = {sid: idxs[-1] for sid, idxs in cands.items()}
    missing = ALL_IDS - set(anchors)
    if missing:
        raise SystemExit(f"Encabezados no encontrados: {sorted(missing)}")

    title_of = {}
    for k, (i, num, name) in enumerate(titles):
        end = titles[k + 1][0] if k + 1 < len(titles) else len(lines)
        title_of[num] = (i, end, name)

    def is_toc_line(i: int, sid: str) -> bool:
        """Línea de índice residual: id repetido, no es el ancla y el resto es un nombre corto."""
        m = ID_RE.match(lines[i])
        if not m or m.group(1) not in ALL_IDS:
            return False
        rest = m.group(2)
        return i != anchors[m.group(1)] and len(rest.split()) <= 5 and "–" not in rest

    # 3) construir regiones (ancla -> siguiente ancla/título)
    regions = []  # (titulo_num, id, nombre, parte0, lines)
    for num, (tstart, tend, tname) in title_of.items():
        own = sorted((anchors[s], s) for s in KNOWN_IDS.get(num, []))
        if num in UNNUMBERED_TITLES:
            body = lines[tstart + 1:tend]
            regions.append((num, UNNUMBERED_TITLES[num], tname.split("–", 1)[1].strip(), body))
            continue
        # introducción del título
        first = own[0][0]
        if num in KEEP_INTRO:
            body = lines[tstart + 1:first]
            if num in INTRO_END:
                cut = next(i for i, l in enumerate(body) if re.match(INTRO_END[num], l))
                body = body[:cut]
            regions.append((num, ROMAN_INV[num] + "-intro", "Introducción", body))
        for k, (idx, sid) in enumerate(own):
            nxt = own[k + 1][0] if k + 1 < len(own) else tend
            head = lines[idx]
            name = ID_RE.match(head).group(2).strip()
            start = idx + 1
            if name.count("(") > name.count(")"):  # nombre en 2 líneas (art. 2.3)
                name += " " + lines[idx + 1]
                start += 1
            regions.append((num, sid, name, lines[start:nxt]))

    # 4) limpiar ruido/índices, partir en partes y armar chunks
    chunks = []
    for num, sid, name, body in regions:
        body = [l for j, l in enumerate(body) if l not in NOISE]
        body = [l for l in body if not _toc_residue(l, anchors, lines, is_toc_line)]
        if not body:
            continue
        first_name, splits = SPLITS.get(sid, (None, []))
        parts = _split(sid, body, first_name, splits)
        for pname, plines in parts:
            if not plines:
                continue
            text = join_wrapped(plines, table=(pname == "actuaciones"))
            chunks.append(_make_chunk(num, sid, name, pname, text, title_of[num][2]))
    return chunks, lines, title_of, anchors


def _toc_residue(line, anchors, lines, is_toc_line) -> bool:
    m = ID_RE.match(line)
    if not m or m.group(1) not in ALL_IDS:
        return False
    rest = m.group(2)
    # Es residuo de índice si el resto es un nombre corto (<=5 palabras, sin guion largo).
    # Las líneas con contenido (p. ej. 7.2.2.1 "– Exige el cumplimiento...") se conservan.
    return len(rest.split()) <= 5 and "–" not in rest and not rest.rstrip().endswith((".", ","))


def _split(sid, body, first_name, splits):
    if not splits:
        return [(None, body)]
    cuts = []
    for pname, rx in splits:
        idx = next((i for i, l in enumerate(body) if re.match(rx, l)), None)
        if idx is None:
            raise SystemExit(f"Marcador no encontrado en {sid}: {rx}")
        cuts.append((idx, pname))
    cuts.sort()
    parts, prev, prev_name = [], 0, first_name
    for idx, pname in cuts:
        parts.append((prev_name, body[prev:idx]))
        prev, prev_name = idx, pname
    parts.append((prev_name, body[prev:]))
    return parts


def _make_chunk(num, sid, name, parte, text, tname):
    titulo_rom = ROMAN_INV[num]
    es_numerado = num not in UNNUMBERED_TITLES and not sid.endswith("-intro")
    articulo = sid if es_numerado else titulo_rom
    cid = f"art_{sid}" if es_numerado else f"tit_{titulo_rom}" if sid == titulo_rom else f"tit_{titulo_rom}_intro"
    if parte:
        cid += f"__{parte}"
    ctx = f"{tname} > {sid if es_numerado else titulo_rom}. {name}" + (f" [{parte.replace('_', ' ')}]" if parte else "")
    return {
        "id": cid,
        "titulo_numero": num,
        "titulo": tname,
        "articulo": articulo,
        "articulo_nombre": name,
        "parte": parte,
        "texto_literal": text,
        "texto_indexable": f"{ctx}\n{text}",
    }


# --------------------------------------------------------------------------- verificación
def verify(chunks, lines, title_of, anchors) -> list[str]:
    """Cada línea de contenido del PDF debe estar (textualmente) en algún chunk.

    Se excluyen explícitamente: encabezados, índices repetidos de cada título y las
    introducciones de relleno ("En el presente apartado se establecen...").
    """
    blob = collapse(" ".join(c["texto_literal"] for c in chunks))
    skip = set(anchors.values())
    for sid, idx in anchors.items():  # nombre de encabezado que sigue en la línea siguiente
        if lines[idx].count("(") > lines[idx].count(")"):
            skip.add(idx + 1)
    for num, (tstart, tend, _) in title_of.items():
        ids = [anchors[s] for s in KNOWN_IDS.get(num, [])]
        if not ids:
            continue
        first = min(ids)
        if num in INTRO_END:  # T4: sólo se saltea la tabla-índice posterior al texto
            cut = next(i for i in range(tstart + 1, first) if re.match(INTRO_END[num], lines[i]))
            skip.update(range(cut, first))
        elif num not in KEEP_INTRO:
            skip.update(range(tstart, first))
        else:  # T5: el índice son líneas con id repetido (se detectan abajo)
            pass
    problems = []
    for i, ln in enumerate(lines):
        if ln in NOISE or TITLE_RE.match(ln) or i in skip:
            continue
        m = ID_RE.match(ln)
        if m and m.group(1) in ALL_IDS and len(m.group(2).split()) <= 5 and "–" not in m.group(2):
            continue  # línea de índice residual
        if collapse(ln) not in blob:
            problems.append(f"línea {i}: {ln[:90]}")
    ids = [c["id"] for c in chunks]
    dup = {x for x in ids if ids.count(x) > 1}
    if dup:
        problems.append(f"ids duplicados: {sorted(dup)}")
    return problems


def _is_heading(ln: str) -> bool:
    m = ID_RE.match(ln)
    return bool(m and m.group(1) in ALL_IDS)


# --------------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", type=Path)
    ap.add_argument("--txt", type=Path, help="texto ya extraído con pdftotext -raw")
    ap.add_argument("--out", type=Path, default=Path("data/reglamento_chunks.json"))
    a = ap.parse_args(argv)
    raw = a.txt.read_text(encoding="utf-8") if a.txt else extract_text(a.pdf)
    chunks, lines, title_of, anchors = parse(raw)
    problems = verify(chunks, lines, title_of, anchors)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{len(chunks)} chunks -> {a.out}")
    if problems:
        print("ATENCIÓN, líneas del PDF que no quedaron en ningún chunk:")
        print("\n".join(problems))
        sys.exit(1)
    print("Verificación OK: todo el texto normativo del PDF está en los chunks.")


if __name__ == "__main__":
    main()