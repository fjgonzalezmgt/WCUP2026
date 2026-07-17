"""Estructura del cuadro de eliminacion del Mundial 2026.

Define los emparejamientos de la ronda de 32 (``R32_MATCHES``) y las
plantillas de rondas posteriores (``ROUND_TEMPLATES``) siguiendo el
calendario oficial publicado por FIFA.

Cada entrada en ``R32_MATCHES`` es un par de referencias ``(tipo, valor)``:

- ``("slot", "1A")`` indica el ganador del grupo A.
- ``("third", ["A", "B", ...])`` indica un tercero mejor de los grupos
  listados, cuya asignacion exacta se resuelve en ``simulator.assign_third_slots``.

``ROUND_TEMPLATES`` mapea el ID de cada partido de la ronda de 16, cuartos,
semifinales y final a los IDs de los dos partidos cuyos ganadores se
enfrentan. El partido por el tercer lugar se define por separado porque lo
disputan los perdedores de las semifinales.
"""

from __future__ import annotations


R32_MATCHES = {
    73: (("slot", "2A"), ("slot", "2B")),
    74: (("slot", "1C"), ("slot", "2F")),
    75: (("slot", "1E"), ("third", ["A", "B", "C", "D", "F"])),
    76: (("slot", "1F"), ("slot", "2C")),
    77: (("slot", "2E"), ("slot", "2I")),
    78: (("slot", "1I"), ("third", ["C", "D", "F", "G", "H"])),
    79: (("slot", "1A"), ("third", ["C", "E", "F", "H", "I"])),
    80: (("slot", "1L"), ("third", ["E", "H", "I", "J", "K"])),
    81: (("slot", "1G"), ("third", ["A", "E", "H", "I", "J"])),
    82: (("slot", "1D"), ("third", ["B", "E", "F", "I", "J"])),
    83: (("slot", "1H"), ("slot", "2J")),
    84: (("slot", "2K"), ("slot", "2L")),
    85: (("slot", "1B"), ("third", ["E", "F", "G", "I", "J"])),
    86: (("slot", "1J"), ("slot", "2H")),
    87: (("slot", "1K"), ("third", ["D", "E", "I", "J", "L"])),
    88: (("slot", "2D"), ("slot", "2G")),
}

THIRD_PLACE_COMBINATIONS = {
    frozenset({"B", "D", "E", "F", "I", "J", "K", "L"}): {
        75: "D",
        78: "F",
        79: "E",
        80: "K",
        81: "I",
        82: "B",
        85: "J",
        87: "L",
    }
}

ROUND_TEMPLATES = {
    "round_of_16": {
        89: (73, 76),
        90: (75, 78),
        91: (74, 77),
        92: (79, 80),
        93: (84, 83),
        94: (82, 81),
        95: (87, 86),
        96: (85, 88),
    },
    "quarterfinal": {
        97: (89, 90),
        98: (93, 94),
        99: (91, 92),
        100: (95, 96),
    },
    "semifinal": {
        101: (97, 98),
        102: (99, 100),
    },
    "final": {
        104: (101, 102),
    },
}

# Partido FIFA 103: perdedor de la semifinal 101 vs perdedor de la 102.
THIRD_PLACE_MATCH_ID = 103
THIRD_PLACE_SEMIFINALS = (101, 102)
