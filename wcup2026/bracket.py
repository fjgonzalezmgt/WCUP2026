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
enfrentan.
"""

from __future__ import annotations


R32_MATCHES = {
    73: (("slot", "2A"), ("slot", "2B")),
    74: (("slot", "1E"), ("third", ["A", "B", "C", "D", "F"])),
    75: (("slot", "1F"), ("slot", "2C")),
    76: (("slot", "1C"), ("slot", "2F")),
    77: (("slot", "1I"), ("third", ["C", "D", "F", "G", "H"])),
    78: (("slot", "2E"), ("slot", "2I")),
    79: (("slot", "1A"), ("third", ["C", "E", "F", "H", "I"])),
    80: (("slot", "1L"), ("third", ["E", "H", "I", "J", "K"])),
    81: (("slot", "1D"), ("third", ["B", "E", "F", "I", "J"])),
    82: (("slot", "1G"), ("third", ["A", "E", "H", "I", "J"])),
    83: (("slot", "2K"), ("slot", "2L")),
    84: (("slot", "1H"), ("slot", "2J")),
    85: (("slot", "1B"), ("third", ["E", "F", "G", "I", "J"])),
    86: (("slot", "1J"), ("slot", "2H")),
    87: (("slot", "1K"), ("third", ["D", "E", "I", "J", "L"])),
    88: (("slot", "2D"), ("slot", "2G")),
}

# Partidos oficiales calendarizados de la ronda de 32 (FIFA 2026).
# Se usan como fuente de verdad en el flujo post-grupos cuando los 32
# clasificados coinciden con estos cruces ya publicados.
R32_SCHEDULED_FIXTURES_2026 = {
    73: ("South Africa", "Canada"),
    74: ("Germany", "Paraguay"),
    75: ("Netherlands", "Morocco"),
    76: ("Brazil", "Japan"),
    77: ("France", "Sweden"),
    78: ("Cote d'Ivoire", "Norway"),
    79: ("Mexico", "Ecuador"),
    80: ("England", "Congo DR"),
    81: ("USA", "Bosnia and Herzegovina"),
    82: ("Belgium", "Senegal"),
    83: ("Portugal", "Croatia"),
    84: ("Spain", "Austria"),
    85: ("Switzerland", "Algeria"),
    86: ("Argentina", "Cabo Verde"),
    87: ("Colombia", "Ghana"),
    88: ("Australia", "Egypt"),
}

ROUND_TEMPLATES = {
    "round_of_16": {
        89: (74, 77),
        90: (73, 75),
        91: (76, 78),
        92: (79, 80),
        93: (83, 84),
        94: (81, 82),
        95: (86, 88),
        96: (85, 87),
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

