"""Rush Hour board engine (bitmask + occupancy grid).

Ported and extended from the ``reasoning-gym`` Rush Hour implementation, which
itself mirrors michaelfogleman's authoritative Go solver semantics
(https://www.michaelfogleman.com/rush/).

Design notes
------------
* The board is a ``width x height`` grid (default 6x6).  Cell index is
  ``row * width + col`` (zero-indexed, row-major).
* ``Piece`` objects carry a 1-D ``position``, a ``size`` (>= 2), and an
  ``orientation`` (Horizontal stride = 1, Vertical stride = width).
* ``pieces[0]`` is ALWAYS the primary piece (the red car): horizontal, sitting
  on the exit row.  Its rendered label is ``A``.  The puzzle is solved when the
  red car's left cell reaches ``target`` (its row's rightmost valid position).
* Walls (immovable obstacles) are stored separately and rendered as ``x``.

This module is intentionally dependency-free (pure Python ``int`` bitmask +
boolean occupancy list) so the same engine powers the environment, the IDA*
solver oracle, and the per-seed puzzle generator without divergence.
"""

from __future__ import annotations

import re

HORIZONTAL = 0
VERTICAL = 1

DEFAULT_WIDTH = 6
DEFAULT_HEIGHT = 6
PRIMARY_ROW = 2
PRIMARY_SIZE = 2
MIN_PIECE_SIZE = 2
MAX_PIECE_SIZE = 3

# Action grammar accepted by parse_action: "A+2", "A-1", "[A+]", "[A-1]", "B +3"
_ACTION_RE = re.compile(r"\[?\s*([A-Za-z])\s*([+-])\s*(\d*)\s*\]?")


class Piece:
    """A single car/truck on the board.

    ``mask`` is a bitmask of all cells the piece occupies, kept in sync with
    ``position`` so collision tests are pure integer ANDs.
    """

    __slots__ = ("position", "size", "orientation", "stride", "mask")

    def __init__(self, position: int, size: int, orientation: int, width: int):
        self.position = position
        self.size = size
        self.orientation = orientation
        self.stride = 1 if orientation == HORIZONTAL else width
        self.mask = 0
        p = position
        for _ in range(size):
            self.mask |= 1 << p
            p += self.stride

    @property
    def fixed(self) -> bool:
        # Walls are modelled as size-1 immovable pieces; real cars have size >= 2.
        return self.size == 1

    def row(self, width: int) -> int:
        return self.position // width

    def col(self, width: int) -> int:
        return self.position % width

    def move(self, steps: int) -> None:
        d = self.stride * steps
        self.position += d
        if steps > 0:
            self.mask <<= d
        else:
            self.mask >>= -d

    def copy(self) -> "Piece":
        new = Piece.__new__(Piece)
        new.position = self.position
        new.size = self.size
        new.orientation = self.orientation
        new.stride = self.stride
        new.mask = self.mask
        return new


class RushHourBoard:
    """Mutable Rush Hour board with move enumeration and undo.

    The board is the single source of truth shared by the environment, the
    solver, and the generator.  It supports two move APIs:

    * :meth:`legal_moves` / :meth:`do_move` / :meth:`undo_move` operate on
      ``(piece_index, steps)`` and are used by the solver/generator.  ``steps``
      may be multi-cell (e.g. slide 3 left); :meth:`legal_moves` enumerates
      every reachable slide distance for each piece in both directions.
    * :meth:`apply_action` operates on a user label + signed steps (``"A"``,
      ``+2``) and is used by the environment.  It slides as far as possible up
      to the requested amount and reports how many cells actually moved.
    """

    def __init__(self, width: int, height: int, pieces: list[Piece], walls: list[int]):
        self.width = width
        self.height = height
        self.pieces = pieces
        self.walls = walls
        self.occupied = [False] * (width * height)
        for piece in pieces:
            self._set_occupied(piece, True)
        for w in walls:
            self.occupied[w] = True

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_string(cls, desc: str, width: int = DEFAULT_WIDTH, height: int | None = None) -> "RushHourBoard":
        """Build a board from a flat description string.

        Encoding: ``o`` or ``.`` = empty, ``x`` = wall, any other letter = a
        vehicle cell (cells sharing a letter form one piece).  Length must equal
        ``width * height``.  The primary (red) car is the piece whose label
        sorts first; it is placed at ``pieces[0]``.
        """
        if height is None:
            height = len(desc) // width
        if len(desc) != width * height:
            raise ValueError(
                f"board string length {len(desc)} != width*height {width * height}"
            )

        positions: dict[str, list[int]] = {}
        walls: list[int] = []
        for i, label in enumerate(desc):
            if label in (".", "o"):
                continue
            if label == "x":
                walls.append(i)
                continue
            positions.setdefault(label, []).append(i)

        labels = sorted(positions.keys())
        pieces: list[Piece] = []
        for label in labels:
            ps = positions[label]
            if len(ps) < MIN_PIECE_SIZE:
                raise ValueError(f"piece {label!r} too small (size {len(ps)})")
            if len(ps) > MAX_PIECE_SIZE:
                raise ValueError(f"piece {label!r} too large (size {len(ps)})")
            stride = ps[1] - ps[0]
            if stride not in (1, width):
                raise ValueError(f"piece {label!r} has invalid shape (stride {stride})")
            if stride == 1 and any(p // width != ps[0] // width for p in ps):
                raise ValueError(f"piece {label!r} wraps across rows")
            for k in range(2, len(ps)):
                if ps[k] - ps[k - 1] != stride:
                    raise ValueError(f"piece {label!r} is not contiguous")
            orientation = HORIZONTAL if stride == 1 else VERTICAL
            pieces.append(Piece(ps[0], len(ps), orientation, width))

        # Identify the primary piece: a horizontal, size-PRIMARY_SIZE car sitting
        # on PRIMARY_ROW.  Move it to index 0 so solver/render conventions hold.
        primary_idx = None
        for idx, piece in enumerate(pieces):
            if (
                piece.orientation == HORIZONTAL
                and piece.size == PRIMARY_SIZE
                and piece.row(width) == PRIMARY_ROW
            ):
                primary_idx = idx
                break
        if primary_idx is None:
            raise ValueError("no primary (red) car found on the exit row")
        if primary_idx != 0:
            pieces.insert(0, pieces.pop(primary_idx))

        return cls(width, height, pieces, walls)

    @classmethod
    def from_pieces(
        cls,
        width: int,
        height: int,
        primary: tuple[int, int],
        vehicles: list[tuple[int, int, int]],
        walls: list[int] | None = None,
    ) -> "RushHourBoard":
        """Build a board programmatically (used by the generator).

        Args:
            primary: ``(row, col)`` of the red car's left cell (size PRIMARY_SIZE,
                horizontal, on PRIMARY_ROW by convention but row is taken from here).
            vehicles: list of ``(position, size, orientation)`` for the other pieces.
            walls: list of wall cell indices.
        """
        pr, pc = primary
        pieces = [Piece(pr * width + pc, PRIMARY_SIZE, HORIZONTAL, width)]
        for position, size, orientation in vehicles:
            pieces.append(Piece(position, size, orientation, width))
        return cls(width, height, pieces, list(walls or []))

    def copy(self) -> "RushHourBoard":
        new = RushHourBoard.__new__(RushHourBoard)
        new.width = self.width
        new.height = self.height
        new.pieces = [p.copy() for p in self.pieces]
        new.walls = list(self.walls)
        new.occupied = list(self.occupied)
        return new

    # ------------------------------------------------------------------
    # Occupancy / geometry helpers
    # ------------------------------------------------------------------

    def _set_occupied(self, piece: Piece, value: bool) -> None:
        idx = piece.position
        for _ in range(piece.size):
            self.occupied[idx] = value
            idx += piece.stride

    @property
    def target(self) -> int:
        """Exit cell index: the red car's left cell when fully right."""
        primary = self.pieces[0]
        return (primary.row(self.width) + 1) * self.width - primary.size

    @property
    def solved(self) -> bool:
        return self.pieces[0].position == self.target

    def memo_key(self) -> tuple[int, ...]:
        """Hashable canonical state: each piece's position (walls are static)."""
        return tuple(p.position for p in self.pieces)

    # ------------------------------------------------------------------
    # Solver / generator move API: (piece_index, steps)
    # ------------------------------------------------------------------

    def legal_moves(self) -> list[tuple[int, int]]:
        """Enumerate every legal ``(piece_index, steps)`` slide.

        For each non-fixed piece, slides outward in both directions until the
        board edge or an occupied cell blocks it, yielding one move per reachable
        distance (so a car that can slide up to 3 left produces steps -1, -2, -3).
        """
        moves: list[tuple[int, int]] = []
        w, h = self.width, self.height
        for i, piece in enumerate(self.pieces):
            if piece.fixed:
                continue
            stride = piece.stride
            if piece.orientation == VERTICAL:
                y = piece.position // w
                reverse_limit = -y
                forward_limit = h - piece.size - y
            else:
                x = piece.position % w
                reverse_limit = -x
                forward_limit = w - piece.size - x

            idx = piece.position - stride
            steps = -1
            while steps >= reverse_limit:
                if self.occupied[idx]:
                    break
                moves.append((i, steps))
                idx -= stride
                steps -= 1

            idx = piece.position + piece.size * stride
            steps = 1
            while steps <= forward_limit:
                if self.occupied[idx]:
                    break
                moves.append((i, steps))
                idx += stride
                steps += 1
        return moves

    def do_move(self, piece_index: int, steps: int) -> None:
        piece = self.pieces[piece_index]
        self._set_occupied(piece, False)
        piece.move(steps)
        self._set_occupied(piece, True)

    def undo_move(self, piece_index: int, steps: int) -> None:
        self.do_move(piece_index, -steps)

    # ------------------------------------------------------------------
    # Environment action API: label + signed steps
    # ------------------------------------------------------------------

    def piece_index_for_label(self, label: str) -> int | None:
        """Map a rendered label (``A`` = red car, ``B``..) to a piece index."""
        if not label or len(label) != 1:
            return None
        idx = ord(label.upper()) - ord("A")
        if 0 <= idx < len(self.pieces) and not self.pieces[idx].fixed:
            return idx
        return None

    def apply_action(self, label: str, steps: int) -> int:
        """Slide a vehicle by EXACTLY ``steps`` cells (signed), all-or-nothing.

        Returns the number of cells actually moved, which is either the full
        requested ``steps`` (when every intermediate cell is free) or ``0``.
        Unlike a "slide as far as legal" semantics, an over-request that cannot
        be fully satisfied (blocked by the edge, a wall, or another vehicle
        before reaching ``|steps|``) moves NOTHING and returns ``0`` — the agent
        must request a distance the car can actually travel. A return of ``0``
        therefore means the move was ineffective (the board is unchanged),
        whether the car was fully blocked or the request overshot.
        """
        idx = self.piece_index_for_label(label)
        if idx is None or steps == 0:
            return 0
        piece = self.pieces[idx]
        direction = 1 if steps > 0 else -1
        stride = piece.stride
        w, h = self.width, self.height
        n = abs(steps)

        # Feasibility pre-check WITHOUT mutating the board: every one of the n
        # cells the piece would pass into must be in-bounds and free. We walk the
        # leading edge arithmetically; the cell just beyond the leading edge is
        # never one of the piece's own cells, so occupancy is position-stable and
        # we don't need to move the piece to test it.
        if direction > 0:
            if piece.orientation == HORIZONTAL:
                if piece.col(w) + piece.size + n > w:
                    return 0
            else:
                if piece.row(w) + piece.size + n > h:
                    return 0
            for k in range(n):
                if self.occupied[piece.position + (piece.size + k) * stride]:
                    return 0
        else:
            if piece.orientation == HORIZONTAL:
                if piece.col(w) - n < 0:
                    return 0
            else:
                if piece.row(w) - n < 0:
                    return 0
            for k in range(n):
                if self.occupied[piece.position - (k + 1) * stride]:
                    return 0

        # All clear: apply the full slide.
        for _ in range(n):
            self.do_move(idx, direction)
        return steps

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _label_grid(self) -> list[str]:
        grid = ["."] * (self.width * self.height)
        for w in self.walls:
            grid[w] = "x"
        for i, piece in enumerate(self.pieces):
            label = "x" if piece.fixed else chr(ord("A") + i)
            idx = piece.position
            for _ in range(piece.size):
                grid[idx] = label
                idx += piece.stride
        return grid

    def render(self, mode: str = "grid_coord") -> str:
        if mode == "grid":
            return self._render_grid()
        if mode == "coord":
            return self._render_coord()
        if mode == "grid_coord":
            return "Grid:\n" + self._render_grid() + "\n\nCoordinates:\n" + self._render_coord()
        raise ValueError(f"invalid render mode: {mode!r}")

    def _render_grid(self) -> str:
        grid = self._label_grid()
        rows = []
        for y in range(self.height):
            start = y * self.width
            rows.append(" ".join(grid[start : start + self.width]))
        # Mark the exit on the primary row with a trailing '>'.
        primary_row = self.pieces[0].row(self.width)
        rows[primary_row] = rows[primary_row] + " >"
        return "\n".join(rows)

    def _render_coord(self) -> str:
        lines = [
            f"Board size: {self.height} rows x {self.width} cols (zero-indexed; "
            f"exit at row {self.pieces[0].row(self.width)}, right edge)."
        ]
        for i, piece in enumerate(self.pieces):
            if piece.fixed:
                continue
            label = chr(ord("A") + i)
            orient = "horizontal" if piece.orientation == HORIZONTAL else "vertical"
            cells = []
            idx = piece.position
            for _ in range(piece.size):
                cells.append((idx // self.width, idx % self.width))
                idx += piece.stride
            tag = " (red car)" if i == 0 else ""
            cells_str = ", ".join(f"({r},{c})" for r, c in cells)
            lines.append(f"{label}{tag}: {orient}, cells [{cells_str}]")
        if self.walls:
            wall_cells = ", ".join(f"({w // self.width},{w % self.width})" for w in self.walls)
            lines.append(f"Walls: [{wall_cells}]")
        return "\n".join(lines)

    def to_string(self) -> str:
        """Flat board string (``o`` empty, ``x`` wall, ``A``.. pieces)."""
        grid = self._label_grid()
        return "".join("o" if ch == "." else ch for ch in grid)


def parse_action(action: str) -> tuple[str, int] | None:
    """Parse a single move action into ``(label, signed_steps)``.

    Accepts ``"A+2"``, ``"A-1"``, ``"[A+]"`` (defaults to 1 step), ``"[B-3]"``.
    The (stripped) string must be *exactly* one move and nothing else.

    Uses ``fullmatch`` rather than ``search`` so that free-form text is rejected:
    an incidental car-label + sign + digits buried in prose (e.g. ``"move car
    b-1 now"``) must NOT be silently executed as a real board move. Returns
    ``None`` whenever the whole string is not a single recognisable move.
    """
    if not action:
        return None
    match = _ACTION_RE.fullmatch(action.strip())
    if match is None:
        return None
    label = match.group(1).upper()
    sign = 1 if match.group(2) == "+" else -1
    digits = match.group(3)
    magnitude = int(digits) if digits else 1
    if magnitude == 0:
        return None
    return label, sign * magnitude
