import asyncio
import enum
import typing
from typing import Any, Callable, Dict, List, TypeAlias

from alphaproof.core.helper import negate_theorem
from leantree import LeanProject, LeanTactic, LeanProofState
from leantree.repl_adapter.interaction import (
    LeanEnvironmentCheckpoint,
    LeanInteractionException,
    LeanProcessException,
)
from leantree.utils import to_sync


# Observations in AlphaProof are the tactic state.
Observation: TypeAlias = LeanProofState

# Actions in AlphaProof are Lean tactics (except for special actions, to start a
# disproof, or to focus on a goal).
Action: TypeAlias = LeanTactic | str

Theorem: TypeAlias = str

TACTIC_TIMEOUT_GRACE = 6.0


class TacticDeadlineExceeded(Exception):
    """Raised when a tactic exceeds its Python-side wall-clock deadline."""


async def _apply_tactic_with_deadline(
    branch: Any,
    action: Action,
    tactic_timeout: float,
) -> list[Any]:
    """Apply a tactic without moving LeanTree to another event loop."""
    task = asyncio.create_task(
        branch.apply_tactic_async(
            action,
            timeout=round(tactic_timeout * 1000),
        )
    )
    done, _ = await asyncio.wait(
        {task},
        timeout=tactic_timeout + TACTIC_TIMEOUT_GRACE,
    )
    if task in done:
        return task.result()

    branch._env.kill_group()
    try:
        await task
    except Exception:
        pass
    raise TacticDeadlineExceeded(
        f'Tactic exceeded its '
        f'{tactic_timeout + TACTIC_TIMEOUT_GRACE:.1f}s deadline.'
    )


apply_tactic_with_deadline = to_sync(_apply_tactic_with_deadline)


class NodeType(enum.Enum):
    """Node type used by the AND-OR search tree."""
    OR = 1
    AND = 2


class State(typing.NamedTuple):
    """Environment tactic state returned after applying an action."""
    id: int
    reward: float
    observation: Observation
    terminal: bool
    num_goals: int


class Environment:
    """Lean environment for one actor."""

    def __init__(
        self,
        project: LeanProject,
        imports: tuple[str, ...] = ('Mathlib',),
    ):
        """Create a LeanTree-backed proof environment."""
        self.project = project
        self.imports = imports
        self._env = self.project.environment()
        self._env.__enter__()
        self._sent_imports = False
        self._checkpoint: LeanEnvironmentCheckpoint | None = None
        self._failed = False
        self._next_state_id = -1
        self._branches: dict[int, Any] = {}
        self._theorems: dict[int, Theorem] = {}

    def close(self) -> None:
        """Stop the underlying Lean process."""
        self._env.stop_safe()

    def reset(self) -> None:
        """Restore the imported Lean state for the next proof game."""
        if self._failed:
            self._env.stop_safe()
            self._env = self.project.environment()
            self._env.__enter__()
            self._sent_imports = False
            self._checkpoint = None
            self._failed = False
        elif self._checkpoint is not None:
            self._env.rollback_to(self._checkpoint)

        self._next_state_id = -1
        self._branches.clear()
        self._theorems.clear()

    def verify(self, lean_code: str, timeout: float) -> None:
        """Verify Lean code from the imported environment checkpoint."""
        checkpoint = self._checkpoint
        assert checkpoint is not None
        self._env.rollback_to(checkpoint)
        try:
            response = self._env.send_command(lean_code, timeout=timeout)
            if any(
                    'sorryAx' in str(message.get('data', ''))
                    for message in response.get('messages', [])
            ):
                raise LeanInteractionException('Proof depends on sorryAx.')
        except LeanProcessException:
            self._failed = True
            raise
        finally:
            self._env.rollback_to(checkpoint)

    def __enter__(self):
        return self

    def __exit__(self, *args, **kwargs):
        self.close()

    def get_next_state_id(self):
        self._next_state_id += 1
        return self._next_state_id

    def initialize(self) -> None:
        """Send configured imports once per Lean process."""
        if self._sent_imports:
            return
        try:
            for module in self.imports:
                self._env.send_command(f'import {module}', timeout=900.0)
        except LeanProcessException:
            self._failed = True
            raise
        self._sent_imports = True
        self._checkpoint = self._env.checkpoint()

    def _state_from_branch(
        self,
        branch: Any,
        reward: float = 0.0,
        theorem: Theorem | None = None,
    ) -> State:
        """Store a LeanTree proof branch and expose it as an AlphaProof state."""
        state_id = self.get_next_state_id()
        self._branches[state_id] = branch
        if theorem is not None:
            self._theorems[state_id] = theorem

        observation = branch.state
        terminal = observation.is_solved()
        return State(
            id=state_id,
            reward=reward,
            observation=observation,
            terminal=terminal,
            num_goals=0 if terminal else 1,
        )

    def _state_from_branches(self, branches: list[Any], reward: float = 0.0) -> State:
        """Store LeanTree's factorized branches as one AlphaProof state."""
        if not branches:
            state_id = self.get_next_state_id()
            self._branches[state_id] = None
            return State(
                id=state_id,
                reward=reward,
                observation=LeanProofState([]),
                terminal=True,
                num_goals=0,
            )

        if len(branches) == 1:
            return self._state_from_branch(branches[0], reward=reward)

        state_id = self.get_next_state_id()
        self._branches[state_id] = branches
        observation = LeanProofState([
            goal
            for branch in branches
            for goal in branch.state.goals
        ])
        return State(
            id=state_id,
            reward=reward,
            observation=observation,
            terminal=False,
            num_goals=len(branches),
        )

    def initial_state(self, theorem: Theorem) -> State:
        """Returns the initial tactic state."""
        self.initialize()
        try:
            branch = self._env.proof_from_sorry(theorem)
        except LeanProcessException:
            self._failed = True
            raise
        return self._state_from_branch(branch, theorem=theorem)

    def step(
        self,
        state_id: int,
        action: Action,
        tactic_timeout: float,
    ) -> State:
        """Applies the action in the given state, returns the new state."""
        if state_id not in self._branches:
            raise ValueError(f'Unknown state id: {state_id}')

        branch = self._branches[state_id]
        tactic = action.tactic if isinstance(action, LeanTactic) else action

        if tactic == 'disprove':
            if state_id not in self._theorems:
                raise ValueError('Can only disprove from an initial theorem state.')
            theorem = negate_theorem(self._theorems[state_id])
            try:
                branch = self._env.proof_from_sorry(theorem)
            except LeanProcessException:
                self._failed = True
                raise
            return self._state_from_branch(branch, theorem=theorem)

        if tactic.startswith('focus_goal '):
            branches = branch
            if not isinstance(branches, list):
                raise ValueError('Can only focus a state with multiple goals.')
            try:
                goal_index = int(tactic.removeprefix('focus_goal ').strip())
            except ValueError as exc:
                raise ValueError(f'Invalid focus action: {tactic}') from exc
            try:
                return self._state_from_branch(branches[goal_index])
            except IndexError as exc:
                raise ValueError(f'Goal index out of range: {goal_index}') from exc

        if branch is None:
            raise ValueError('Cannot apply a tactic to a terminal state.')
        if isinstance(branch, list):
            raise ValueError('Use focus_goal <i> before applying a tactic.')

        try:
            branches = apply_tactic_with_deadline(
                branch,
                action,
                tactic_timeout,
            )
        except TacticDeadlineExceeded:
            self._failed = True
            raise
        except LeanProcessException:
            self._failed = True
            raise
        except Exception as exc:
            raise ValueError(f'Invalid tactic {tactic!r}: {exc}') from exc
        return self._state_from_branches(branches, reward=-1.0)
