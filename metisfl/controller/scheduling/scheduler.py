
from abc import ABC, abstractmethod
from typing import List


class Scheduler(ABC):

    num_learners: int = 0

    def add_learner(self) -> None:
        """Add a learner to the scheduler."""
        self.num_learners += 1
        
    def remove_learner(self) -> None:
        """Remove a learner from the scheduler."""
        self.num_learners -= 1

    @abstractmethod
    def schedule(
        self,
        learner_id: str,
    ) -> List[str]:
        """Schedule the next batch of learners.

        Parameters
        ----------
        learner_id : str
            The ID of the learner to schedule.

        Returns
        -------
        List[str]
            The IDs of the learners to schedule.
        """
        pass

    @abstractmethod
    def __str__(self) -> str:
        pass
