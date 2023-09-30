
""" MetisFL Synchronous Scheduler """


from typing import List

from loguru import logger
from metisfl.controller.scheduling.scheduler import Scheduler


class SynchronousScheduler(Scheduler):

    learner_ids: List[str] = []
    global_iteration: int = 0

    def schedule(
        self,
        learner_id: str,
    ) -> List[str]:
        """Schedule the next batch of learners, synchronously.

        Parameters
        ----------
        learner_id : str
            The ID of the learner to schedule.

        Returns
        -------
        List[str]
            The IDs of the learners to schedule.
        """

        self.learner_ids.append(learner_id)

        if len(self.learner_ids) < self.num_learners:
            return []

        self.global_iteration += 1

        to_schedule = self.learner_ids.copy()

        logger.info(
            f"Starting Federation Round {self.global_iteration} with {len(to_schedule)} learners."
        )

        return to_schedule

    def __str__(self) -> str:
        return "SynchronousScheduler"
