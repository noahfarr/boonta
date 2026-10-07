import numpy as np

from boonta.utils import Text, load_tokenizer


class Transcript:

    def __init__(self, repo_id: str, name: str = "generation", num_examples: int = 4):
        self._tokenizer = load_tokenizer(repo_id)
        self.name = name
        self._num_examples = num_examples

    def craft(self, algorithm, environment, state, logs) -> Text:
        actions = np.asarray(logs["action"])
        *_, num_envs = actions.shape
        transcript = "\n\n".join(
            f"{env}: {self._tokenizer.decode(actions[:, env].tolist())}"
            for env in range(min(self._num_examples, num_envs))
        )
        return Text(self.name, transcript)
