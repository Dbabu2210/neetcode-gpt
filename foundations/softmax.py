import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array of logits
        # Hint: subtract max(z) for numerical stability before computing exp
        # return np.round(your_answer, 4)

        maxi = np.max(z)
        e = z - maxi 
        numerator = np.exp(e)
        denominator = np.sum(numerator)
        return np.round((numerator / denominator), 4)
