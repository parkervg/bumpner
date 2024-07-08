from guidance.models import LlamaCpp
from bumpner import Bumpner
import time

if __name__ == "__main__":
    model = LlamaCpp(
        './ggml-model-Q4_0.gguf',
        n_ctx=1028,
        echo=False,

    )
    bumpner = Bumpner(
        model,
        """
        PERSON:
          description: People, including fictional.

        PRODUCT:
          description: Products offered by a company.

        ORG:
          description: Companies, agencies, institutions, etc.

        IDNUMBER:
          description: Identifier for Aperture Science employees
        """,
    )
    start = time.time()
    result = bumpner("""
    I work at Aperature Science with Mike.
    We work on cool products like the portal gun together. his num is s23ahg.
    """
    )
    print(f"Took {time.time() - start} seconds")
    result.visualize()
