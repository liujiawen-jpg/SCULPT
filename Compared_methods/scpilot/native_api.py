"""Connect an already-trained native network to native scPILOT inference."""
from __future__ import annotations


def model_view(plan, genes):
    import anndata as ad
    import numpy as np
    import pandas as pd
    from scpilot.egd_model import EGD_model

    # This attaches the current trained network without initializing or loading
    # an alternative prediction head. predict_target_distribution provides the
    # legal single-target source atlas at each inference request.
    model = EGD_model.__new__(EGD_model)
    model.adata = ad.AnnData(np.zeros((1,len(genes)),dtype="f4"),
        var=pd.DataFrame(index=genes))
    model.module = plan.module
    model.n_latent = int(plan.module.n_latent)
    model.device = plan.module.device
    model.is_trained_ = True
    return model
