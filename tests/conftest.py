import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from sklearn.preprocessing import LabelEncoder

from scg_vae.constants import DatasetEnum


@pytest.fixture
def adata_test_tokenize():
    X = sparse.csr_matrix(
        [
            [1, 0, 0, 2, 0],
            [0, 0, 1, 0, 0],
            [3, 2, 1, 2, 1],  # with ties, sorting gives precedence to index
        ]
    )
    var_names = np.array(["gene1", "gene2", "gene3", "gene4", "gene5"])
    n_cells = X.shape[0]
    rng = np.random.default_rng(42)
    adata = ad.AnnData(
        X=X,
        obs=pd.DataFrame(
            {
                "batch": rng.choice(["batch_0", "batch_1"], size=n_cells),
                "cell_type": rng.choice(["cell_type_1", "cell_type_2", "cell_type_3"], size=n_cells),
                DatasetEnum.SUSPENSION.value: rng.choice(["suspension_1", "suspension_2"], size=n_cells),
                DatasetEnum.ASSAY.value: rng.choice(["assay_1", "assay_2"], size=n_cells),
            }
        ),
    )
    adata.var_names = var_names
    return adata


@pytest.fixture
def anndatas(tmp_path):
    """Create mock anndata files for testing."""
    n_cells = 100
    n_genes = 50
    n_batches = 3
    rng = np.random.default_rng(42)

    # Create temporary anndata files
    adata_paths = []
    for i in range(n_batches):
        X = sparse.random(n_cells, n_genes, density=0.1, format="csr")
        obs = pd.DataFrame(
            {
                "batch": [i] * n_cells,
                "cell_type": rng.choice(["cell_type_1", "cell_type_2", "cell_type_3"], size=n_cells),
            }
        )
        adata = ad.AnnData(X=X, obs=obs)
        adata.var_names = [f"gene_{j}" for j in range(n_genes)]
        path = tmp_path / f"adata_{i}.h5ad"
        adata.write_h5ad(path)
        adata_paths.append(str(path))

    label_encoder = LabelEncoder()
    label_encoder.fit(obs["cell_type"])

    return adata_paths, n_cells, n_genes, label_encoder
