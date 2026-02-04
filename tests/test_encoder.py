import numpy as np

from scg_vae.constants import DatasetEnum
from scg_vae.encoder import VocabularyEncoder


def test_vocabulary_encoder(adata_test_tokenize):
    adata_test_tokenize.var["feature_id"] = adata_test_tokenize.var_names.astype(str)
    encoder = VocabularyEncoder(
        adata=adata_test_tokenize,
        metadata_cells=adata_test_tokenize.obs.copy(),
        metadata_genes=adata_test_tokenize.var[["feature_id"]].copy(),
        labels=[DatasetEnum.ASSAY.value, DatasetEnum.SUSPENSION.value, "batch", "cell_type"],
        n_genes=5,
        assay_vocab_size=2,
        suspension_vocab_size=2,
    )
    encode_cell_type = ["cell_type_1", "cell_type_2", "cell_type_3"]
    encoded_cell_type = encoder.encode_metadata(encode_cell_type, label="cell_type").tolist()
    encode_batch = ["batch_0", "batch_1"]
    encoded_batch = encoder.encode_metadata(encode_batch, label="batch").tolist()
    encode_assay = ["assay_1", "assay_2"]
    encoded_assay = encoder.encode_metadata(encode_assay, label=DatasetEnum.ASSAY.value).tolist()
    encode_suspension = ["suspension_1", "suspension_2"]
    encoded_suspension = encoder.encode_metadata(encode_suspension, label=DatasetEnum.SUSPENSION.value).tolist()

    assert encoded_cell_type == [None, 0, 1]
    assert encoded_batch == [0, 1]
    assert encoded_assay == [0, 1]
    assert encoded_suspension == [0, 1]
