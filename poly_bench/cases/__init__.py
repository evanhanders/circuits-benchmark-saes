from .poly_case import PolyCase, PolyBenchDataset
from .duplicate_remover import HighLevelDuplicateRemover, DuplicateRemoverDataset
from .left_greater import HighLevelLeftGreater, LeftGreaterDataset
from .paren_checker import HighLevelParensBalanceChecker, BalancedParensDataset
from .unique_extractor import HighLevelUniqueExtractor, UniqueExtractorDataset

str_to_model_dict: dict[str, type[PolyCase]] = {
    str(HighLevelDuplicateRemover()): HighLevelDuplicateRemover,
    str(HighLevelLeftGreater()): HighLevelLeftGreater,
    str(HighLevelParensBalanceChecker()): HighLevelParensBalanceChecker,
    str(HighLevelUniqueExtractor()): HighLevelUniqueExtractor
}

str_to_dataset_dict: dict[str, type[PolyBenchDataset]] = {
    str(HighLevelDuplicateRemover()): DuplicateRemoverDataset,
    str(HighLevelLeftGreater()): LeftGreaterDataset,
    str(HighLevelParensBalanceChecker()): BalancedParensDataset,
    str(HighLevelUniqueExtractor()): UniqueExtractorDataset
}