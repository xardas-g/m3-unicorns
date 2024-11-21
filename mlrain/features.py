from enum import Enum

from .data import *
from sensai.data_transformation import DFTNormalisation, SkLearnTransformerFactoryFactory
from sensai.featuregen import FeatureGeneratorRegistry, FeatureGeneratorTakeColumns


class FeatureName(Enum):
    LOCATION_PARAMS = "location_params"
    SUNSHINE_HOURS = "sunshine_hours"
    #MUSICAL_DEGREES = "musical_degrees"
    #MUSICAL_CATEGORIES = "musical_categories"
    #LOUDNESS = "loudness"
    #TEMPO = "tempo"
    #DURATION = "duration"
    #YEAR = "year"


registry = FeatureGeneratorRegistry()

#registry.register_factory(FeatureName.MUSICAL_DEGREES, lambda: FeatureGeneratorTakeColumns(COLS_MUSICAL_DEGREES,
#    normalisation_rule_template=DFTNormalisation.RuleTemplate(skip=True)))

registry.register_factory(FeatureName.LOCATION_PARAMS, lambda: FeatureGeneratorTakeColumns(COLS_FULL_LOCATION,
    categorical_feature_names=COLS_FULL_LOCATION))
registry.register_factory(FeatureName.SUNSHINE_HOURS, lambda: FeatureGeneratorTakeColumns(COL_SUNSHINE,
    normalisation_rule_template=DFTNormalisation.RuleTemplate(
        transformer_factory=SkLearnTransformerFactoryFactory.StandardScaler())))

#registry.register_factory(FeatureName.LOUDNESS, lambda: FeatureGeneratorTakeColumns(COL_LOUDNESS,
#    normalisation_rule_template=DFTNormalisation.RuleTemplate(
#        transformer_factory=SkLearnTransformerFactoryFactory.StandardScaler())))

#registry.register_factory(FeatureName.TEMPO, lambda: FeatureGeneratorTakeColumns(COL_TEMPO,
#    normalisation_rule_template=DFTNormalisation.RuleTemplate(
#        transformer_factory=SkLearnTransformerFactoryFactory.StandardScaler())))

#registry.register_factory(FeatureName.DURATION, lambda: FeatureGeneratorTakeColumns(COL_DURATION_MS,
#    normalisation_rule_template=DFTNormalisation.RuleTemplate(
#        transformer_factory=SkLearnTransformerFactoryFactory.StandardScaler())))

#registry.register_factory(FeatureName.YEAR, lambda: FeatureGeneratorTakeColumns(COL_YEAR,
#    normalisation_rule_template=DFTNormalisation.RuleTemplate(
#        transformer_factory=SkLearnTransformerFactoryFactory.StandardScaler())))
