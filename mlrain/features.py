from enum import Enum

from .data import *
from sensai.data_transformation import DFTNormalisation, SkLearnTransformerFactoryFactory
from sensai import VectorRegressionModel
from sensai.featuregen import FeatureGeneratorRegistry, FeatureGeneratorTakeColumns, FeatureGenerator


class FeatureName(Enum):
    LOCATION_PARAMS = "location_params"
    SUNSHINE_HOURS = "sunshine_hours"
    MONTH = "month"
    WINDDIR = "wind-direction"
    HUMIDITY3PM = "humidity3pm"
    MEAN_RAIN_DAYS = "mean-rain-days"
    #MUSICAL_DEGREES = "musical_degrees"
    #MUSICAL_CATEGORIES = "musical_categories"
    #LOUDNESS = "loudness"
    #TEMPO = "tempo"
    #DURATION = "duration"
    #YEAR = "year"

class FeatureGeneratorMeanRainDays(FeatureGenerator):
    def __init__(self):
        super().__init__(normalisation_rule_template=DFTNormalisation.RuleTemplate(
            transformer_factory=SkLearnTransformerFactoryFactory.MaxAbsScaler()))
        self.col_target = COL_GEN_RAINDAYSPERYEAR
        self._y = None

    def _fit(self, x: pd.DataFrame, y: pd.DataFrame = None, ctx=None):
        df: pd.DataFrame = pd.concat([x, y], axis=1)[[COL_LOCATION, self.col_target]]
        df[self.col_target] = df[self.col_target].apply(lambda cls: 1 if cls == 1 else 0)
        gb = df.groupby(COL_LOCATION)
        s = gb.sum()[self.col_target]
        s.name = "sum"
        c = gb.count()[self.col_target]
        c.name = "cnt"
        m = s / c
        m.name = "mean"
        print("Mean", m)
        self._y = df[[self.col_target]]
        self._values = pd.concat([s, c, m], axis=1)

    def _generate(self, df: pd.DataFrame, ctx=None) -> pd.DataFrame:
        ctx: VectorRegressionModel
        is_training = ctx.is_being_fitted()

        if is_training:
            def val_t(t):
                lookup = self._values.loc[getattr(t, COL_LOCATION)]
                s = lookup["sum"] - self._y.loc[t.Index][self.col_target]
                c = lookup["cnt"] - 1
                if c == 0:
                    return np.nan
                else:
                    return s / c

            values = [val_t(t) for t in df.itertuples()]

            # clean up
            self._y = None
            self._values.drop(columns=["sum", "cnt"])
        else:
            def val_i(artist_name):
                try:
                    return self._values.loc[artist_name]["mean"]
                except KeyError:
                    return np.nan

            values = df[COL_LOCATION].apply(val_i)

        return pd.DataFrame({"mean-rain-days": values}, index=df.index)


registry = FeatureGeneratorRegistry()

#registry.register_factory(FeatureName.MUSICAL_DEGREES, lambda: FeatureGeneratorTakeColumns(COLS_MUSICAL_DEGREES,
#    normalisation_rule_template=DFTNormalisation.RuleTemplate(skip=True)))

registry.register_factory(FeatureName.LOCATION_PARAMS, lambda: FeatureGeneratorTakeColumns(COLS_FULL_LOCATION,
    categorical_feature_names=COLS_FULL_LOCATION))
registry.register_factory(FeatureName.WINDDIR, lambda: FeatureGeneratorTakeColumns(COL_WINDGUSTDIR,
    categorical_feature_names=COL_WINDGUSTDIR))
registry.register_factory(FeatureName.SUNSHINE_HOURS, lambda: FeatureGeneratorTakeColumns(COL_SUNSHINE,
    normalisation_rule_template=DFTNormalisation.RuleTemplate(
        transformer_factory=SkLearnTransformerFactoryFactory.StandardScaler())))
registry.register_factory(FeatureName.MONTH, lambda: FeatureGeneratorTakeColumns(COL_MONTH,
    normalisation_rule_template=DFTNormalisation.RuleTemplate(
        transformer_factory=SkLearnTransformerFactoryFactory.StandardScaler())))
registry.register_factory(FeatureName.HUMIDITY3PM, lambda: FeatureGeneratorTakeColumns(COL_HUMIDITY3PM,
    normalisation_rule_template=DFTNormalisation.RuleTemplate(
        transformer_factory=SkLearnTransformerFactoryFactory.StandardScaler())))
registry.register_factory(FeatureName.MEAN_RAIN_DAYS, FeatureGeneratorMeanRainDays)
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
