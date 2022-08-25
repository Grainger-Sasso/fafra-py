from typing import List


from src.risk_classification.input_metrics.metric_names import MetricNames
from src.mvp.report_generation.report_generator import ReportGenerator


class FaFRA:
    def __init__(self):
        self.custom_metric_names = tuple(
            [
                MetricNames.SIGNAL_MAGNITUDE_AREA,
                MetricNames.COEFFICIENT_OF_VARIANCE,
                MetricNames.STANDARD_DEVIATION,
                MetricNames.MEAN,
                MetricNames.SIGNAL_ENERGY,
                MetricNames.ROOT_MEAN_SQUARE
            ]
        )
        self.gait_metric_names: List[str] = [
            'PARAM:gait speed',
            'BOUTPARAM:gait symmetry index',
            'PARAM:cadence',
            'Bout Steps',
            'Bout Duration',
            'Bout N',
            'Bout Starts',
            # Additional gait params
            'PARAM:stride time',
            'PARAM:stride time asymmetry',
            'PARAM:stance time',
            'PARAM:stance time asymmetry',
            'PARAM:swing time',
            'PARAM:swing time asymmetry',
            'PARAM:step time',
            'PARAM:step time asymmetry',
            'PARAM:initial double support',
            'PARAM:initial double support asymmetry',
            'PARAM:terminal double support',
            'PARAM:terminal double support asymmetry',
            'PARAM:double support',
            'PARAM:double support asymmetry',
            'PARAM:single support',
            'PARAM:single support asymmetry',
            'PARAM:step length',
            'PARAM:step length asymmetry',
            'PARAM:stride length',
            'PARAM:stride length asymmetry',
            'PARAM:gait speed asymmetry',
            'PARAM:intra-step covariance - V',
            'PARAM:intra-stride covariance - V',
            'PARAM:harmonic ratio - V',
            'PARAM:stride SPARC',
            'BOUTPARAM:phase coordination index',
            'PARAM:intra-step covariance - V',
            'PARAM:intra-stride covariance - V',
            'PARAM:harmonic ratio - V',
            'PARAM:stride SPARC',
            'BOUTPARAM:phase coordination index'
        ]
        self.ra_model_path = ''

    def perform_risk_assessment(self, assessment_path):
        # Generate risk metrics
        mg = MetricGen()
        ra_metrics = mg.generate_ra_metrics(assessment_path)
        # Assess risk using risk model
        model = Model()
        ra_results = model.assess_fall_risk(self.ra_model_path)
        # Generate risk report
        rg = ReportGenerator()
        rg.generate_report(assessment_path, '', '', '')


class Model:
    def __init__(self):
        pass

    def assess_fall_risk(self, model_path):
        pass


class MetricGen:
    def __init__(self):
        pass

    def generate_ra_metrics(self, assessment_path):
        pass


def main():
    fafra = FaFRA()
    path = ''
    ra = fafra.perform_risk_assessment('/home/grainger/Desktop/test_risk_assessments/customers/customer_Grainger/site_Breed_Road/batch_0000000000000001_2022_08_25/assessment_0000000000000001_2022_08_25/')
