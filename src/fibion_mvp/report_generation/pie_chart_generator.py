

class SKDHPieChartGenerator:
    def __init__(self):
        pass

    def gen_pie_charts(self, skdh_results_path):
        # Read in SKDH results file (JSON format)
        # Grab activity and sleep data
        # Create pie charts for activity and sleep
        pass


def main():
    skdh_path = '/home/grainger/Desktop/skdh_testing/ml_model/input_metrics/skdh/skdh_results_20220815-171703.json'
    pcg = SKDHPieChartGenerator()
    pcg.gen_pie_charts(skdh_path)


if __name__ == '__main__':
    main()

