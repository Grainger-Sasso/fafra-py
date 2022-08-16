import json
from matplotlib import pyplot as plt


class SKDHPieChartGenerator:
    def __init__(self):
        pass

    def gen_pie_charts(self, skdh_results_path):
        # Read in SKDH results file (JSON format)
        skdh_results_data = self.read_json(skdh_results_path)
        # Grab activity and sleep data
        activity_data = skdh_results_data['act_metrics']
        sleep_data = skdh_results_data['sleep_metrics']
        # Create pie charts for activity and sleep
        self.create_act_pie_chart(activity_data)
        print('yeah pie')
        pass

    def create_act_pie_chart(self, act_data):
        total_time_min = act_data['N hours'][0] * 60.0
        wear_time_min = act_data['N wear hours'][0] * 60.0
        no_data_min = total_time_min - wear_time_min
        sed_time_min = act_data['wake sed 5s epoch [min]'][0]
        light_time_min = act_data['wake light 5s epoch [min]'][0]
        mod_time_min = act_data['wake mod 5s epoch [min]'][0]
        vig_time_min = act_data['wake vig 5s epoch [min]'][0]
        act_times = [
            sed_time_min,
            light_time_min,
            mod_time_min,
            vig_time_min
        ]
        labels = [
            'Sedentary time',
            'Light activity time',
            'Moderate activity time',
            'Vigorous activity time'
        ]
        explode = [0.1, 0.1, 0.1, 0.1]
        colors = ['aliceblue', 'skyblue', 'dodgerblue', 'steelblue']
        if no_data_min > 0.0:
            act_times.append(no_data_min)
            labels.append('No data')
            explode.append(0.1)
            colors.append('silver')
        act_times = [time / total_time_min for time in act_times]
        fig1, ax1 = plt.subplots()
        ax1.pie(act_times, labels=labels, autopct='%1.1f%%', startangle=90, explode=explode, shadow=True, colors=colors)
        ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        plt.show()

    def create_sleep_pie_chart(self, sleep_data):
        pass

    def read_json(self, path):
        with open(path, 'r') as f:
            data = json.load(f)
        return data


def main():
    skdh_path = '/home/grainger/Desktop/skdh_testing/ml_model/input_metrics/skdh/skdh_results_20220815-171703.json'
    pcg = SKDHPieChartGenerator()
    pcg.gen_pie_charts(skdh_path)


if __name__ == '__main__':
    main()

