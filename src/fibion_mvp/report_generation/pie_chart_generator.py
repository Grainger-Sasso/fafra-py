import os
import time
import json
from matplotlib import pyplot as plt


class SKDHPlotGenerator:
    def __init__(self):
        pass

    def gen_skdh_plots(self, skdh_results_path, output_path):
        # Read in skdh results
        skdh_results_data = self.read_json(skdh_results_path)
        # Generate pie charts
        act_pie_path, sleep_pie_path = self.gen_pie_charts(skdh_results_data, output_path)
        # Generate activity heat maps
        return act_pie_path, sleep_pie_path

    def gen_pie_charts(self, skdh_results_data, output_path):
        # Grab activity and sleep data
        activity_data = skdh_results_data['act_metrics']
        sleep_data = skdh_results_data['sleep_metrics']
        # Create pie charts for activity and sleep
        act_fig, act_plot = self.create_act_pie_chart(activity_data)
        sleep_fig, sleep_plot = self.create_sleep_pie_chart(sleep_data)
        metric_file_name = 'model_input_metrics_' + time.strftime("%Y%m%d-%H%M%S") + '.json'
        act_plot_path = os.path.join(output_path, str('activity_pie_chart_' + time.strftime("%Y%m%d-%H%M%S") + '.png'))
        sleep_plot_path = os.path.join(output_path, str('sleep_pie_chart_' + time.strftime("%Y%m%d-%H%M%S") + '.png'))
        act_fig.savefig(act_plot_path)
        sleep_fig.savefig(sleep_plot_path)
        return act_plot_path, sleep_plot_path

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
        colors = ['powderblue', 'skyblue', 'dodgerblue', 'steelblue']
        if no_data_min > 0.0:
            act_times.append(no_data_min)
            labels.append('No data')
            explode.append(0.1)
            colors.append('silver')
        act_times = [time / total_time_min for time in act_times]
        fig1, ax1 = plt.subplots()
        ax1.pie(act_times, labels=labels, autopct='%1.1f%%', startangle=90, explode=explode, shadow=True, colors=colors, textprops={'fontsize': 16})
        ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        return fig1, ax1

    def create_sleep_pie_chart(self, sleep_data):
        sleep_percent = sleep_data['percent time asleep'][0]
        non_sleep_percnet = 100.0 - sleep_percent
        percentages = [
            sleep_percent
        ]
        labels = [
            'Percent time asleep'
        ]
        colors = [
            'slateblue'
        ]
        explode = [
            0.1
        ]
        if non_sleep_percnet > 0.0:
            percentages.append(non_sleep_percnet)
            labels.append('Percent time awake')
            colors.append('darkslateblue')
            explode.append(0.1)
        fig1, ax1 = plt.subplots()
        ax1.pie(percentages, labels=labels, autopct='%1.1f%%', startangle=90, explode=explode, shadow=True, colors=colors, textprops={'fontsize': 16})
        ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        return fig1, ax1

    def read_json(self, path):
        with open(path, 'r') as f:
            data = json.load(f)
        return data


def main():
    skdh_path = '/home/grainger/Desktop/skdh_testing/ml_model/input_metrics/skdh/skdh_results_20220815-171703.json'
    output_path = '/home/grainger/Desktop/skdh_testing/fafra_results/reports/pie_charts/'
    pcg = SKDHPlotGenerator()
    pcg.gen_skdh_plots(skdh_path, output_path)


if __name__ == '__main__':
    main()

